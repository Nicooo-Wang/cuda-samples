# cute_06 练习解答

题面在 `ex.cu`，README 里的对应节号写在每道题开头的注释里。

---

## 1 —— 从 fragment 数出 MMA 覆盖范围（§1）

`make_tiled_mma(atom, Layout<Shape<_4,_1,_1>>{})`：

- 线程数 = 4 个 warp = **128**。
- M 方向覆盖 = `16 × 4` = **64**（沿 M 重复 4 次）。

```cpp
int threads_expect = 128;
int m_cover = 64;
```

> 为什么从线程排布看而不是 `partition_fragment_C`？fragment 是**每个线程自己那份**
> （一个 16×8 原子的切片），`size<1>(fragC)` 恒为 1。覆盖范围是"整块有多大"
> 的问题，要从 TiledMMA 的线程排布（沿 M/N 重复几次）看。

---

## 2 —— naive 慢在哪：A/B 被重复读几次（§1.3）

一个 A 元素 `A[m][k]` 属于 M 方向的第 `m/BM` 行 CTA，但被 N 方向所有 CTA
（`0..N/BN-1`）使用。所以重复读次数 = `N/BN`。

BN=8 时：`2048/8 = 256` 次。BN=64 时：`2048/64 = 32` 次。

```cpp
int reuse_big = 32;
```

> 这就是"大 tile 的价值"：BN 越大，A 被重读的次数越少。但 tile 太大 smem 装不下、
> CTA 数太少。真实 GEMM 在 128~256 之间权衡（见 README §6 的 cuBLAS 差距分析）。

---

## 3 —— 改错：ldmatrix 缺 Tile 排列（§2.2）

`make_tiled_mma` 缺第三个参数（排列）：

```cpp
auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2>>{},
                          Tile<_32, _32, _16>{});
```

`Tile<_32,_32,_16>` 把 4 个 16×8×16 原子重排成 32×32×16 的"大原子"，ldmatrix
（一次取 4×8 half）的线程映射才能对齐。缺了它，`make_tiled_copy_A(ldm, mma)`
报 `TiledCopy uses too few vals`。

---

## 4 —— 手写 cp.async 双缓冲的 prologue（§3）

STAGES=2 时 prologue 发 `STAGES-1 = 1` 批，填进 stage 0：

```cpp
for (int s = 0; s < STAGES - 1; ++s) {
    cute::copy(tc.partition_S(gA(_, _, k_tile)), tc.partition_D(sA(_, _, s)));
    cute::copy(tc.partition_S(gB(_, _, k_tile)), tc.partition_D(sB(_, _, s)));
    cp_async_fence();
    ++k_tile;
}
```

要点：`cp_async_fence()` 保证 cp.async 的可见性（别让后面的指令重排到它前面）；
`++k_tile` 让 mainloop 从第 1 块开始算（第 0 块已经在飞）。

---

## 5 —— 手写 TMA+WGMMA 的 barrier 同步（§4）

单 k tile（K=BK），一次 TMA + 一次 WGMMA：

```cpp
// producer: 1 个 lane 声明字节数 + 发 TMA
if (warp == 0 && one) {
    Bar::arrive_and_expect_tx(&bar[0], txb);
    copy(ta.with(bar[0]), tAgA(_, 0), tAsA(_, Int<0>{}));
    copy(tb.with(bar[0]), tBgB(_, 0), tBsB(_, Int<0>{}));
}

// consumer: 等这批到齐 (单次, phase = 0)
Bar::wait(&bar[0], 0);
```

要点：

- `arrive_and_expect_tx(bar, txb)` **必须在 copy 前**：mbarrier 需要知道这一轮
  等多少字节（`txb = A 字节 + B 字节`）。
- 单次 TMA phase 恒 0。多 stage 的 phase 翻转（`(k/STAGES) & 1`）是 cute_04 §5 的事。

---

## 6 —— cluster 最小 kernel（§6）

```cpp
int rank = cute::block_rank_in_cluster();  // 0..CLUSTER_M*CLUSTER_N-1
cute::cluster_sync();
```

输出里每个 CTA 打印自己的 rank（0,1,2,3），`cluster_sync()` 保证所有 CTA 的
"after sync" 都出现在任意一个 "before" 之后 —— 这就是全 cluster 栅栏。
对比 `__syncthreads`（只同步一个 CTA 内），cluster_sync 同步的是 cluster 里
所有 CTA。
