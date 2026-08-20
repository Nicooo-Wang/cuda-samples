# cute_05 练习解答

题面在 `ex.cu`，README 里的对应节号写在每道题开头的注释里。

---

## 1 —— 从 TV 布局手算一个线程拿几个元素（§1.3）

LayoutA/B/C_TV 把"线程个数 × 每股元素"展平。每题只数个数：

- **B**：`8×16` half = 128 元素 ÷ 32 线程 = **4 个/线程**。
- **C**：`16×8` float = 128 元素 ÷ 32 线程 = **4 个/线程**。（累加器必须进寄存器，
  所以 C 一定分到线程手里）

```cpp
int b_expect = 4;   // ①
int c_expect = 4;   // ②
```

核对 `LayoutB_TV: ((_4,_8),(_2,_2))` → 线程 4×8，每线程 `2×2=4` 个 ✓。
`LayoutC_TV: ((_4,_8),(_2,_2))` → 同理 4 个 ✓。

---

## 2 —— fragment 是寄存器还是描述符（§3.2）

WGMMA 下 A/B 的 fragment 是描述符，不是寄存器：

```cpp
auto tAsA = thr.partition_A(sA);      // ① partition_A
auto tArA = thr.make_fragment_A(tAsA); // ② make_fragment_A
```

关键观察：`print(tArA)` 开头是 `GMMA::DescriptorIterator`，而
`partition_fragment_C` 是 `ptr[32b]`（真数组，32 个 float/线程）。
所以"WGMMA 不占寄存器"只对 **A/B** 成立，**累加器 C 仍然是真寄存器**，
每条 WGMMA 都得在寄存器里攒结果。

---

## 3 —— 手写 WGMMA 四句（§3.3）

缺的是最后一句：

```cpp
warpgroup_wait<0>();
```

`gemm()` 是异步的，`commit_batch()` 只是打包，**必须 wait 才能保证累加器写完了**。
少它：写回 C 时读到的是没算完的寄存器 → 结果错且难查（因为有时序依赖）。

---

## 4 —— 修 WGMMA 的 TiledMMA 陷阱（§5.1）

SM90 的原子 `SM90_64x64x16_SS` 已经要求 128 线程（一个 warpgroup），
所以 `make_tiled_mma` 的第二个参数在这里是"用几个 warpgroup"而不是"几个 warp"。
`Layout<Shape<_2,_1,_1>>{}` 会得 `size==256`，要求两个 warpgroup ——
blockDim=128 时半个 tile 没人算，后果是**静默的错**。

正确写法是**裸原子**，让 CuTe 自动重复：

```cpp
auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
```

`partition_fragment_C(128×64)` 会自动给出 `MMA_M = _2` —— 同一个 warpgroup
沿 M 把指令发两次。线程数仍是 128，每个线程 64 个 float。

> 对比 SM80 的 `SM80_16x8x16_TN`：它只要 32 线程，所以第二个参数真的要说是
> "2×2 个 warp"。这两代的反差正是 §5.1 强调的。

---

## 5 —— 手写一段 TMA 搬运（§4.2）

三个 TODO：

```cpp
// ① 坐标 tensor (条件 1)
auto mA = tma_a.get_tma_tensor(make_shape(Int<128>{}, Int<64>{}));

// ② 事务字节数 (条件 5 配套): A 是 128x64 half
constexpr int ntrans = int(sizeof(half_t)) * 128 * 64;   // = 16384

// ③ 发 TMA
copy(tma_a.with(bar[0]), tAgA(_, 0), tAsA(_, Int<0>{}));
```

要点：

- `get_tma_tensor` 返回的是**坐标** tensor，`copy` 会取 `tAgA(_, k)` 的坐标元组，
  不拿普通 gmem tensor 去传，否则报 `explode_tuple`。
- **只有 1 个 lane 发**：`if (warp == 0 && one)`。`elect_one_sync` 是每 warp 选一个，
  跨 warpgroup 必须再限定 leader warp（这里只有 1 个 warpgroup，所以够了）。
- `arrive_and_expect_tx(bar, ntrans)` 必须在 `copy` 前，mbarrier 才知道这一轮等多少字节。

---

## 6 —— 单 CTA 的 TMA + WGMMA GEMM（§5）

练习 5 的 TMA 通路 + 练习 3 的 WGMMA 四句，外面套 k 循环。
关键是对齐两处：

1. **mbarrier phase 每轮翻转**：`Bar::wait(&bar[0], k & 1)`。第一次等 phase-0，
   第二次等 phase-1，第三次回到 phase-0……这就是"单缓冲 + phase bit"在裸 mbarrier
   上的样子。`k & 1` 只对 stage=2 起效；stage 更多要 `k % STAGES`（cute_06 v3 会做）。
2. **每轮算完要 `__syncthreads()`**：等所有人都读完 smem，下一轮 TMA 才能覆盖。
   这就是单缓冲的代价 —— 也正是 cute_04 §6 用多 stage 要消除的东西。

完整的 mainloop：

```cpp
for (int k = 0; k < NK; ++k) {
    if (warp == 0 && one) {
        Bar::arrive_and_expect_tx(&bar[0], txb);
        copy(ta.with(bar[0]), tAgA(_, k), tAsA(_, Int<0>{}));
        copy(tb.with(bar[0]), tBgB(_, k), tBsB(_, Int<0>{}));
    }
    Bar::wait(&bar[0], k & 1);
    auto tCrA = thr.make_fragment_A(thr.partition_A(sA2));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB2));
    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    __syncthreads();
}
```
