# cute_06 · 完整 GEMM：从 naive 到对标 cuBLAS 的 Hopper 流水线

> 前置：cute_04（TMA + 多 stage + mbarrier）、cute_05（MMA atom + WGMMA）。
> 这一章把前面所有零件拧成一台机器。每一版只引入一个新概念，全部可独立运行、
> 和 CPU 参考逐点比对、并报告 TFLOP/s。

---

## §0 路线图：从零件到机器

cute_04 和 cute_05 给了你全部零件：

| 零件 | 在哪学的 | 干什么 |
|---|---|---|
| swizzle smem layout | cute_04 §3 | smem 里怎么摆不撞 bank |
| TMA | cute_04 §5 | 一个线程描述整块，硬件搬 gmem→smem |
| mbarrier + PipelineState | cute_04 §6 | 按字节数的 producer/consumer 同步 |
| MMA atom / TiledMMA | cute_05 §1-2 | 一条 Tensor Core 指令怎么发 |
| WGMMA | cute_05 §3 | Hopper 的 warpgroup 级异步 MMA |

cute_06 的答案：**一个 GEMM 是什么**。六版，每一版只换一块：

```
v0  naive         没有 smem, 每个 CTA 直接从 gmem 读
v1  smem 单缓冲   A/B 先进 smem, K 循环只读 smem      (+5x)
v2  cp.async      Ampere 流水线: 搬 k+1 和算 k 重叠   (+小)
v3  TMA+WGMMA     Hopper 流水线: 两个硬件异步引擎     (+6x)
v4  Warp Spec     producer 只搬 / consumer 只算        (诚实的: 未必更快)
capstone cluster  把 CTA 组成小队                       (诚实的: 未必更快)
                    ──
cuBLAS 参考        ~878 TFLOP/s (2048^3 本机实测)
```

> 数字都是本机（H200, sm_90a）实测。v0 到 v3 是巨大的台阶，v4 和 capstone
> 是"概念上更高级、工程上未必更快"的诚实展示 —— 这一章最重要的训练是
> **知道每个优化在什么条件下才赢**。

---

## §1 v0 —— naive：先搭骨架

跑 `cute_gemm_v0.cu`。它没有任何 smem、没有任何流水线，每个 CTA：

```
  1. 从 gmem 读 A/B 的 tile 进寄存器 (fragment)
  2. 用 TiledMMA 算 C 的一块
  3. 写回 gmem
```

骨架是后面五版的**共同结构**：

```cpp
auto gA = local_tile(mA, Shape<Int<BM>,Int<BK>>{}, make_coord(blockIdx.y, _));
auto gB = local_tile(mB, Shape<Int<BN>,Int<BK>>{}, make_coord(blockIdx.x, _));
auto gC = local_tile(mC, Shape<Int<BM>,Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

auto tCrC = thr.partition_fragment_C(gC);
clear(tCrC);

for (int k = 0; k < nk; ++k) {
    copy(thr.partition_A(gA(_,_,k)), thr.partition_fragment_A(gA(_,_,k)));  // 搬
    copy(thr.partition_B(gB(_,_,k)), thr.partition_fragment_B(gB(_,_,k)));
    gemm(mma, tArA, tBrB, tCrC);                                            // 算
}
copy(tCrC, thr.partition_C(gC));                                            // 写回
```

**三个致命伤**（全部来自访存）：

1. 每个 CTA 只算 `64×8`，gmem 读几乎没合并（B 的 8 行每行只有 8 列）。
2. A/B 每个元素被大量 CTA 重复读：一个 A 元素被 `N/8` 个 CTA 读。
3. 没有 smem = 没有跨 K 循环的复用。

实测（2048³）：**~11.7 TFLOP/s**。这个数字是后面所有版本的"起点"。

---

## §2 v1 —— smem 中转：每个 CTA 只从 gmem 读一次

经典解法：**先把 A/B 的一块搬进 smem，之后 K 循环只读 smem**。

```
        gmem                          smem                寄存器
   A[BM×BK] ──cp.async──▶ ┌─────────┐ ──ldmatrix──▶ fragment ──mma──▶ 累加器
   B[BN×BK] ──cp.async──▶ │ swizzle │ ──ldmatrix──▶ fragment ─┘
                          └─────────┘
```

### §2.2 关键：ldmatrix 配 swizzle

smem → 寄存器用 **ldmatrix**（`SM75_U32x4_LDSM_N`）—— 一条指令从 smem 取
`4×8` half 装进 4 个寄存器。它天生配 swizzle 布局，因为 swizzle 就是
为了让"4 个线程的 4 字节"不撞 bank。

**一个必须知道的坑**：ldmatrix 的 TiledCopy 必须**用 MMA 定制**：

```cpp
auto mma = make_tiled_mma(SM80_16x8x16_TN{}, Layout<Shape<_2,_2>>{},
                          Tile<_32,_32,_16>{});   // ← 排列，不是形状！
auto s2r = make_tiled_copy_A(make_ldmatrix(), mma);
cute::copy(s2r, tlA.partition_S(sA), tlA.retile_D(tCrA));  // 注意是 retile_D
```

- **`Tile<_32,_32,_16>`** 把 4 个 `16×8×16` 原子重排成 `32×32×16` 的大原子，
  ldmatrix 的线程映射才能对上。缺了它编译期报 `TiledCopy uses too few vals`。
- **`retile_D`** 不是 `partition_D`：fragment 已经是"我这份"寄存器，不能再按线程切，
  只需重排成 ldmatrix 期望的形状。

同步还是 `__syncthreads`（所有线程既搬又算，单缓冲）。实测（2048³）：
**~67 TFLOP/s** —— 比 v0 快 **5.7×**，因为 A/B 每块只从 gmem 读一次。

---

## §3 v2 —— 多 stage cp.async：Ampere 的流水线

v1 的毛病：搬和算串行，每轮两个 `__syncthreads`，两边各闲一半。

Ampere 的解法是 **cp.async + 多 stage**：

```
cp.async 是异步的: 发出指令就返回, 不占寄存器等数据
-> 提前把 k+1, k+2 的搬发出去, 在它们飞的时候算 k

smem 开成 N 份 (stage): 搬的写 stage[i], 算的读 stage[j], 轮转
```

```
prologue: 发 STAGES-1 批 (灌满)
mainloop:
  (a) 发下一批:  copy(tAgA(_,_,k_next), tAsA(_,_,k_pipe_write))   // 异步
  (b) 等第 k 批: cp_async_wait<0>(); __syncthreads()
  (c) 算第 k 批: ldmatrix + gemm
```

这一版 = 官方 `wgmma_sm90.cu` 的 SM80 版。实测（2048³，STAGES=3）：
**~71 TFLOP/s** —— 比 v1 只快一点点。

> 为什么没大跳？因为 64×64 的 tile 太小，计算本身就少，cp.async 的重叠空间有限。
> **这本身就是一条重要的教训**：流水线只能隐藏延迟，不能创造吞吐。
> 真正的大跳要等 v3 —— Hopper 把指令密度和搬运都换成硬件引擎。

---

## §4 v3 —— TMA + WGMMA + mbarrier：Hopper 的正解

Hopper 把 v2 的两个短板都换成了**硬件异步引擎**：

| | v2 (Ampere) | v3 (Hopper) |
|---|---|---|
| gmem→smem | cp.async，每线程发 load | **TMA**，1 个 lane 发 1 条指令 |
| smem→寄存器 | ldmatrix，每线程搬 | **没有**，WGMMA 硬件直接读 smem |
| 计算 | SM80 MMA 16×8 | **WGMMA** 64×64 |
| 同步 | cp_async_wait + `__syncthreads` | **mbarrier**，按字节数 |

```
prologue: producer 把 STAGES 批全发出去
mainloop (双 state 独立转):
  consumer:  ProducerBar::wait(producer_mbar[read])   // 等这批到齐
             WGMMA 四句
             ConsumerBar::arrive(consumer_mbar[read]) // 释放
             ++read_state
  producer:  ConsumerBar::wait(consumer_mbar[write])  // 等这批空了
             arrive_and_expect_tx + 发 TMA
             ++write_state
```

这是 cute_04 §6 跑通的单 CTA 多 stage，铺满整个 grid。两个 `PipelineState`
**都从 0 开始**，不要在 prologue 里预推进（会死锁）。

注意 128×128×64 × 3 stage = 96KB > 48KB 静态上限，所以用 `extern __shared__` +
`cudaFuncSetAttribute`（cute_04 §6.3 的台阶）。

实测（2048³）：**~428 TFLOP/s** —— 比 v2 快 **6×**。这就是 Hopper 的兑现。

---

## §5 v4 —— Warp Specialization：谁搬，谁算

v3 的 mainloop 里，128 个线程**既发 TMA 又做 WGMMA**。发 TMA 的只有 1 个 lane，
其余 127 个在等；然后 128 个一起做 WGMMA。这就是"线程分工"的问题。

Warp Specialization 的答案：**把线程分成两组，各干各的**。

```
                256 线程 = 2 个 warpgroup
  ┌───────────────────────────────┬───────────────────────────────┐
  │ wg0 (warps 0-3) = consumer    │ wg1 (warps 4-7) = producer    │
  │ 只做 WGMMA                    │ 只发 TMA                      │
  │ 不关心数据从哪来               │ 一路抢跑, 把流水线灌满         │
  └───────────────────────────────┴───────────────────────────────┘
              ▲ 用 mbarrier 通信 ▲ (就是 v3 那两组 barrier)
```

两个手写 WS 的坑（cute_04 §6.4 踩过）：

- **consumer 必须落在 wg0**（warps 0-3）。反过来 warpgroup_arrive/commit 会出问题。
- producer 的 TMA 只能由 1 个 lane 发，且必须限定在 producer 组：
  `elect_one_sync()` 是**每个 warp 选一个** → 要 `(one && warp == PRODUCER_WARP)`。

### §5.3 诚实的实测

手写最小 WS（2048³）：**~393 TFLOP/s**，比 v3（428）**慢**。

为什么？三个原因，都是真实工程：

1. 256 线程 = 2 个 warpgroup 挤一个 SM，占用率降了。
2. consumer 只有 128 线程算，producer 在 128×128 tile 上抢跑空间不大
   （TMA 本来就够快）。
3. 真正的 WS 收益靠 **setmaxnreg**（producer 少给寄存器）+ 更大的 tile +
   更细的 epilogue 分工 —— 那需要 CUTLASS 全套。

**所以**：概念上 WS 是对的（谁搬谁算分开），工程上要全套才兑现。这正是
CUTLASS 官方 WS kernel 结构复杂的原因。这一版的价值是让你**看到分工的骨架**，
不是让你手写一个比 CUTLASS 快的。

---

## §6 capstone —— Block Cluster：把 CTA 组成小队

v0-v4 都是"一个 CTA 独立算一块"。capstone 加最后一块拼图：**Block Cluster**。

```
cluster: 把几个 CTA 绑定成一组, 保证同时驻留在同一个 SM 上
  - 更小的调度粒度
  - cluster_sync (比 __syncthreads 高一层)
  - 为 TMA multicast 铺路
```

launch 时多一个 cluster 维度：

```cpp
dim3 cluster_dims(2, 2);   // cluster 形状 (x, y)
cutlass::ClusterLaunchParams params{grid, block, cluster_dims, smem_bytes};
cutlass::launch_kernel_on_cluster(params, kptr, ...);
```

kernel 内两个新 API：

```cpp
int cluster_local = cute::block_rank_in_cluster();  // 0..CLUSTER_M*CLUSTER_N-1
cute::cluster_sync();                                // 全 cluster 栅栏
```

### §6.2 那 multicast 呢？（为什么这一版没有）

**原理**：在 2×2 cluster 里，同一行（cy 相同）的两个 CTA 读**同一块 A**，
同一列的两个读同一块 B。非 multicast 时每个 CTA 各搬各的，A 的 gmem/L2 流量
是 2 倍。TMA multicast 让一个 CTA 发一次，硬件把同一块数据同时写进多个
CTA 的 smem —— gmem 只读一次。

**为什么这一版没上**：hand-rolled multicast 需要 TMA TiledCopy 的全套 machinery
（`make_tma_copy` + `get_slice` + `partition_S/D`），很容易在 rank/坐标上出错，
是 CUTLASS `sm90_mma_tma_gmma_ss_warpspecialized.hpp` 几百行的活。
这一版先把 cluster 机制讲干净，multicast 的原理和它在 CUTLASS 里的用法
见上面的官方文件。

### §6.3 诚实的实测

cluster 版（2048³）：**~290 TFLOP/s**，比 v3（428）慢。因为 4 个 CTA 挤一个
SM 争抢资源，tile 又小。**cluster 不是免费的** —— 它的价值在 multicast 和
跨 CTA 协作（例如 fp8 的 scale 交换、split-K 的归约），不是单跑变快。

### 汇总

```
 版本                         TFLOP/s (2048^3)
 v0 naive                      ~11.7
 v1 smem 单缓冲                ~66.9
 v2 cp.async 3-stage           ~70.6
 v3 TMA+WGMMA 3-stage          ~428
 v4 Warp Spec (手写最小)       ~393
 capstone cluster              ~290
 cuBLAS fp16 (参考)            ~878
```

### 为什么追不上 cuBLAS（以及为什么正常）

1. **tile 调度**：cuBLAS 用 swizzled rasterization 让相邻 CTA 吃相邻 tile，
   L2 复用最大化。我们 v0-v4 都是最朴素的 row-major grid。
2. **epilogue**：cuBLAS 用 TMA store + 融合算子（scale、bias、activation），
   我们只是普通 copy。
3. **setmaxnreg + 完整 WS**：寄存器再分配。
4. **多精度 / split-K / 各种 kernel 选择**：cuBLAS 是几十个 kernel 的库。

但这条路径（TMA+WGMMA+pipeline+cluster）**就是 CUTLASS 的路径**。你学的是
怎么走这条路，不是重新发明 cuBLAS。

---

## §7 练习

跑 `make ex`，把 `exercises/ex.cu` 里的 TODO 填掉，再跑 `make ex`，
每题填对会打印 PASS。解答在 `exercises/solutions.md`。

| # | 主题 | 对应 |
|---|---|---|
| 1 | 从 fragment 数出 MMA 的覆盖范围 | §1 |
| 2 | 为什么 naive 慢（A/B 被重复读几次） | §1.3 |
| 3 | 改错：ldmatrix 缺 `Tile` 排列 | §2.2 |
| 4 | 手写 cp.async 双缓冲的 prologue | §3 |
| 5 | 手写 TMA+WGMMA mainloop 的 barrier 部分 | §4 |
| 6 | 手写一个 cluster 的最小 kernel | §6 |

---

## §8 代码地图 & 下一步

```
cute_06_gemm_basic/
├── common.h               # 填矩阵 / CPU GEMM / 比对 / TFLOP/s
├── cute_gemm_v0.cu        # §1  naive: 骨架
├── cute_gemm_v1.cu        # §2  smem 中转 (Ampere 单缓冲)
├── cute_gemm_v2.cu        # §3  多 stage cp.async (Ampere 上限)
├── cute_gemm_v3.cu        # §4  TMA+WGMMA+mbarrier (Hopper 正解)
├── cute_gemm_v4.cu        # §5  Warp Specialization
├── cute_gemm_capstone.cu  # §6  Block Cluster
├── exercises/             # §7  6 道练习 + solutions.md
├── README.md
└── Makefile               # make all / make run / make ex
```

**下一步 (cute_07)**：capstone 里"tile 调度"是追不上 cuBLAS 的头号原因。
cute_07 讲 **Persistent Block**：把 grid 固定成 SM 个数，每个 CTA 循环吃多个
tile —— 这才能做 swizzled rasterization、L2 复用、和真正的 tile 调度。

---

> **记法**：TFLOP/s = 2·M·N·K / 1e12 / ms。`2` 是因为一次乘加算两个 flop。
> 所有数字都是本机（H200, sm_90a, CUDA 12.8, CUTLASS 4.6）实测，
> 共享机器上跑会有 ±10% 波动，以你机器打印为准。
