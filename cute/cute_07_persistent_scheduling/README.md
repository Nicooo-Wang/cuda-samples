# cute_07 · Persistent Block 与 tile 调度：让 GEMM 吃满整个 SM

> 前置：cute_06（TMA+WGMMA 完整 GEMM）。
> 这一章讲**最后一个大概念**：不是"怎么搬怎么算"，而是"一个 CTA 活多久、
> 每个 CTA 吃哪些 tile"。它与访存-计算流水线正交 —— 你可以给 cute_06 的
> GEMM 套一层 persistent 循环，不改任何搬运/计算代码。这层正交性正是
> 它独立成章的原因。

---

## §0 路线图：为什么还需要一章

cute_06 的 GEMM 是"每个 tile 一个 CTA"：

```
grid = (N/BN, M/BM)   ← 和 tile 数一样多
每个 CTA 算一块 C 就结束
```

这有三个问题：

1. **每个 CTA 都有 launch/初始化/收尾的开销**（barrier init、fence、prologue）。
2. **tile 的顺序由硬件决定**（一般 row-major），我们控制不了 —— 没法做调度。
3. 反复 launch 的间隙，SM 可能有空档。

Persistent kernel 换个思路：

```
grid = SM 个数 (132 on H200)
每个 CTA 常驻, 循环吃 tile:
  while (还有我的 tile) {
    拿坐标 -> 搬 -> 算 -> 写回
  }
```

```
cute_06:  grid = 256 CTAs, 每个活 1 个 tile
cute_07:  grid = 132 CTAs, 每个活 ~2 个 tile (2048^3)

  ┌─ CTA 0 ─┐  ┌─ CTA 1 ─┐         ┌─ CTA 0 (常驻) ──────────┐
  │ tile 0   │  │ tile 1   │         │ tile 0 → tile 2 → tile 4 │
  └─────────┘  └─────────┘         └───────────────────────────┘
  ↑ 每次 launch 都要: 初始化 barrier、fence、等 SM 空闲
```

---

## §1 v0 —— Persistent 结构：一个 CTA 活到 grid 结束

跑 `cute_gemm_p0.cu`。和 cute_06 v3 的差别只有两处：

```cpp
// ★1  tile 一维编号 -> (by, bx) 坐标
int by = tile_id / num_tiles_n;
int bx = tile_id % num_tiles_n;

// ★2  persistent 主循环
for (int tile_id = blockIdx.x; tile_id < num_tiles; tile_id += gridDim.x) {
    ... 就是 cute_06 v3 的 TMA + WGMMA + PipelineState ...
}
```

### §1.2 一个必须处理的坑：barrier 的 phase 跨 tile 要重置

每个 tile 是一段独立的 GEMM：`PipelineState` 从 0 开始，但 barrier 的 phase
在上一个 tile 结束时已经翻转过了。**不重置就死锁**：

```cpp
for (int s = 0; s < STAGES; ++s) {
    ProducerBar::init(&producer_mbar[s], 1);
    ConsumerBar::init(&consumer_mbar[s], NTHR);
}
cutlass::arch::fence_barrier_init();
__syncthreads();   // 所有人看到重置后的 barrier 再开始这个 tile
```

这是 persistent 的"隐藏成本"：每 tile 一次全 block 栅栏 + barrier 重置。
tile 越大（K 越长），这个成本摊得越薄。

### §1.3 实测

```
512x512x512:    0.014 ms   19.4 TFLOP/s   (和 cute_06 v3 差不多)
2048x2048x2048: 0.055 ms  312.5 TFLOP/s
```

和 v3（421 TFLOP/s）比略低 —— 因为 grid=132 < v3 的 grid=256，每 CTA 的 tile
更多，加上每 tile 的 barrier 重置。**persistent 不是免费的**，它的价值在调度权。

---

## §2 v1 —— Tile 调度：rasterization 与 L2 复用

跑 `cute_gemm_p1.cu`。v0 的 tile 分配是 `CTA i 吃 tile_id = i, i+grid, i+2*grid`，
tile_id→(by,bx) 是 row-major。问题：

```
row-major 时, 同一时刻的 CTA 吃:
  CTA 0 吃 (0,0), CTA 1 吃 (0,1), ..., CTA 15 吃 (0,15)
  CTA 16 吃 (1,0)  ← 此时 CTA 0-15 可能还没吃完 (0,*)!

A 的复用 (同一行 CTA 共享同一块 A) 跨了"代" —— 时间上错开, L2 留不住。
```

**rasterization**（tile 调度）的目标：让同一时刻在跑的 CTA 吃**空间上相邻**
的 tile，它们共享的 A/B 块在时间上靠近，L2 命中率上升。

```
row-major:                     swizzled (块大小 4):
 0→(0,0)  1→(0,1)  2→(0,2)     0→(0,0)  1→(1,0)  2→(2,0)  3→(3,0)
 4→(0,4)  5→(0,5)  6→(0,6)     4→(0,1)  5→(1,1)  6→(2,1)  7→(3,1)
 ...                           ...
 相邻 tile_id 沿行走           相邻 tile_id 聚成 4x4 方块
```

### §2.3 实测（诚实的）

```
2048 row-major  311.7 TFLOP/s
2048 swizzled   308.6 TFLOP/s   ← 没差别
```

为什么没差别？**A+B = 17MB，L2 = 63MB** —— 全部装得下，怎么调度都命中。
rasterization 只有在 **A+B >> L2** 时才可能有用（见 capstone 的 8192 实测，
以及那里更诚实的结论）。

---

## §3 capstone —— 扫尺寸看 L2 压力

跑 `cute_gemm_capstone.cu`。固定 persistent + 两种调度，扫多个尺寸。

```
L2 = 63 MB (本机)
2048^3: A+B = 17 MB  < L2 -> 调度无所谓
4096^3: A+B = 67 MB  ~= L2 -> 临界
8192^3: A+B = 268 MB > L2 -> 理论上有调度空间
```

### §3.3 诚实的实测结论

```
2048 row-major  313.4 TFLOP/s    2048 swizzled  307.1
8192 row-major  273.4 TFLOP/s    8192 swizzled  265.8
```

**简化 swizzle 没赢过 row-major**。三个原因，都是真实工程：

1. **负载均衡 > L2**：132 CTA 吃 4096 个 tile（8192^3），每 CTA ~31 个，
   tile 大小的波动（最后一批不够分）比 L2 收益大。
2. **简化 swizzle 不够好**：只聚了 M 方向 4 个 tile，2D 空间局部性远不如
   CUTLASS 的 bit-reversal + tile-block 调度（那是几百行调度代码）。
3. 差距在噪声内（±2%）。

**正确的结论**：

1. **persistent 结构**是 CUTLASS/cuBLAS 的标准（grid = SM 数）。
2. **rasterization 的收益**要在 CUTLASS 级的调度复杂度下才兑现 ——
   这也是 cuBLAS 是"库"而不是"一个 kernel"的原因。
3. 想优化 L2，**先看 L2 大小和 A+B 的比值**，再决定要不要花力气。

---

## §4 练习

跑 `make ex`，把 `exercises/ex.cu` 里的 TODO 填掉。解答在 `exercises/solutions.md`。

| # | 主题 | 对应 |
|---|---|---|
| 1 | tile 编号 → 坐标的两种映射 | §2.1 |
| 2 | persistent 主循环的边界条件 | §1.1 |
| 3 | 手写 barrier phase 重置 | §1.2 |

---

## §5 代码地图 & 系列收尾

```
cute_07_persistent_scheduling/
├── common.h
├── cute_gemm_p0.cu        # §1  persistent 结构 (grid = SM 数)
├── cute_gemm_p1.cu        # §2  rasterization (row-major vs swizzled)
├── cute_gemm_capstone.cu  # §3  尺寸扫描 + L2 分析
├── exercises/             # §4  3 道练习 + solutions.md
├── README.md
└── Makefile
```

到这里，cute 系列全部讲完：

| 章 | 主题 | 你得到了什么 |
|---|---|---|
| 01-02 | Layout / Tensor | CuTe 的坐标语言 |
| 03 | Copy Atom | 搬运的语义 |
| 04 | Swizzle / TMA / Multi-stage | smem 摆法 + 硬件搬运 + 流水线 |
| 05 | MMA Atom / WGMMA | 计算引擎 |
| 06 | 完整 GEMM + WS + Cluster | 把引擎拧成机器 |
| 07 | Persistent + 调度 | 机器的生命周期 |

**你手里已经有 CUTLASS 的全部核心概念。** 下一步的三种走法：
读 CUTLASS 源码（`sm90_mma_tma_gmma_ss_warpspecialized.hpp` 现在应该
能读懂了）、用 CUTLASS 库写自己的算子、或者去读 cuBLAS 的 kernel 选择逻辑
（它就是一个巨大的调度器）。

---

> **记法**：persistent kernel = grid 固定为 SM 数、CTA 循环吃 tile 的 kernel。
> rasterization = 决定"哪个 CTA 先吃哪个 tile"的调度策略，目标是空间局部性。
