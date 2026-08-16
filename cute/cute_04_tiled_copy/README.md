# Section 04: Smem Layout, Bank Conflict, Swizzle, 协作式 Copy Atom

## 本章要解决的问题

Section 03 教会了我们怎么用 `TiledCopy` 把 gmem 数据搬进来。但它一直在搬的是一维连续数组。真实的 GEMM kernel 要做的是一件更微妙的事：先把 gmem 的数据搬进 **shared memory（smem）**，然后从 smem 喂给 Tensor Core。

这里有两处独立的"效率陷阱"：

1. **smem 的摆放方式决定读写速度。** smem 有 32 个 bank，每次访问一个 bank 序列。如果一个 warp 的 32 个 lane 全撞到同一个 bank，硬件要拆成 32 次，带宽损失 32 倍。标量代码一不小心就会触发，但 CuTe 的 layout 让我们能在 **host 上可视化**这件事，而且能在不改 kernel 代码的前提下直接换掉 layout 做对比。

2. **Tensor Core 有一套人类读不懂的线程↔数据映射。** `ldmatrix` 这个指令负责从 smem 装载 MMA 需要的 fragment，但 32 个线程各自拿到的那 8 个 half 是哪些元素，完全不符合任何直觉顺序（thr0 拿 0, 1, 128, 129, 8, 9, 136, 137）。手写这个映射几乎必错，而且错了不报错只算出错的结果。`make_tiled_copy_A` + `retile_D` 是让 MMA 自己说出"我要什么排布"再去喂数据，规避这个问题的唯一可靠方式。

本章讲清楚四件事：**smem bank 模型**（§1–2）、**Swizzle 是什么及为何它是 Hopper 上的唯一出路**（§3–4）、**协作式 atom ldmatrix**（§5–6）、以及 **make_tiled_copy_A 和 retile_D 的分工**（§7）。capstone 用矩阵转置把 §1–4 的理论变成可以量化的数字。

---

## §1 smem 的 32 个 bank

smem 按 **4 字节（一个 float）** 为单位轮流分配给 32 个 bank：

```
float 下标 :  0   1   2  ...  31 | 32  33  34  ...
bank      :  0   1   2  ...  31 |  0   1   2  ...
```

规则只有一条：

> 一个 warp 的 32 个 lane，如果各自落在 32 个**不同** bank → 这次访存一个周期完成。  
> 如果 N 个 lane 落在**同一个** bank → 硬件串行化，拆成 N 次，即 **N-way conflict**。

这是硬件限制，无法绕过，只能用 layout 把它避开。

**哪种访问会触发冲突？** 以一个 `(32,32):(32,1)` 的 float tile（row-major）为例：

```
按行读 s(0, 0..31):  偏移 0, 1, 2, ..., 31  → bank 0, 1, 2, ..., 31  → 无冲突
按列读 s(0..31, 0):  偏移 0, 32, 64, ...,992 → bank 0, 0, 0, ..., 0   → 32-way conflict
```

根源在于：行 stride = 32 = bank 数，所以下一行同一列的偏移 = 偏移 + 32，恰好绕回同一个 bank。这不是凑巧，是任何 stride 等于 32 倍数的 float layout 必然面对的问题。

---

## §2 传统解法：padding

最直接的修法：把行 stride 从 32 改成 33，每行多占一个 float 的位置。

```cpp
// 有冲突
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

// 无冲突
auto pad   = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
```

stride 改成 33 之后，按列读的偏移是 `0, 33, 66, 99, ...`，每行错开 1 个 bank，32 个 lane 落在 32 个不同 bank，冲突消除。

代价有三：

| 代价 | 具体 |
|---|---|
| 浪费 smem | size = 1024，但要占 cosize = 1055（+3.0%） |
| 破坏对齐 | 行首不再是 128B 对齐，宽向量指令（`LDG.E.128`）用不了 |
| 硬件不接受 | Hopper 的 TMA 和 WGMMA 对 smem layout 有硬性要求，这种 layout 不在其中 |

前两条是性能代价，第三条是 Hopper 上的硬性限制。这就是为什么需要 Swizzle。

---

## §3 Swizzle：对偏移的比特做异或

`Swizzle<B, M, S>` 不改 shape，也不多占一个字节——它只改"逻辑坐标 → 偏移"这个映射，做法是**把偏移的某几个比特异或到另几个比特上**。

```cpp
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
auto swz   = composition(Swizzle<5, 0, 5>{}, plain);
```

三个参数的含义：

- `B = 5`：参与异或的比特宽度（2^B = 32，刚好是 tile 的行宽）
- `M = 0`：每个 bank 单元的起始比特偏移
- `S = 5`：异或的距离（同样是 5，让行内偏移去改行间偏移）

效果：按列读时，plain 的每一行都撞 bank 0（偏移 0, 32, 64...），swizzled 之后偏移变成 0, 33, 66, 99...，每行错开一个 bank：

```
row |  plain 偏移  bank |  swz 偏移  bank
  0 |           0     0 |          0     0
  1 |          32     0 |         33     1
  2 |          64     0 |         66     2
  3 |          96     0 |         99     3
 ...
```

三个关键不变量：

1. **cosize 不变**：plain 和 swizzled 都是 1024，一个字节都没多用。
2. **行方向仍然连续**：`swz(0, 0..7)` = 0, 1, 2, 3, 4, 5, 6, 7，合并访存不受影响。
3. **是双射**：每个逻辑坐标映射到唯一的偏移，不丢数据也不重叠。

三种 layout 的对比（实测 32×32 float tile）：

| layout | cosize | 列读冲突 | 行方向 |
|---|---|---|---|
| `plain (32,32):(32,1)` | 1024 | 32-way | 连续 |
| `padded (32,32):(33,1)` | 1055 | 1-way | 连续但不对齐 |
| `Swizzle<5,0,5>` | 1024 | 1-way | 连续 |

Swizzle 同时消掉冲突、保持对齐、不多占空间。Hopper 只给你这一条路。

---

## §4 GMMA::Layout_K_SW*_Atom —— Hopper 指定的 swizzle

上一节证明了 Swizzle 是正确的选择。但 Hopper 的 WGMMA 更进一步：它不接受任意的 swizzle，只接受这四种官方认可的 swizzle 原子：

| 原子 | layout（half_t） | K 方向最小长度 |
|---|---|---|
| `GMMA::Layout_K_SW128_Atom` | `Sw<3,4,3> ∘ (8,64):(64,1)` | 64 |
| `GMMA::Layout_K_SW64_Atom` | `Sw<2,4,3> ∘ (8,32):(32,1)` | 32 |
| `GMMA::Layout_K_SW32_Atom` | `Sw<1,4,3> ∘ (8,16):(16,1)` | 16 |
| `GMMA::Layout_K_INTER_Atom` | `Sw<0,4,3> ∘ (8,8):(8,1)` | 8 |

这些原子是"最小单元"，用 `tile_to_shape` 铺到需要的大小：

```cpp
// 把 K_SW128 铺成 (BM=64, BK=64) 的 smem layout
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                        make_shape(Int<64>{}, Int<64>{}));

// 加一个 PIPE=3 的 stage 维
auto sA_3stage = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<64>{}, Int<64>{}, Int<3>{}));
```

选哪个原子的规则只有一条：**BK 必须能被原子的 K 长度整除**。比如 BK=32 时 SW128（K 长 64）不可用，SW64（K 长 32）可用。用了不能整除的组合，`tile_to_shape` 会编译失败：

```
"tile_to_shape: block shape does not divide the target shape"
```

Section 05 和 06 的 WGMMA 代码里会直接用这四种原子。这里先见一面，知道"smem layout 不是随便填"。

---

## §5 ThrID > 1：协作式 atom（ldmatrix）

Section 03 见过的所有 atom（`UniversalCopy<float>`、`UniversalCopy<uint128_t>`、`cp.async`）有一个共同点：**ThrID = 1**，即一个线程独立发一条指令。

`ldmatrix` 完全不同：**32 个线程协作才能发出一条指令，每线程各提供一个 smem 指针，硬件把 4 个 8×8 的 half 矩阵分发给所有参与线程**。

```
Copy_Atom<UniversalCopy<uint128_t>, float>   ThrID =  1   NumValSrc = 4
Copy_Atom<SM75_U32x4_LDSM_N, half_t>        ThrID = 32   NumValSrc = 8
```

一次 ldmatrix 操作的总量 = `ThrID × NumValSrc = 32 × 8 = 256 个 half = 4 个 8×8 矩阵`。名字里的 `U32x4` 就是这个 4。

三个变体：

| atom | 搬的矩阵数 | 特点 |
|---|---|---|
| `SM75_U32x4_LDSM_N` | 4 个 8×8 | 标准，加载 A 矩阵 |
| `SM75_U32x2_LDSM_N` | 2 个 8×8 | 只搬一半 |
| `SM75_U16x8_LDSM_T` | 4 个 8×8 | 加载时顺手转置 |

`_T` 变体解决了"B 矩阵在内存里是 row-major，但 MMA 要 col-major"的问题：在把 smem 装入寄存器的同时完成转置，不需要额外一次转置操作。

---

## §6 ldmatrix 的线程↔数据映射

ldmatrix 最反直觉的地方是：**哪个线程拿到哪些元素，是由硬件规定的，没有任何直觉规律可循。**

以一块 16×16 half（值 = 线性下标 0..255），配合 `SM80_16x8x16_F32F16F16F32_TN`（warp MMA，32 线程），用 ldmatrix 装载 A 矩阵：

每个线程拿到的第一个元素：

```
thr0..15:   0   2   4   6  16  18  20  22  32  34  36  38  48  50  52  54
thr16..31: 64  66  68  70  80  82  84  86  96  98 100 102 112 114 116 118
```

不是 0, 1, 2, 3... —— thr0 到 thr3 拿的是 0, 2, 4, 6，thr4 直接跳到 16。

thr0 拿到的全部 8 个元素是：**0, 1, 128, 129, 8, 9, 136, 137**。这个顺序是 Tensor Core 硬件规定的，完全不是连续的，也不是任何常见的矩阵遍历顺序。

这就是"协作式 atom"的价值所在：你不需要搞清楚这个映射，你只需要告诉 CuTe"我要给这个 MMA 的 A 矩阵喂数据"，它生成正确的代码。手写这个映射几乎必错，而且错了不会报错，只是算出错误的结果。

---

## §7 make_tiled_copy_A 与 retile_D 的分工

ldmatrix 和 MMA 要用同一块寄存器（fragment），但各自有不同的"看法"：

- **MMA 的视角**：我要 `(MMA, MMA_M, MMA_K)` 这样 rank-3 的 fragment，每个 MMA 维对应一个 8×8 矩阵块
- **ldmatrix 的视角**：我一次装载 `(8, 1)` 这么一片，就是 8 个 half 一组

这就需要两步：

```cpp
// 1. MMA 告诉我它要什么排布，分配寄存器
auto tCrA = thr_mma.make_fragment_A(tCsA);       // (MMA, MMA_M, MMA_K)

// 2. ldmatrix 换一个视角看同一块寄存器
auto tiled_ldsm = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
auto tXrA = tiled_ldsm.get_slice(threadIdx.x).retile_D(tCrA);   // ldmatrix 的视角

// 3. 用 ldmatrix 的视角搬，再用 MMA 的视角算
copy(tiled_ldsm, tXsA, tXrA);     // 按 ldmatrix 视角写入寄存器
gemm(mma, tCrA, tCrB, tCrC);      // 按 MMA 视角读取寄存器
```

**`retile_D` 不搬数据，不分配内存，只是换一个 layout 去看同一块寄存器。** 这和 Section 02 的 `local_tile` 是同一类操作——view，不是 copy。

三个函数的分工：

| 函数 | 作用 |
|---|---|
| `make_tiled_copy_A(atom, mma)` | 生成"给 MMA 的 A 喂数据"的 TiledCopy，确保线程映射和 MMA 兼容 |
| `partition_S(sA)` | 我这个线程要从 smem 读哪些 |
| `retile_D(tCrA)` | 把 MMA 的 fragment 换成 copy 能用的视角 |

对 B 矩阵有对应的 `make_tiled_copy_B`。Section 05 会看到 Hopper 的 WGMMA 直接读 smem，连寄存器 fragment 这一步都省了。

---

## §8 Capstone：矩阵转置三版对比

矩阵转置是验证 smem layout 效果最干净的例子：

- 从 gmem 按行读（合并，无冲突）→ 写入 smem
- 从 smem 按列读 → 按行写 gmem（合并）

所有的 bank conflict 都被挤到"按列读 smem"这一步，smem layout 是唯一的变量。

### §8.1 三种 smem layout

```cpp
// v2: plain，bank conflict 满载
auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                         make_stride(Int<TILE>{}, Int<1>{}));

// v3: padded，stride 33
auto pad = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                       make_stride(Int<TILE + 1>{}, Int<1>{}));

// v4: swizzle，同样无冲突但不多占空间
auto swz = composition(Swizzle<5, 0, 5>{}, plain);
```

三个 layout 的属性（TILE = 32，单位 float）：

| layout | cosize | 列读冲突 |
|---|---|---|
| plain `(32,32):(32,1)` | 1024 | 32-way |
| padded `(32,32):(33,1)` | 1055 | 1-way |
| `Swizzle<5,0,5>` | 1024 | 1-way |

### §8.2 实测带宽（8192×8192 float，sm_90）

| 版本 | 时间 (ms) | 带宽 (GB/s) |
|---|---|---|
| v1 naive（不过 smem） | 0.962 | 558.1 |
| v2 plain smem（32-way conflict） | 0.353 | 1518.9 |
| v3 padded（stride 33） | 0.178 | 3015.7 |
| v4 Swizzle<5,0,5> | 0.184 | 2915.0 |

### §8.3 结果怎么理解

**v1 → v2（+172%）**：smem 中转把读写都变成合并访存，消掉了 v1 跨步写 gmem 的惩罚。哪怕 v2 有 32-way conflict，过 smem 的收益也远超冲突代价。

**v2 → v3/v4（+2×）**：消掉 32-way conflict，带宽直接翻倍。这和理论预测一致：bank conflict 会让 smem 吞吐变成 1/32，消掉它全都还回来了。

**v3 vs v4（1.99x vs 1.92x）**：两者都消掉了冲突，所以性能相近。v3（padding）略快，可能因为偏移计算更简单。但 v3 的 cosize 更大，而且在 Hopper 上根本不能用——TMA 和 WGMMA 要求 smem layout 必须是官方 swizzle 原子。v4 才是真正通用的选择。

**kernel 一行不改**：v2/v3/v4 共用同一个 `transpose_smem` kernel，layout 由 host 传进去，这就是 Section 03 §4 "host 描述，kernel 索引"的直接回报。想对比三种方案，改的是 host 的三行声明，不是 kernel。

---

## 本章代码地图

| 文件 | 内容 | 对应 README |
|---|---|---|
| `cute_tiled_v0.cu` | bank model、padding、Swizzle、GMMA atoms | §1–4 |
| `cute_tiled_v1.cu` | ldmatrix ThrID、映射探针、retile_D | §5–7 |
| `cute_tiled_capstone.cu` | 转置四版实测对比 | §8 |
| `exercises/ex.cu` | 6 道可自检练习 | 见下 |

---

## 练习

每题配了自动 PASS/FAIL 检查。进 `exercises/` 目录，把 `ex.cu` 里的 `TODO` 填完，`make run` 验证。参考解答在 `exercises/solutions.md`。

**练习 1 — 数 bank ★☆☆**（对应 §1）  
一个 `(32,32):(32,1)` 的 float tile，一个 warp 读第 5 列（`s(0..31, 5)`）时，最热的 bank 被请求几次？先算出来再运行验证。

**练习 2 — 为 half 选 padding ★★☆**（对应 §2）  
tile 换成 `(32,64)` 的 half（2 字节）。要用 padding 消掉列方向冲突，行 stride 至少要加几个 half？提示：half 每个 bank 装 2 个；先算 plain 情形的冲突，再想"错开一个 bank"需要几个 half。

**练习 3 — Swizzle 的三个不变量 ★★☆**（对应 §3）  
给 `(32,32):(32,1)` 套上 `Swizzle<5,0,5>`，回答三个判断题（true/false）：  
A：swizzle 之后 cosize 变大了？  
B：swizzle 之后行方向（`s(0, 0..7)`）还是连续的？  
C：swizzle 之后 `s(r,c)` 到偏移的映射还是双射？

**练习 4 — 选对 GMMA swizzle 原子 ★★☆**（对应 §4）  
你要给 WGMMA 准备一块 `(BM=128, BK=32)` 的 half smem。SW128/SW64/SW32/INTER 的 K 方向长度分别是 64/32/16/8，哪些原子可以用？填一个 4 位 bitmask（bit0=SW128，bit1=SW64，bit2=SW32，bit3=INTER）。

**练习 5 — ldmatrix 搬多少 ★★☆**（对应 §5）  
`Copy_Atom<SM75_U32x4_LDSM_N, half_t>` 一次操作总共搬多少个 half？

**练习 6 — 修一个 smem layout bug ★★★**（对应 §3）  
`ex6_kernel` 做的是 32×32 float 的转置，结果正确，但按列读 smem 时有 32-way conflict。只改 `slay` 的定义（不改任何访存代码），把冲突消掉。padding 和 swizzle 两条路都可以试，数组已按 `33*32` 开好了。
