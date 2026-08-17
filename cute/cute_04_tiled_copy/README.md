# Section 04: Smem Layout 与 Swizzle（SM90）

## 本章要解决的问题

cute_03 解决了"怎么把数据搬进来"，但一直搬的是一维连续数组。真实的 GEMM 要多绕一跳：
**gmem → smem → Tensor Core**。这一跳带来一个 cute_03 没有的问题：

> 数据搬进 smem 之后，**摆放方式**决定了下一步读它的速度。摆错了，读 smem 慢 32 倍。

本章就讲这一件事。四个层次：

| §  | 内容 | 一句话 |
|---|---|---|
| §1 | bank 模型 | 为什么"按列读"会慢 32 倍 |
| §2 | padding | 最直观的修法，以及它为什么在 SM90 上是死路 |
| §3 | **Swizzle 的映射机制** | 逐比特讲清 `Swizzle<B,M,S>` 到底怎么算偏移 |
| §4 | **怎么用** | 搬运代码要不要改？（答案：一行都不用改，但有一个陷阱） |
| §5 | SM90 硬件 | TMA 是什么、WGMMA 为什么强制要 swizzle |
| §6 | capstone | 转置四版实测 |

**本章统一用 SM90（H200, `-arch=sm_90a`）。** 不讲 SM80 过渡写法，不讲 `ldmatrix`
（那是 SM80 的 smem→寄存器指令，SM90 的 WGMMA 直接读 smem，根本不需要它）。

---

## §1 bank 模型：为什么按列读会慢 32 倍

smem 硬件被切成 **32 个 bank**，按 4 字节轮流分配：

```
float 下标 :   0    1    2    3   ...   31 |  32   33   34  ...
bank      :   0    1    2    3   ...   31 |   0    1    2  ...
              └──────── 一轮 32 个 ────────┘   └── 绕回来 ──┘
```

规则只有一条：

> 一个 warp 的 32 个 lane 落在 **32 个不同 bank** → 一个周期完成。
> 有 **N 个 lane 落在同一个 bank** → 硬件串行拆成 N 次，即 **N-way conflict**。

### 一个 32×32 float tile 上的两种读法

layout 是 `(32,32):(32,1)`，即 row-major，行 stride = 32。偏移公式 `off(r,c) = r*32 + c`：

```
        c=0    c=1    c=2    c=3   ...
 r=0      0      1      2      3   ...      ← 按行读: 偏移连续
 r=1     32     33     34     35   ...
 r=2     64     65     66     67   ...
 r=3     96     97     98     99   ...
         ↑
      按列读: 偏移 0, 32, 64, 96 ...
```

把偏移换算成 bank（`bank = offset % 32`）：

```
按行读 s(0, 0..31):   偏移 0  1  2  3 ... 31    bank 0  1  2  3 ... 31   ✓ 32 个不同 bank
按列读 s(0..31, 0):   偏移 0 32 64 96 ...       bank 0  0  0  0 ...  0   ✗ 全撞 bank 0
```

**根源**：行 stride = 32，bank 数也 = 32。下一行的同一列 = 偏移 +32 = `32 % 32 = 0`，
bank 号纹丝不动。这不是巧合，任何 stride 是 32 倍数的 float layout 都会这样。

而转置这个操作**必然要按列读**——这就是冲突的来源。

---

## §2 padding：能修，但 SM90 上是死路

最直观的修法：把行 stride 从 32 改成 33。

```cpp
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
auto pad   = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
//                                                                        ↑ 只改这里
```

每行多占一个 float，列方向的偏移就变成 `0, 33, 66, 99, ...`，bank 号 `33 % 32 = 1`，
每行错开 1 个 bank：

```
        r:    0    1    2    3    4  ...   31
 pad 偏移:    0   33   66   99  132  ...  1023
     bank:    0    1    2    3    4  ...   31    ✓ 32 个都不同
```

冲突消除。**实测（v0 输出）：32-way → 1-way。** 代价三条：

| 代价 | 具体 |
|---|---|
| 多占 smem | `size = 1024` 但 `cosize = 1055`（+3.0%） |
| 破坏对齐 | 行首不再 128B 对齐 |
| **WGMMA 编译期拒绝** | SM90 的 Tensor Core 根本不接受这种 layout |

前两条是性能问题，第三条是硬墙。**实测**（探针，`SM90_64x64x16_F32F16F16_SS`）：

```
padded stride BK+8  →  编译失败:
  static assertion failed: "Not a canonical GMMA_K Layout: Expected stride failure."
```

注意是**编译期**失败，不是跑出错结果。SM90 的 WGMMA 只认几种"标准" layout，
padding 出来的不在其中。所以在 SM90 上，padding 这条路走不通——必须用 Swizzle。

---

## §3 Swizzle 到底怎么映射

这是本章的核心。`Swizzle<B, M, S>` **不改 shape，不多占一个字节**，它只改
"逻辑坐标 → 偏移"这个映射函数，做法是**把偏移的某几个比特异或到另几个比特上**。

### §3.1 三个参数的含义

先看偏移的二进制。32×32 float tile 的偏移是 10 位（0..1023）：

```
     bit:   9   8   7   6   5 │  4   3   2   1   0
            └──── r 的 5 位 ───┘  └──── c 的 5 位 ────┘
            (因为 off = r*32 + c, 高 5 位就是 r, 低 5 位就是 c)
```

`Swizzle<B, M, S>` 的三个参数各管一段：

```
     bit:   9   8   7   6   5 │  4   3   2   1   0
            └───── B 位 ──────┘  └─ M 位 ─┘
                  ↑                  ↑         ↑
            "被异或的目标"      "保护区: 这 M 位不参与"
                  └──── 距离 S ────────┘

  B = 参与异或的比特数（异或几位）
  M = 最低几位不动（保护 2^M 个元素保持连续）
  S = 异或的距离（目标位和源位相隔几位）
```

映射公式：

```
swz_off(r,c) = plain_off  XOR  ( ((plain_off >> S) & mask_B) << M )
                                  └── 取出高位段 ──┘   └ 移到低位 ┘
```

### §3.2 `Swizzle<5,0,5>` 逐步手算

B=5, M=0, S=5。取 `(r,c) = (3,5)`，`plain_off = 3*32 + 5 = 101`：

```
  plain_off = 101 = 0b0001100101
                       └─r=3─┘└c=5┘
                        00011  00101

  第 1 步: 取出高 5 位 (r)          = 0b00011 = 3
  第 2 步: M=0, 不左移              = 3
  第 3 步: 和低位异或  c XOR r      = 0b00101 XOR 0b00011 = 0b00110 = 6
  第 4 步: 拼回去                   = r*32 + 6 = 96 + 6 = 102

  swz_off(3,5) = 102          ← 实测 v0 输出正是 102 ✓
```

一句话：**用行号去打乱列号**。`c_new = c XOR r`。

### §3.3 打印出整张映射表

这是理解 swizzle 最快的方式——`plain` 和 `swizzled` 并排看（v0 会打印这张表）：

```
plain 偏移 (前 8 行 × 12 列)                swizzled 偏移 Sw<5,0,5>
      c= 0  1  2  3  4  5  6  7             c= 0  1  2  3  4  5  6  7
 r=0     0  1  2  3  4  5  6  7        r=0     0  1  2  3  4  5  6  7   ← r=0: XOR 0, 不变
 r=1    32 33 34 35 36 37 38 39        r=1    33 32 35 34 37 36 39 38   ← 两两交换
 r=2    64 65 66 67 68 69 70 71        r=2    66 67 64 65 70 71 68 69   ← 每 2 个一组交换
 r=3    96 97 98 99 ...                r=3    99 98 97 96 103 102 ...   ← 每 4 个一组倒转
 r=4   128 129 130 131 ...             r=4   132 133 134 135 128 ...
 r=5   160 161 162 163 ...             r=5   165 164 167 166 161 ...
 r=6   192 193 194 195 ...             r=6   198 199 196 197 194 ...
 r=7   224 225 226 227 ...             r=7   231 230 229 228 227 ...
       ↑                                     ↑
   列方向 bank 全是 0                   列方向 bank = 0,1,2,3,4,5,6,7  ✓
```

看第 0 列（竖着读）：

```
  r:          0    1    2    3    4    5    6    7
  plain:      0   32   64   96  128  160  192  224   → bank 0 0 0 0 0 0 0 0   ✗
  swizzled:   0   33   66   99  132  165  198  231   → bank 0 1 2 3 4 5 6 7   ✓
```

**swizzle 后的列偏移序列和 padding 的一模一样（0, 33, 66, 99...），但 cosize 还是 1024。**
padding 是靠"多占空间"把行推开，swizzle 是靠"在原地重排"达到同样效果。

### §3.4 M 参数：连续性和消冲突的权衡

这是最容易被漏掉、但决定"搬运代码能不能用宽指令"的参数。

`M` 保护最低 M 位不参与异或，即 **2^M 个相邻元素保持连续**。实测（探针全坐标扫描）：

| swizzle | 列读最坏 | 行内最短连续段 | 128-bit atom |
|---|---|---|---|
| `plain` | **32-way** | 32 个 float | 可用 |
| `Sw<5,0,5>` | 1-way | **1 个 float** | **编译失败** |
| `Sw<4,1,4>` | 2-way | 2 个 float | 编译失败 |
| `Sw<3,2,3>` | 4-way | **4 个 float** | **可用** |

`Sw<5,0,5>` 把冲突消得最干净，但 M=0 意味着**每个元素单独被打乱**，行内一个连续对都
不剩。用 128-bit atom（一线程搬 4 个连续 float）去搬它，编译期就挂：

```
static assertion failed:
  "Copy_Traits: dst failed to vectorize into registers. Layout is incompatible with this CopyOp."
```

`Sw<3,2,3>` 保住了 4 个 float 连续（M=2 → 2²=4），所以 128-bit atom 能用，代价是
只把冲突压到 4-way 而不是 1-way。

> **选参数的规则**：先定 `M` = 你要用的向量宽度（128-bit float 搬运 → 4 个 float → M=2；
> half 的 16B → 8 个 half → M=3），再让 `B`、`S` 去消冲突。
> **先保住向量化，再谈消冲突**——这也是 §5 里 GMMA 官方原子全都是 `M=4` 的原因。

### §3.5 三个不变量

| 不变量 | 为什么重要 | 实测 |
|---|---|---|
| **cosize 不变** | 不多占 smem | plain 1024 = swz 1024 ✓ |
| **是双射** | 不丢数据、不重叠 | 扫全 1024 个坐标无重复 ✓ |
| **行读仍无冲突** | 写 smem 那一步不能变慢 | 全行扫描 1-way ✓ |

注意第三条是"行读**无冲突**"，不是"行内偏移**连续**"——这两件事不一样。
`Sw<5,0,5>` 的行内偏移是 `33 32 35 34`：不连续（所以不能向量化），但 32 个 lane
仍落在 32 个不同 bank（所以不冲突）。§3.4 的表就是在区分这两者。

---

## §4 怎么用：搬运代码要不要改？

### §4.1 结论先说

**逻辑代码一行都不用改。** swizzle 藏在 layout 里，`s(r,c)` 这个写法自动走新映射。

唯一要改的是**如果你用了宽向量 atom**，得按 §3.4 选一个 `M` 够大的 swizzle，
否则编译失败。

### §4.2 三步接口

```cpp
// ── host 侧 ──────────────────────────────────────────────
// 第 1 步: 先写出朴素 layout（描述"逻辑形状"）
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}),
                         make_stride(Int<32>{}, Int<1>{}));

// 第 2 步: 套一层 swizzle（描述"实际怎么摆"）
auto slay = composition(Swizzle<3, 2, 3>{}, plain);
//                      ^^^^^^^^^^^^^^^^^ 只有这一行是新增的

// 第 3 步: 传进 kernel（静态 layout 是空类型, sizeof==1, 传参不花钱）
my_kernel<<<grid, block>>>(d_in, d_out, slay);
```

```cpp
// ── kernel 侧: 和没有 swizzle 时完全一样 ──────────────────
template <class SLay>
__global__ void my_kernel(float const* in, float* out, SLay slay) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];   // ← 注意 cosize, 不是 size
    auto s = make_tensor(make_smem_ptr(raw), slay);

    s(r, tx) = in[...];      // 写: 和 plain 写法一字不差
    __syncthreads();
    out[...] = s(tx, r);     // 读: 和 plain 写法一字不差
}
```

三个要点：

1. **`cosize_v<SLay>` 不是 `size`**。padding 的 cosize（1055）比 size（1024）大，
   用 size 开数组会越界。swizzle 两者相等，但统一写 cosize 不会错。
2. **`__align__(128)`**。TMA 硬性要求；不对齐会运行时 `misaligned address`。
3. **layout 在 host 构造，kernel 只做索引**。这是 CUTLASS 的通例，也是能"只换一行
   声明就对比四种方案"的原因。

### §4.3 tile 和内存怎么分配

以 capstone 的转置为例，把三层尺寸的关系画出来：

```
gmem 里的大矩阵 M×N (8192×8192 float)
┌──────────────────────────────────────┐
│ ┌────┐ ┌────┐ ┌────┐                 │   每个小方块 = 一个 block 负责的 TILE×TILE
│ │blk │ │blk │ │blk │  ...            │   grid = (N/TILE, M/TILE) = (256, 256)
│ │0,0 │ │0,1 │ │0,2 │                 │
│ └────┘ └────┘ └────┘                 │
│ ┌────┐                               │
│ │blk │      ...                      │
│ │1,0 │                               │
│ └────┘                               │
└──────────────────────────────────────┘
           ↓  一个 block 内部
    TILE×TILE = 32×32 float 的 smem 缓冲
    ┌─────────────────────┐
    │  smem tile (32,32)  │  ← 由 slay 描述怎么摆
    └─────────────────────┘
           ↑  NTHR=256 个线程协作填充
      tx = threadIdx.x % 32   (列)
      ty = threadIdx.x / 32   (行), 每次跳 NTHR/32 = 8 行
```

对应的三行代码：

```cpp
constexpr int TILE = 32;   // smem 方块边长 —— 决定 smem 用量 (32*32*4 = 4KB)
constexpr int NTHR = 256;  // 每 block 线程数 —— 决定要循环几趟填满 tile

static dim3 grid() { return dim3(N / TILE, M / TILE); }   // 每个 block 一个方块
```

### §4.4 sweep：线程怎么扫过这个 tile

256 个线程要填 32×32 = 1024 个格子，所以每个线程管 4 个，分 4 趟：

```
线程编排:  tx = threadIdx.x % 32     ← 32 个线程横着排，覆盖一整行
          ty = threadIdx.x / 32     ← 分成 8 组（256/32），一组管一行

        c=0 ────────────────────────► c=31
 r=0    [t0  t1  t2 ...          t31]  ← ty=0  第 1 趟
 r=1    [t32 t33 ...            t63]  ← ty=1
 ...
 r=7    [t224 ...              t255]  ← ty=7
 r=8    [t0  t1 ...            t31 ]  ← ty=0  第 2 趟 (r += 8)
 ...
 r=31                                  ← 第 4 趟结束
```

```cpp
// 写 smem: 按行走。同一个 warp 的 32 个 lane 有连续的 tx -> 合并访存, 无冲突
for (int r = ty; r < TILE; r += NTHR / TILE)
    s(r, tx) = in[size_t(by + r) * N + bx + tx];
                                  //  ↑ 同一 warp 里连续 -> gmem 合并

__syncthreads();

// 读 smem: 按列走。s(tx, r) 的第一个下标是 tx -> 同一 warp 扫过一整列
for (int r = ty; r < TILE; r += NTHR / TILE)
    out[size_t(bx + r) * M + by + tx] = s(tx, r);
                                     // ↑ 这一步就是 §1 说的"按列读", 冲突全在这里
```

关键是最后一行 `s(tx, r)`：**tx 在第一个（行）位置**，所以一个 warp 的 32 个 lane
读的是同一列的 32 行。这正是 §1 里 32-way conflict 的场景，也正是 swizzle 要修的地方。

而写 gmem 的 `out[(bx+r)*M + by+tx]` 里 tx 是连续的 → 写是合并的。
**过 smem 的意义就在这里：读和写都合并，代价是中间那一步按列读 smem。**

### §4.5 实测：同一个 kernel，只换 layout

四行用的是**同一个 kernel**（探针实测，8192² float 转置）：

```
                写 smem   读 smem   cosize   时间      带宽        加速
  plain          1-way    32-way     1024   0.421 ms  1276 GB/s   —
  pad 33         1-way     1-way     1055   0.167 ms  3213 GB/s   2.52x
  Sw<5,0,5>      1-way     1-way     1024   0.252 ms  2129 GB/s   1.67x
  Sw<3,2,3>      1-way     4-way     1024   0.190 ms  2821 GB/s   2.21x
```

读法：

- **消冲突确实值 2 倍以上** —— plain 的 32-way 是唯一瓶颈。
- **`Sw<5,0,5>` 冲突最少（1-way）却不是最快**。因为 M=0 破坏了行内连续性，
  编译器发不出宽访存指令，省下的 bank 周期被更多的指令数吃掉了。
- **`Sw<3,2,3>` 是更好的选择**：4-way 冲突但保住 4-float 连续，反而更快。
- padding 在这个纯转置里最快，但它在 SM90 上过不了 WGMMA 的编译期检查（§2）。

> 这张表纠正一个常见误解：**"冲突越少越快"不成立**。真正要平衡的是
> `冲突次数 × 每次代价` 和 `指令条数`。§3.4 的 M 参数就是这个平衡点的旋钮。

---

## §5 SM90 的两个硬件引擎

### §5.1 TMA（Tensor Memory Accelerator）是什么

Hopper 新增的一个**独立硬件单元**，专门做 gmem ↔ smem 的整块搬运。

和 SM80 `cp.async` 的根本区别：

```
SM80 cp.async:  每个线程算自己的地址 → 发自己那一份
   ┌──────────────────────────────────────┐
   │ t0 算地址→发  t1 算地址→发  ... t255 │  256 个线程都在算地址
   └──────────────────────────────────────┘

SM90 TMA:  一个线程描述整块 → 硬件自己搬
   ┌──────────────────────────────────────┐
   │ 1 个线程: "把 gmem(128,64) 那块搬到   │  其余 255 个线程可以去干别的
   │            smem 这里" → 硬件接手      │
   └──────────────────────────────────────┘
```

TMA 的硬件特性：

| 特性 | 说明 |
|---|---|
| **descriptor 驱动** | host 侧建一个 128 字节的描述符，记下 gmem 形状/步长/smem 摆法 |
| **一个线程发起** | `elect_one_sync()` 选出一个 lane 发指令，不是每线程各发 |
| **自带边界处理** | 越界自动填 0，不需要写 predicate |
| **异步 + mbarrier** | 不用 `__syncthreads`，用 `ClusterTransactionBarrier` 按字节数等 |
| **硬件做 swizzle** | descriptor 里带 swizzle 模式，搬的过程中就把数据摆成 swizzled 形式 |
| **multicast** | 一次搬运可以同时灌进 cluster 里多个 CTA 的 smem |

最后两条是本章的关键：**swizzle 是写进 TMA descriptor 的**，所以 smem layout 不能
随便填——它是 TMA 和 WGMMA 之间的一份合同。

### §5.2 TMA 接受哪些 layout：实测

我实测了五种 smem layout 走 TMA load（128×64 half，`make_tma_atom` + `tma_partition`）：

```
                TMA 落数      consumer 侧列读 (32 lane)
  SW128 atom     正确            8-way
  SW64 atom      正确            8-way
  SW32 atom      正确            8-way
  INTER atom     正确            4-way
  plain 行主序   正确           32-way
```

**TMA 自己不挑 layout——连 plain row-major 都搬得对。** swizzle 的价值不在 TMA 这一
侧，而在**消费者**那一侧（最后一列数字）：数据摆进 smem 之后，WGMMA 或你自己的 kernel
去读它时的冲突数。

> 这一点值得强调，因为很容易误解成"TMA 要求 swizzle"。**TMA 不要求，WGMMA 要求。**

### §5.3 WGMMA 才是那道硬墙

`SM90_64x64x16_F32F16F16_SS` 实测四种 layout：

```
  SW128 atom              编译通过, 结果正确
  INTER atom (Sw<0>)      编译通过, 结果正确
  plain row-major         编译失败: "Not a canonical GMMA_K Layout"
  padded stride BK+8      编译失败: "Not a canonical GMMA_K Layout"
```

WGMMA 只认**规范形式（canonical）**的 layout。它不是靠寄存器读数据——而是把 smem 地址
和摆法编码成一个 **descriptor**，硬件按 descriptor 直读 smem。descriptor 里只有几个
比特存 swizzle 模式，所以能表达的摆法就那么几种，plain 和 padded 都不在其中。

这也解释了 SM90 和 SM80 的分工差异：

```
SM80:  smem ──ldmatrix──► 寄存器 ──mma──► 结果
              (显式搬一次, 要 fragment 寄存器, 要 retile)

SM90:  smem ─────────descriptor────────► wgmma ──► 结果
              (不搬! 没有 fragment 寄存器)
```

实测印证（`partition_fragment_A` 的返回类型）：

```
SM90: GMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)     ← 是描述符, 每线程 8 字节
SM80: 真正的寄存器数组, 要先 copy(tCsA, tCrA) 填进去
```

`size(mma)`：SM80 是 **32**（一个 warp），SM90 是 **128**（一个 warpgroup = 4 warp）。

> 所以本章不讲 `ldmatrix`：在 SM90 上它是多余的一跳。SM80 那条链路（`make_fragment_A`
> / `retile_D` / `ldmatrix`）解决的是"怎么把 smem 搬进寄存器"，而 WGMMA 根本不需要
> 寄存器。这也是为什么 cute_05 讲 WGMMA 时不会再出现 fragment 拷贝。

### §5.4 GMMA 官方 swizzle 原子

WGMMA 认的"规范 layout"就是这四个原子（实测打印）：

```
  SW128 = Sw<3,4,3> o (8,64):(64,1)     K 方向 64 个 half = 128 字节
  SW64  = Sw<2,4,3> o (8,32):(32,1)     K 方向 32 个 half =  64 字节
  SW32  = Sw<1,4,3> o (8,16):(16,1)     K 方向 16 个 half =  32 字节
  INTER = Sw<0,4,3> o (8,8):(8,1)       K 方向  8 个 half =  16 字节
```

三点值得注意：

1. **名字里的数字是字节数**：SW128 = 一行 128 字节。
2. **M 全都是 4**：2⁴ = 16 个 half = 32 字节。对上 §3.4 的规则——先保证向量化
   （16B/32B 访问），再消冲突。这就是官方参数为什么长这样。
3. **INTER 是 `Sw<0>`**，即"不 swizzle"。它也是合法的规范 layout，只是消冲突效果
   最弱（实测 4-way，SW128 是 8-way ... 注意这里 SW128 反而更高，因为 tile 铺开后
   两者的行分布不同；具体数值见 v0 输出）。

用 `tile_to_shape` 把原子铺到需要的大小：

```cpp
// 铺成 (BM=128, BK=64)
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                        make_shape(Int<128>{}, Int<64>{}));

// 加一个 PIPE=3 的 stage 维（多 stage 流水线用, cute_06 会用到）
auto sA3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                         make_shape(Int<128>{}, Int<64>{}, Int<3>{}));
```

**唯一的规则：BK 必须能被原子的 K 长度整除。** 否则编译期报

```
"tile_to_shape: block shape does not divide the target shape"
```

| BK (half) | SW128(64) | SW64(32) | SW32(16) | INTER(8) |
|---|---|---|---|---|
| 64 | ✓ | ✓ | ✓ | ✓ |
| 32 | ✗ | ✓ | ✓ | ✓ |
| 16 | ✗ | ✗ | ✓ | ✓ |

**实用规则：选能用的里面 K 最长的**（对齐最大 = 访存最宽）。BK=64 选 SW128。

---

## §6 Capstone：转置四版

矩阵转置是检验 smem layout 最干净的例子：读 gmem 合并、写 gmem 合并，
**所有冲突都被挤到"按列读 smem"这一步**，于是 layout 成了唯一的变量。

五版：

```cpp
// v1: 不过 smem, 直接跨步写 gmem       —— 没有 smem 就只能牺牲读或写一边
// v2: 过 smem, plain layout            —— 32-way conflict
// v3: 过 smem, padding stride 33       —— 消冲突, 但 SM90 上 WGMMA 拒绝
// v4: 过 smem, Swizzle<5,0,5>          —— 冲突最少, 但 M=0 挡住宽向量
// v5: 过 smem, Swizzle<3,2,3>          —— M=2, 和 GMMA 官方原子同路线
```

v2–v5 **共用同一个 `transpose_smem` kernel**，差别只在 host 侧那一行 layout 声明。

### 实测（8192² float，sm_90a）

| 版本 | 时间 (ms) | 带宽 (GB/s) | 相对 plain | 列读 |
|---|---|---|---|---|
| v1 naive | 0.982 | 547 | — | — |
| v2 plain | 0.434 | 1238 | 1.00x | 32-way |
| v3 padded | 0.185 | 2903 | **2.34x** | 1-way |
| v4 `Sw<5,0,5>` | 0.198 | 2718 | **2.20x** | 1-way |
| v5 `Sw<3,2,3>` | 0.199 | 2697 | **2.18x** | 4-way |

怎么读这张表：

- **v2 → v3/v4/v5 的 2.2x 以上**是消掉 32-way conflict 的直接收益。
- **v3/v4/v5 三者只差几个百分点**：冲突已经不是瓶颈了。所以选哪个**不该看性能**，
  该看可用性——padding 在 SM90 上过不了 WGMMA 的编译期检查（§5.3），
  `Sw<5,0,5>` 用不了宽向量 atom（§3.4）。只有 `Sw<3,2,3>` 这一路线两条都满足，
  这也正是 GMMA 官方原子的选择（它们 M 全 = 4）。
- **性能排序和可用性排序不是一回事**，这是本章最重要的工程结论。

> 数字随机器和邻居负载波动，看相对关系。本机测时请指定空闲卡：
> `CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_capstone`

---

## 本章代码地图

| 文件 | 内容 | 对应 |
|---|---|---|
| `cute_tiled_v0.cu` | bank 模型、padding、**Swizzle 逐比特推导 + 映射表** | §1–3 |
| `cute_tiled_v1.cu` | **无 swizzle vs 有 swizzle 的搬运对比**、M 参数权衡 | §4 |
| `cute_tiled_v2.cu` | **SM90: TMA 实测、WGMMA 的 layout 要求、GMMA 原子** | §5 |
| `cute_tiled_capstone.cu` | 转置四版实测 | §6 |
| `exercises/ex.cu` | 6 道可自检练习 | 全章 |

跑：`make run`（依次跑 v0/v1/v2/capstone），`make ex`（练习）。

---

## 练习

进 `exercises/`，把 `ex.cu` 里的 `TODO` 填完，`make run` 自动判 PASS/FAIL。
参考解答在 `exercises/solutions.md`。

**练习 1 — 数 bank ★☆☆**（§1）
`(32,32):(32,1)` 的 float tile，一个 warp 读第 5 列时最热的 bank 被请求几次？

**练习 2 — 手算 swizzle 映射 ★★☆**（§3.2）
`Swizzle<5,0,5>` 作用在 `(32,32):(32,1)` 上，手算 `swz_off(6, 3)`。
按 §3.2 的四步走，先别跑代码。

**练习 3 — 选 M 保住向量化 ★★★**（§3.4）
你要用 128-bit atom（每线程 4 个连续 float）搬一个 32×32 float tile。
`Swizzle<5,0,5>` / `Swizzle<4,1,4>` / `Swizzle<3,2,3>` 里哪些能用？填 3 位 bitmask。
注意冲突最少的那个未必能用。

**练习 4 — 选对 GMMA 原子 ★★☆**（§5.4）
给 WGMMA 准备 `(BM=128, BK=32)` 的 half smem。四个原子哪些可用？填 4 位 bitmask。

**练习 5 — 谁挑 layout ★★☆**（§5.2 §5.3）
一块 plain row-major 的 smem：TMA 能不能搬对？WGMMA 能不能编译过？两个 true/false。
把 `ex.cu` 里的 `EX5_TRY_PLAIN_WGMMA` 改成 1 可以亲眼看到报错。

**练习 6 — 修一个 smem layout bug ★★★**（§3 §4）
`ex6_kernel` 转置结果正确但列读 32-way 冲突。**只改 `slay` 一行**，
同时满足两个检查：列读 ≤ 4-way **且** 行内连续 ≥ 4 个 float。
第二个检查会排除掉那个"看起来最优"的答案。
