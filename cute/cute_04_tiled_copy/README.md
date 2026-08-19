# Section 04: Smem Layout、Swizzle 与 TMA 搬运（SM90）

## §0 路线图：这一章解决什么

cute_03 学会了"怎么把数据搬进 smem"——但搬进去之后，**怎么摆**决定了下一步读它的
速度。这一章围绕一个不断追问的链条展开：

> 数据摆进 smem 会撞 bank（§1）→ padding 能修但 SM90 不让（§2）→ 所以要用
> swizzle，它到底怎么映射（§3）→ 会了映射，搬运代码怎么写（§4）→
> 但 SM90 有专门的搬运硬件 TMA，它怎么用、和手写有什么不同（§5）→
> 会搬一块了，怎么让搬和算重叠起来（§6）→ 全部用上，capstone 实测（§7）

| §  | 讲什么 | 结尾抛出 | 代码 |
|---|---|---|---|
| §1 | bank 模型：为什么按列读慢 32 倍 | 怎么修？ | v0 |
| §2 | padding：能修，但 SM90 上是死路 | 那用什么？ | v0 |
| §3 | Swizzle 映射机制（逐比特 / 手算 / 映射表 / M 参数权衡） | 搬运代码要改吗？ | v0 |
| §4 | 用 CuTe 语义搬运：两个方向都用 `copy()` | 硬件有更好的办法吗？ | v1 |
| §5 | TMA：没它怎么搬 → 有它怎么搬 → 四种 swizzle 模式 | 搬和算能重叠吗？ | v2 |
| §6 | Multi-stage：单缓冲 → Double Buffer → Super Buffer → 线程去哪了 | 全用上多快？ | v3 |
| §7 | Capstone：转置六版（含 TMA 版） | 下一章 | capstone |
| §8 | 代码地图 + 8 道练习 + 交接到 05/06/07 | — | ex.cu |

**本章统一用 SM90（H200, `-arch=sm_90a`）。** 不讲 SM80 过渡写法、不讲 `ldmatrix`
（那是 SM80 的 smem→寄存器指令，SM90 的 WGMMA 直接读 smem，不需要它）。

> 本机 `nvidia-smi` 把卡标成 `L20X / 8.9`，但运行时是 **cc 9.0 / 132 SM / 150GB /
> clusterLaunch=1** 的真 Hopper。所有结论都用 `-arch=sm_90a` 实测得到。

---

## §1 bank 模型：为什么按列读慢 32 倍

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

layout 是 `(32,32):(32,1)`（row-major，行 stride = 32）。偏移公式 `off(r,c) = r*32 + c`：

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
bank 号纹丝不动。而转置这个操作**必然要按列读**——这就是冲突的来源。

> §2 之前先问一句：怎么修才能不撞 bank？

---

## §2 padding：能修，但 SM90 上是死路

最直观的修法：把行 stride 从 32 改成 33。

```cpp
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
auto pad   = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
//                                                                        ↑ 只改这里
```

每行多占一个 float，列方向偏移变成 `0, 33, 66, 99, ...`，bank 号 `33 % 32 = 1`，
每行错开 1 个 bank，32 行落到 32 个不同 bank——**冲突消除**。

代价三条：

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

是**编译期**失败，不是跑出错。WGMMA 只认几种"标准"layout，padding 不在其中。
所以在 SM90 上 padding 这条路走不通——**必须用 Swizzle**。

> §2 说"必须用 swizzle"，那 swizzle 到底怎么把偏移映射到别处？§3 逐比特讲。

---

## §3 Swizzle 到底怎么映射

`Swizzle<B, M, S>` **不改 shape、不多占一个字节**，只改"逻辑坐标 → 偏移"这个映射
函数，做法是**把偏移的某几个比特异或到另几个比特上**。

### §3.1 三个参数的含义

32×32 float tile 的偏移是 10 位（0..1023）：

```
     bit:   9   8   7   6   5 │  4   3   2   1   0
            └──── r 的 5 位 ───┘  └──── c 的 5 位 ────┘
            (因为 off = r*32 + c, 高 5 位就是 r, 低 5 位就是 c)

  B = 参与异或的比特数（异或几位）
  M = 最低几位不动（保护 2^M 个元素保持连续）
  S = 异或的距离（目标位和源位相隔几位）
```

映射公式：

```
swz_off(r,c) = plain_off  XOR  ( ((plain_off >> S) & mask_B) << M )
                                  └─ 取出高位段 ─┘   └ 移到低位 ┘
```

### §3.2 `Swizzle<5,0,5>` 逐步手算

B=5, M=0, S=5。取 `(r,c) = (3,5)`，`plain_off = 3*32 + 5 = 101`：

```
  plain_off = 101 = 0b0001100101
                       └─r=3─┘└c=5┘
                        00011  00101

  第 1 步: 取出高 5 位 (r)          = 0b00011 = 3
  第 2 步: M=0, 不左移              = 3
  第 3 步: 和低 5 位异或   c XOR r   = 0b00101 XOR 0b00011 = 0b00110 = 6
  第 4 步: 拼回去                   = r*32 + 6 = 96 + 6 = 102

  swz_off(3,5) = 102          ← 实测 v0 输出正是 102 ✓
```

一句话：**用行号去打乱列号**，`c_new = c XOR r`。

### §3.3 打印出整张映射表

`plain` 和 `swizzled` 并排看（v0 会打印这张表）：

```
plain 偏移 (前 8 行 × 12 列)                swizzled 偏移 Sw<5,0,5>
      c= 0  1  2  3  4  5  6  7             c= 0  1  2  3  4  5  6  7
 r=0     0  1  2  3  4  5  6  7        r=0     0  1  2  3  4  5  6  7   ← r=0: XOR 0, 不变
 r=1    32 33 34 35 36 37 38 39        r=1    33 32 35 34 37 36 39 38   ← 两两交换
 r=2    64 65 66 67 68 69 70 71        r=2    66 67 64 65 70 71 68 69   ← 每 2 个一组交换
 r=3    96 97 98 99 ...                r=3    99 98 97 96 103 102 ...   ← 每 4 个一组倒转
 ...

  r:          0    1    2    3    4    5    6    7
  plain:      0   32   64   96  128  160  192  224   → bank 0 0 0 0 0 0 0 0   ✗
  swizzled:   0   33   66   99  132  165  198  231   → bank 0 1 2 3 4 5 6 7   ✓
```

**swizzle 后的列偏移序列和 padding 的一模一样（0,33,66,99...），但 cosize 还是 1024。**
padding 靠"多占空间"把行推开，swizzle 靠"在原地重排"达到同样效果。

### §3.4 M 参数：连续性和消冲突的权衡

`M` 保护最低 M 位不参与异或，即 **2^M 个相邻元素保持连续**。实测（探针全坐标扫描）：

| swizzle | 列读最坏 | 行内最短连续段 | 128-bit atom |
|---|---|---|---|
| `plain` | **32-way** | 32 个 float | 可用 |
| `Sw<5,0,5>` | 1-way | **1 个 float** | **编译失败** |
| `Sw<4,1,4>` | 2-way | 2 个 float | 编译失败 |
| `Sw<3,2,3>` | 4-way | **4 个 float** | **可用** |

`Sw<5,0,5>` 把冲突消得最干净，但 M=0 意味着每个元素单独被打乱，行内一个连续对都不剩。
用 128-bit atom（一次搬 4 个连续 float）去搬它，编译期就挂：

```
static assertion failed:
  "Copy_Traits: dst failed to vectorize into registers. Layout is incompatible with this CopyOp."
```

`Sw<3,2,3>` 保住 4 个 float 连续（M=2 → 2²=4），128-bit atom 能用，代价是 4-way 冲突。

> **选参数的规则**：先定 `M` = 你要用的向量宽度，再让 `B`、`S` 去消冲突。
> **先保住向量化，再谈消冲突**——这是 §5 里 GMMA 官方原子全是 `M=4` 的原因。

### §3.5 三个不变量

| 不变量 | 为什么 | 实测 |
|---|---|---|
| **cosize 不变** | 不多占 smem | plain 1024 = swz 1024 ✓ |
| **是双射** | 不丢数据、不重叠 | 扫全 1024 坐标无重复 ✓ |
| **行读仍无冲突** | 写 smem 那步不能变慢 | 全行扫描 1-way ✓ |

注意第三条是"行读**无冲突**"，不是"行内偏移**连续**"。`Sw<5,0,5>` 行内是
`33 32 35 34`：不连续（不能向量化），但 32 个 lane 仍落 32 个 bank（不冲突）。

> §3 说清楚了映射，但**搬运代码要不要改**？§4 直接回答这个问题。

---

## §4 怎么用：搬运代码要不要改？

### §4.1 结论先说

**逻辑代码一行都不用改。** swizzle 藏在 layout 里，`s(r,c)` 自动走新映射。
只有用了宽向量 atom 时，要按 §3.4 把 `M` 选对，否则编译失败。

### §4.2 三个接口步骤

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
3. **layout 在 host 构造，kernel 只做索引**。这是 CUTLASS 通例，也是能"只换一行
   声明就对比四种方案"的原因。

### §4.3 tile 和内存怎么分配

以转置为例，把三层尺寸的关系画出来：

```
gmem 里的大矩阵 M×N (4096×4096 float)
┌──────────────────────────────────────┐
│ ┌────┐ ┌────┐ ┌────┐                 │   每个小方块 = 一个 block 负责的 TILE×TILE
│ │blk │ │blk │ │blk │  ...            │   grid = (N/TILE, M/TILE)
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
      ty = threadIdx.x / 32   (行)
```

对应代码：

```cpp
constexpr int TILE = 32;   // smem 方块边长 —— 决定 smem 用量 (32*32*4 = 4KB)
constexpr int NTHR = 256;  // 每 block 线程数 —— 决定每线程搬几个
static dim3 grid() { return dim3(N / TILE, M / TILE); }   // 每个 block 一个方块
```

### §4.4 sweep：线程怎么扫过这个 tile

256 个线程填 32×32 = 1024 个格子，每线程管 4 个，分 4 趟：

```
线程编排:  tx = threadIdx.x % 32     ← 32 个线程横着排, 覆盖一整行
          ty = threadIdx.x / 32     ← 分成 8 组 (256/32)

        c=0 ────────────────────────► c=31
 r=0    [t0  t1  t2 ...          t31]  ← ty=0  第 1 趟
 r=1    [t32 t33 ...            t63]  ← ty=1
 ...
 r=8    [t0  t1 ...            t31 ]  ← ty=0  第 2 趟 (r += 8)
```

关键在中转后的那一行 `s(tx, r)`：**tx 在第一个（行）位置**，一个 warp 的 32 个 lane
读的是同一列的 32 行——正是 §1 的 32-way conflict，也是 swizzle 要修的地方。
写 gmem 那边 `out[(bx+r)*M + by+tx]` 里 tx 连续 → 合并写。

### §4.5 两个方向都用 CuTe 语义

v1 的结论用一个层层递进的四版展示（都用 `Sw<3,2,3>`，gmem 两侧都合并访存）：

```
写法                             带宽 (GB/s)     说明
v1a  裸 / 裸                       2678          起点
v1b  copy / 裸                     3298          语义不一致, 但更快 (编译器起效)
v1c  copy / copy  ← 本章目标写法    3436         语义统一, 而且最快
```

**store 侧能用 `copy()` 的关键是一个"转置视图"**：swizzled layout 不能简单换 mode
来转置，但可以复合一个转置的 layout：

```cpp
// sT 满足 sT(m,n) == s(n,m); swizzle 被保留, 只是 stride 从 (32,1) 变 (1,32)
auto sT = make_tensor(s.data(),
            composition(slay, make_layout(make_shape(Int<TILE>{},Int<TILE>{}),
                                          make_stride(Int<TILE>{},Int<1>{}))));
```

于是"读 sT 的行" == "读 s 的列"，冲突还在原处，但代码是 `copy()`：

```cpp
copy(tc, thr.partition_S(gIn), thr.partition_D(s));    // load:  gmem -> smem
copy(tc, thr.partition_S(sT),  thr.partition_D(gOut)); // store: smem^T -> gmem
```

### §4.6 一个必须避免的"反例"：把转置塞进 gmem stride

v1 里专门演示了一个诱人但错误的写法：不建转置视图，直接让 gmem 目的 layout
用 `(1,N)` 而不是 `(N,1)`，转置就自动完成。**结果正确，但慢到 swizzle 完全失效**：

```
把转置放 gmem 侧 (非合并写):  全 523 GB/s, swz 收益 1.00x
把转置放 smem 侧 (合并写):    plain 2668 / Sw323 3436 GB/s, swz 收益 1.29x
```

非合并的 gmem 写成了唯一瓶颈，bank 冲突根本排不上号。

> **优化顺序：先保证 gmem 合并，再谈 smem 消冲突。** 顺序错了，后面的 swizzle
> 全白做。

> §4 让我们会搬一块了。但 gmem→smem 还是"每线程算地址"。SM90 有专门的搬运硬件
> TMA，§5 讲它。

---

## §5 TMA：把搬运交给硬件

### §5.1 没有 TMA 时怎么搬：cp.async，每线程算自己的地址

v1 的搬运在 SM80/SM90 上都等价于：每个线程算自己的地址、发一条 `cp.async`。
v2 第一节跑的就是这个基准（128×64 half tile）：

```
SM80 cp.async:  每个线程算自己的地址 → 发自己那一份
   ┌──────────────────────────────────────┐
   │ t0 算地址→发  t1 算地址→发  ... t255 │  256 个线程都在算地址
   └──────────────────────────────────────┘

发指令的线程数   128 个 (每人一条)
地址计算         每线程各算一次
边界处理         要自己写 predicate
同步             cp_async_wait + __syncthreads
寄存器           每线程都占几个存地址
```

### §5.2 有了 TMA 怎么搬：一个线程描述整块

Hopper 新增的 **独立硬件单元** 专做 gmem↔smem 整块搬运。同一个 tile 换成 TMA：

```
SM90 TMA:  一个线程描述整块 → 硬件自己搬
   ┌──────────────────────────────────────┐
   │ 1 个线程: "把 gmem(128,64) 那块搬到   │  其余 127 个线程可以去干别的
   │             smem 这里" → 硬件接手     │
   └──────────────────────────────────────┘

发指令的线程数   1 个 lane, 一共一条
地址计算         host 侧 descriptor 算好
边界处理         硬件自动填 0
同步             ClusterTransactionBarrier 按字节数等
寄存器           几乎为 0
swizzle          写进 descriptor
```

**descriptor（host 侧那 128 字节）** 里存了：

```
      gmem 基地址
      gmem 形状 (GM, GK), stride (GK, 1)
      tile 形状 (TM, TK)
      smem 的 swizzle 模式      <- 就是 §5.3 要讲的四种
      元素类型 / 越界填充策略
```

kernel 里只说"搬第 (0,0) 块"，其余全在 descriptor 里。这就是 TMA 的全部含义。

五个硬性条件（v2 [§5.2](cute_tiled_v2.cu) 都标在代码里，踩过坑的都在这里）：

1. **src 必须是 `tma.get_tma_tensor(shape)`** —— 坐标 tensor，不是普通 gmem tensor；
2. **descriptor 在 host 用真实设备指针构造**（用 `nullptr` 会运行时报 `TMA descriptor 201`）；
3. **smem `__align__(128)`**；
4. **smem layout 必须带 PIPE 维**，建 atom 时传切片 `slay(_,_,Int<0>{})`；
5. **partition 用 `tma_partition`**，不是 `partition_S/D`。

### §5.3 TMA 的四种 swizzle 模式

descriptor 里的 swizzle 字段只有几个取值，CuTe 封成四个 layout 原子。
**名字里的数字是一行占多少字节**：

```
   SW128 = Sw<3,4,3> o (8,64):(64,1)   一行 128 字节 = 64 个 half
   SW64  = Sw<2,4,3> o (8,32):(32,1)   一行  64 字节 = 32 个 half
   SW32  = Sw<1,4,3> o (8,16):(16,1)   一行  32 字节 = 16 个 half
   INTER = Sw<0,4,3> o (8,8) :(8,1)    一行  16 字节 =  8 个 half  (不 swizzle)
```

v2 用同一个 TMA kernel 跑四种模式（128×64 half tile）：

```
    SW128   一行 128 字节   落数 正确   consumer 列读 =  8-way
    SW64    一行  64 字节   落数 正确   consumer 列读 =  8-way
    SW32    一行  32 字节   落数 正确   consumer 列读 =  8-way
    INTER   一行  16 字节   落数 正确   consumer 列读 =  4-way
    plain   row-major       落数 正确   consumer 列读 = 32-way  (§5.4 有)
```

两点结论：

1. **四种模式 TMA 都搬得对**——选哪个不影响正确性，只影响 consumer 读 smem 时撞几路 bank。
2. **四个原子的 M 全 = 4**（2^4 = 16 half = 32 字节）。这正是 §3.4 规则的体现：
   先保住向量宽度，再消冲突。官方参数为什么长这样。

**怎么选**：唯一硬约束是 TK 必须能被原子的 K 长度整除。

| TK (half) | SW128(64) | SW64(32) | SW32(16) | INTER(8) |
|---|---|---|---|---|
| 64 | ✓ | ✓ | ✓ | ✓ |
| 32 | ✗ | ✓ | ✓ | ✓ |
| 16 | ✗ | ✗ | ✓ | ✓ |

实用规则：选能用的里面一行字节数最大的。TK=64 选 SW128。

### §5.4 谁知道要 swizzle？TMA 不挑，WGMMA 挑

容易误解成"TMA 要求 swizzle"。**TMA 不要求**——连 plain row-major 都搬得对（§5.3
最后一行）。真正拒绝 plain 的是下游的 **WGMMA**，而且是**编译期**拒绝：

```
WGMMA (SM90_64x64x16_F32F16F16_SS) 实测:
  SW128 atom              编译通过, 结果正确
  INTER atom (Sw<0>)      编译通过, 结果正确
  plain row-major         编译失败: "Not a canonical GMMA_K Layout"
  padded stride BK+8      编译失败: "Not a canonical GMMA_K Layout"
```

WGMMA 不靠寄存器读数据，而是把 smem 地址和摆法编码成 **descriptor**，硬件照它直读
smem。descriptor 里只有几个比特存 swizzle，能表达的摆法就 §5.3 那四种。

所以这一章的 layout 是一份**合同**：

```
TMA (生产者) ---- smem layout ----> WGMMA (消费者)
 不挑, 都能搬          由消费者定       只认 4 种规范形式
```

这也解释了 SM80/SM90 的分工差异（v2 [[§5.4]](cute_tiled_v2.cu) 实测）：

```
SM80:  smem ──ldmatrix──► 寄存器 ──mma──► 结果    (显式搬一次, 要 fragment 寄存器)
SM90:  smem ───descriptor───► wgmma ──► 结果    (不搬! 没有寄存器 fragment)
```

`partition_fragment_A` 在 SM90 返回 `GMMA::DescriptorIterator`（8 字节/线程），而不是
寄存器数组——所以 mainloop 里**没有 `copy(tCsA, tCrA)`**，ldmatrix 在 SM90 是多余的一跳。
这也是为什么本章统一不讲 ldmatrix。

> §5 告诉我们怎么搬一块、怎么选 swizzle。但搬一块和算一块目前是**串行**的：
> 搬的时候计算单元闲着，算的时候搬运引擎闲着。§6 把它们叠起来。

---

## §6 Multi-stage：让搬运和计算重叠

v3 用一个有真实计算量的负载来演示重叠：

```
C = A * B^T    A: 64x4096    B: 64x4096    C: 64x64 (half in, float out)
K 方向切成 64 块, 每块 (A+B) 搬 16 KB, 喂给 WGMMA 累加。
这就是一个单 CTA 的 GEMM mainloop —— cute_06 的完整 GEMM 就是把它铺满 grid。
```

数值（v3 实测，本机）：

| 版本 | smem | 时间 (ms) | TFLOP/s | 相对单缓冲 |
|---|---|---|---|---|
| v3a 单缓冲 | 16 KB | 0.0333 | 1.0 | 1.00x |
| v3b **Double Buffer** | 32 KB | 0.0291 | 1.2 | **1.14x** |
| v3c Super Buffer (3) | 48 KB | 0.0277 | 1.2 | 1.20x |
| v3c Super Buffer (4) | 64 KB | 0.0275 | 1.2 | 1.21x |

> 数字随邻居负载波动，看相对关系。绝对数值低是因为负载只有一个 CTA——真正吃满
> 吞吐要等 cute_06 铺满整个 grid。这里看重叠本身。

### §6.1 单缓冲：两个引擎各闲一半

只有一个 smem buffer，所以时间线是：

```
TMA   : [搬0]      [搬1]      [搬2]
WGMMA :      [算0]      [算1]      [算2]
```

每一轮必须等搬完才能算、等算完才能搬下一块（否则覆盖正在用的数据）。
两个引擎轮流干活，各闲一半。

### §6.2 Double Buffer：2 个 buffer 轮换

有了两个 buffer，`搬 k+1` 和 `算 k` 能同时进行：

```
TMA   : [搬0][搬1][搬2][搬3]
WGMMA :      [算0][算1][算2]
          ^^^^ 搬 k+1 和算 k 重叠
```

需要**两组 barrier**：

```
full[s]   生产者→消费者:  "buffer s 已装满, 可以算了"  按字节数等 (TMA 专用)
empty[s]  消费者→生产者:  "buffer s 已用完, 可以覆盖了" 按到达数等
```

用 mbarrier 而不是 `__syncthreads`：`__syncthreads` 是全 block 栅栏，会强制所有人
停在同一行；这里需要的是"针对某个 buffer 的细粒度等待"，让两组工作各走各的。

**一个必踩的坑（v3 §6.2 实测，第一版直接死锁）：**

```cpp
auto wst = cutlass::PipelineState<STAGES>();   // ← 都从 0 开始
auto rst = cutlass::PipelineState<STAGES>();
// prologue 把 STAGES 个 buffer 都填满后, 千万别想着"write_state 也该预推进 ++STAGES 次"
// —— 预推进会让第一次 empty->wait 用错 phase, 死锁两分钟。
```

### §6.3 Super Buffer：更多 stage，以及 48KB 那道台阶

stage 越多，能容忍的延迟越长（TMA 提前搬得更远）。但**收益递减**——看 §6 的表，
2→3 只有 ~5% 提升，3→4 几乎没了。smem 却是线性涨的。

再往上还有一道**硬台阶**：静态 `__shared__` 上限 **48KB**。实测 v3 的 STAGES=3
（48KB）勉强在限内，STAGES=4（64KB）直接 ptxas 报错：

```
ptxas error: uses too much shared data (0xc000 max)
```

要吃到 H200 的 227KB 必须换 **动态 smem**：

```cpp
// kernel 里
extern __shared__ char smem[];
half_t* rawA = reinterpret_cast<half_t*>(smem);      // A 放偏移 0 (descriptor 期望)
half_t* rawB = rawA + cosize_v<SLayA>;               // B 紧跟

// host 侧, launch 前
cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
// 并且 launch 时在 smem_size_in_bytes 里申请
```

v2 的 §5.2 用静态（单缓冲 16KB 而已），v3 的 STAGES=3/4 用动态。
**这就是官方例子用 `SharedStorage` 结构体 + `cudaFuncSetAttribute` 的原因**——
SM90 GEMM 的 smem 动辄 100KB 以上，绕不开这道台阶。

### §6.4 线程去哪了 / Warp Specialization

先纠正一个常见误解。**"用了 TMA 线程就空着了"这个说法不成立。** 看 v3b 的循环体：

```
    1 个 lane   发 TMA
  128 个线程    一起做 WGMMA     <- 其余 127 个在算, 不是闲着
```

真正的问题是别的：

> **同一批线程既要发搬运又要等计算，两件事被串在一条指令流上。**
> `warpgroup_wait<0>()` 之后所有线程（包括发 TMA 那个 lane）都得等 MMA 完成，
> 于是"发下一块 TMA"这件事被 MMA 挡住了。

解法是**换分工方式**：

```
v3b 统一分工:                Warp Specialization (cute_06 完整实现):
+---------------------+      +------------------+------------------+
| warp 0 1 2 3        |      | producer: 只发TMA| consumer: 只算   |
| 都做 WGMMA          |      | 不参与 MMA       | 不做 WGMMA 以外的|
| 其中 1 个 lane 发TMA|      | 一路往前抢跑     | 事               |
+---------------------+      +------------------+------------------+
        |                           |               |
   一条指令流                   各自独立的指令流, 用 mbarrier 通信
```

producer warp 不做 MMA，所以能一路往前把后面几块 TMA 全发出去，不被任何
`warpgroup_wait` 挡住。这就是 **Warp Specialization**。

v3 只给概念不给完整 kernel，原因有两条（都写死在 v3 §6.4 注释里）：

1. **概念正确性 v3b/v3c 已经证明**——128 个线程在算、只有 1 个 lane 在搬，重叠全靠
   mbarrier 而非 `__syncthreads`；
2. **WS 的收益要在大 GEMM 上才显**——单 CTA 的 64×64 负载连 TMA 延迟都没喂饱，
   硬上 WS 只会把 mbarrier 开销加回去。

完整的 WS kernel（2 个 warpgroup + `setmaxnreg` 寄存器再分配：producer 少要、
consumer 多要）是 **cute_06 的 capstone**。

> §6 用 pipeline 证明了重叠的价值。§7 把本章所有工具用到一个完整的转置上，实测。

---

## §7 Capstone：转置 + TMA

矩阵转置是检验 smem layout 最干净的例子：读 gmem 合并、写 gmem 合并，
**所有冲突都被挤到"按列读 smem"这一步**，于是 layout 成了唯一的变量。

### v1–v5：float 8192² 转置

```
v1 naive（不过 smem）            v2-v5 共用同一个 transpose_smem kernel,
v2 plain（32-way）              差别只在 host 侧那一行 layout 声明:
v3 padded (stride 33)           swap/plain/Sw<5,0,5>/Sw<3,2,3>
v4 Swizzle<5,0,5>
v5 Swizzle<3,2,3>
```

实测：

| 版本 | 时间 (ms) | 带宽 (GB/s) | 相对 plain | 列读 | SM90 可用 |
|---|---|---|---|---|---|
| v1 naive | 0.982 | 547 | — | — | ✓ |
| v2 plain smem | 0.434 | 1238 | 1.00x | 32-way | ✓ |
| v3 padded | 0.185 | 2903 | **2.34x** | 1-way | ✗ WGMMA 拒绝 |
| v4 `Sw<5,0,5>` | 0.198 | 2718 | **2.20x** | 1-way | 冲突最少不能向量化 |
| v5 `Sw<3,2,3>` | 0.199 | 2697 | **2.18x** | 4-way | ✓ 和 GMMA 官方案 |

怎么读：

- **v2→v3/v4/v5 的 2.2x 以上**是消掉 32-way conflict 的直接收益；
- **v3/v4/v5 只差几个百分点**——冲突已经不是瓶颈。选哪个该看**可用性**不看性能：
  padding 在 SM90 过不了 WGMMA 编译期检查；`Sw<5,0,5>` 用不了宽向量 atom；
  只有 `Sw<3,2,3>` 两条都满足（M=2 → 4-float 连续），这正是 GMMA 官方原子的路线。
- **性能排序 ≠ 可用性排序**，这是本章最重要的工程结论。

### v6：TMA 版转置

v6 把 gmem→smem 换成 TMA（§5.2 的五个硬条件 + §5.3 的 SW128 smem layout），
先用上面的转置视图读出来，再合并写 gmem：

```
v6  TMA 版转置  (half 2048×2048, tile 128x128)
    转置结果 正确    ~0.011 ms   ~1500 GB/s   (随机器波动)
```

**为什么 v6 单独展示、不在 v1-v5 的 float 表里？** 三个理由：

1. **TMA 的 SW128 descriptor 天生是 16-bit 的**（§5.3）：一行 128 字节 = 64 个 half。
   给 float 用要另配几何，不是天然适合。
2. **TMA 擅长的是"整块搬进 smem 喂 WGMMA"**——它的价值由 §6 的流水线证明了。
   纯转置这种"转换操作"瓶颈在 bank 冲突，不在搬运指令，TMA 帮不上大忙。
3. 所以 v6 的存在是为了证明**搬运这一步能交给硬件**，且换对；它不参与和 v1-v5
   的性能比较（数据本身就不同：half/2048² vs float/8192²）。

> §7 之后，你已经有了：swizzle layout（§3）、CuTe 语义搬运（§4）、TMA（§5）、
> multi-stage 流水线（§6）。下一章（cute_05 MMA）把这些接上 Tensor Core。

---

## §8 代码地图 + 练习 + 交接

| 文件 | 内容 | 对应 |
|---|---|---|
| `cute_tiled_v0.cu` | bank 模型、padding、Swizzle 逐比特推导 + 映射表 + M 参数 | §1–3 |
| `cute_tiled_v1.cu` | CuTe 语义搬运：两端都用 `copy()`、转置视图、反例 | §4 |
| `cute_tiled_v2.cu` | TMA：cp.async 对照、五种 layout、四种 swizzle、WGMMA 合同 | §5 |
| `cute_tiled_v3.cu` | Multi-stage：单缓冲 → Double → Super → WS 概念 | §6 |
| `cute_tiled_capstone.cu` | 转置六版（含 TMA 版）实测 | §7 |
| `exercises/ex.cu` | 8 道可自检练习 | §8 |

跑：`make run`（依次跑 v0/v1/v2/v3/capstone），`make ex`（练习）。

## 练习

进 `exercises/`，把 `ex.cu` 里的 TODO 填完，`make run` 自动判 PASS/FAIL。
参考解答在 `exercises/solutions.md`。（8 道，后两题给足提示。）

**练习 1 — 数 bank ★☆☆**（§1）
`(32,32):(32,1)` 的 float tile，一个 warp 读第 5 列时最热的 bank 被请求几次？

**练习 2 — 手算 swizzle 映射 ★★☆**（§3.2）
`Swizzle<5,0,5>` 作用在 `(32,32):(32,1)` 上，手算 `swz_off(6, 3)`。按 §3.2 四步走。

**练习 3 — 选 M 保住向量化 ★★★**（§3.4）
128-bit atom 搬 32×32 float tile，`Sw<5,0,5>`/`Sw<4,1,4>`/`Sw<3,2,3>` 哪些能用？
注意冲突最少的那个未必能用。

**练习 4 — 选对 GMMA 原子 ★★☆**（§5.4）
`(BM=128, BK=32)` 的 half smem，SW128/64/32/INTER 哪些可用？

**练习 5 — 谁挑 layout ★★☆**（§5.2 §5.3）
一块 plain row-major 的 smem：TMA 能搬对吗？WGMMA 能编译过吗？两个 true/false。

**练习 6 — 修一个 layout bug ★★★**（§3 §4）
转置结果正确但列读 32-way。**只改 `slay` 一行**，同时满足列读 ≤4-way 且行内连续 ≥4。

**练习 7 — 手写一段 TMA 搬运 ★★★**（§5.2）
用 TMA 把 GM×GK 的一个 TM×TK tile 搬进 smem。五个 TODO 对应五个硬性条件，
提示都在 v2 §5.2 的 `copy_tma_kernel` 里。**盖住参考实现默写一遍才是目的**。

**练习 8 — TMA Double Buffer ★★★**（§6.2）
把单缓冲 TMA→WGMMA 改成 2-stage。三个 TODO；**最易错的一行是 `PipelineState`
不能预推进**（写错的症状是死锁，练习里跑 5 次专门抓它）。

### 交接

| 到 | 学什么 | 这里已经给了什么 |
|---|---|---|
| cute_05 | MMA / WGMMA / TMA 拼成算子 | §5.3 swizzle layout、§5.4 WGMMA 挑 layout、§4 CuTe 语义 |
| cute_06 | 完整 GEMM：多 stage 铺满 grid + Warp Specialization + Block Cluster | §6 的 pipeline `PipelineState` 就是 06 mainloop 的骨架 |
| cute_07 | Persistent Block / tile 调度 | §6.4 的 WS 概念落点 |

这三个后续章节正是 H200 五大特性（TMA、WGMMA、Warp Spec、Persistent、Cluster）
的补全：本章已经实际跑通 TMA + WGMMA + multi-stage，Warp Spec 和 Persistent 在
06/07 各自主场。
