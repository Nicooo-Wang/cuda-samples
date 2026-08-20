# cute_05 · MMA Atom：Tensor Core 是什么，以及 Hopper 改了什么

> 前置：cute_01（Layout）、cute_02（Tensor）、cute_03（Copy Atom）、
> cute_04（TMA → 边界 → Swizzle → Multi-stage）。
>
> 这一章是**从"搬数据"跳到"算数据"** 的关口。
> 读这一章时你会反复被要求"回 cute_04 看一眼"—— 因为搬运和计算在 Hopper 上
> 是**同一个问题**的两半：TMA 把数据摆进 smem，WGMMA 直接从 smem 取。摆法错了，
> 计算端第一个报错的就是你。

---

## §0 路线图：这一章解决什么

前四章我们一直在问"**怎么把数据搬到该在的地方**"：

- cute_03：gmem ↔ 寄存器，`Copy_Atom`
- cute_04：gmem ↔ **smem**，TMA 整块搬；smem 里怎么摆（swizzle）才不撞 bank

但**数据到位之后，谁来算？** 你写的 `C[i][j] += A[i][k]*B[k][j]` 在 GPU 上是
一个线程一个线程地乘加的（SIMT）。Tensor Core 不是这样 —— 它是**一条指令算一块矩阵**。
这一章回答三连问：

1. **一条 Tensor Core 指令长什么样？**（§1–§2，用 Ampere 的 MMA 讲清楚）
2. **Hopper 把它改成了什么？**（§3）
3. **数据怎么喂进去？**（§4）

然后 capstone 把它们拼成一个能跑真实尺寸的 GEMM。

```
        cute_04                       cute_05
 ┌──────────────────┐         ┌──────────────────────────┐
 │ gmem ──TMA──▶ smem │         │  smem ──WGMMA──▶ 累加器    │
 │   (怎么搬)         │         │    (怎么算)               │
 └──────────────────┘         └──────────────────────────┘
      ↑ 已经会了                    ↑ 这一章要学
```

> 为什么先用 Ampere 的 MMA 讲？因为**它看得见**。一条 `mma.sync` 指令的
> A/B/C 全在寄存器里，每个线程拿几个元素能打印出来数。Hopper 的 WGMMA
> 把 A/B 变成了"硬件按 descriptor 去读 smem"，寄存器里只剩累加器 ——
> 你要不是先见过"在寄存器里的版本"，根本看不出 Hopper 改了什么。

---

## §1 一条 MMA 指令的三个形状

打开 `cute_mma_v0.cu`，跑一下，第一屏就是这一节的东西。都来自
`make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{})` 的 `print()`：

```
TiledMMA
  ThrLayoutVMNK:  (_32,_1,_1,_1):(_1,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _32:_1
  Shape_MNK:  (_16,_8,_16)
  LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

一条 MMA 指令 = 三样东西：

### §1.1 Shape_MNK：算多大一块

名字 `SM80_16x8x16` 直接就是这个形状：**M=16, N=8, K=16**。
所以 `mma.sync` 一条指令算的是：

```
   C[16×8]  +=  A[16×16]  ·  B[8×16]^T
              ↑        ↑   ↑
              M×K        N×K（存的是 B^T）
```

名字后半段 `F32F16F16F32` = 累加器 f32、A f16、B f16、C f32（累加器比输入宽，
这是为了 K 方向累加不舍入）。`TN` 是摆法：**A 行主 + B 行主**，两边 K 都连续。

### §1.2 ThrID：谁参与这条指令

`ThrID: _32` 意思是「这条指令由 32 个线程**共同**发出」—— 一个 warp。
不是 1 个线程（那是 SIMT 的算法），也不是整个 block。**Tensor Core 是 warp 协作的。**
`size(mma) == 32`。

### §1.3 LayoutA/B/C_TV：数据怎么分给线程

抓住 `_TV` 两个字母：**T**hread / **V**alue。`LayoutA_TV` 把 (线程, 值) 映射到
A 的逻辑坐标。上面 A 是：

```
((_4,_8),   (_2,_2,_2))   :   ((_32,_1),  (_16,_8,_128))
 └─线程排布─┘  └─每股值排列─┘    └线程 stride─┘  └─值 stride─┘
```

读法：32 个线程摆成 `4×8`；每个线程拿 `2×2×2 = 8` 个 A 元素。核对：
`16×16 = 256` 个 half ÷ 32 线程 = **每人 8 个** ✓。B 是 `8×16/32 = 4` 个/线程，
C 是 `16×8/32` = **每人 4 个 float** ✓。

> **一句话总结 §1**：一条 Tensor Core 指令 = 「几个线程协作 + 算多大 + 数据怎么摊给线程」。
> CuTe 把这三样打包成一个 **`MMA_Atom`**。

---

## §2 从"整块"到"我这一份"：partition_fragment

`LayoutA_TV` 是 **32 个线程整体** 的地图。写 kernel 时每个线程只关心**自己那几份**
寄存器。`partition_fragment_A/B/C` 干的就是这个：

```cpp
TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
ThrMMA  thr = mma.get_thread_slice(threadIdx.x);

auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<16>{}, Int<16>{}), ...);
auto fA = thr.partition_fragment_A(gA);   // 我这份 A: 8 个 half, 寄存器
auto fB = thr.partition_fragment_B(gB);   // 我这份 B: 4 个 half, 寄存器
auto fC = thr.partition_fragment_C(gC);   // 我这份 C: 4 个 float, 寄存器
```

`cute_mma_v0.cu` 的 §2.2 把一条真的 MMA 跑通了，四步：

```cpp
copy(tAgA, tArA);   // 1. gmem -> 我这份寄存器
copy(tBgB, tBrB);
clear(tCrC);        // 2. 累加器清零（MMA 是 C += A*B）
gemm(mma, tArA, tBrB, tCrC);  // 3. 一条指令
copy(tCrC, tCgC);   // 4. 寄存器 -> gmem
```

**关键设计**：第 1 步和第 4 步用**同一个** `thr.partition_*(g*)`，
所以每个线程读哪几个、写哪几个自动自洽 —— 你**不需要手算** lane id 到
坐标的公式。

### §2.3 TiledMMA：一条不够大，用多个 warp 拼

一条 SM80 MMA 只算 `16×8`。真实 tile 是 `128×128` 起步，怎么办？
`make_tiled_mma` 的第二个参数说"用几个 warp、怎么摆"去重复这条指令：

```cpp
make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>{})   // 2×2 个 warp = 128 线程
// 覆盖 M×N = 32×16, 指令还是那一条 16×8×16, 只是发 4 次
```

⚠️ 这里**只在 SM80 上成立**。到了 §3 你会发现 WGMMA 完全不一样 ——
先卖个关子。

---

## §3 WGMMA：Hopper 把 MMA 改成了什么

跑 `cute_mma_v1.cu`，把两个 atom 并排打出来：

```
  --- Ampere: SM80_16x8x16 ---        --- Hopper: SM90_64x64x16_SS ---
  ThrID:      _32                      ThrID:      _128
  Shape_MNK:  (_16,_8,_16)             Shape_MNK:  (_64,_64,_16)
  LayoutA_TV: ((_4,_8),(_2,_2,_2))     LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
               线程 stride=_32                       线程 stride=**0**
```

### §3.1 两处改变

| | Ampere MMA | Hopper WGMMA |
|---|---|---|
| 谁发 | warp（32） | **warpgroup（128 = 4 warp）** |
| 一条指令算 | 16×8×16 | **64×64×16** |
| A/B 放哪 | 寄存器（每人 8/4 个） | **不进寄存器，硬件直接读 smem** |
| mainloop | `copy(tCsA,tCrA); gemm(...)` | 没有那行 copy，直接 `gemm(...)` |

看到 `LayoutA_TV` 里线程维的 stride 从 `_32` 变成 **`_0`** 了吗？这是判断句：
**stride 为 0 = 每个线程"看到"的都是同一整块 A**。因为根本没有线程去读 A ——
是 WGMMA 硬件拿着一个 **descriptor**（描述 smem 摆法的地址句柄）自己去读。

### §3.2 A/B 的 fragment 变成了 descriptor

```
partition_A(sA)     -> Sw<3,4,3>_smem_ptr o ((_64,_16),_1,_4)   <- 还是 smem tensor
make_fragment_A(..) -> GMMA::DescriptorIterator o (_1,_1,_4)     <- 不是寄存器!
partition_fragment_C-> ptr[32b] o ((_2,_2,_8),_1,_1)             <- 累加器仍是真寄存器
```

所以 Hopper 的 mainloop 比 Ampere **少一步**：

```
Ampere: smem --ldmatrix--> 寄存器 --mma--> 累加器
Hopper: smem ----------------wgmma-------> 累加器
```

那条 `ldmatrix`（把 smem 摊到每个线程寄存器）的指令，在 Hopper 上被硬件吃掉了。

### §3.3 WGMMA 的固定四句

```cpp
warpgroup_arrive();          // 告诉硬件"我要开始发 WGMMA 了, smem 已就绪"
gemm(mma, tCrA, tCrB, tCrC); // 发指令（异步, 发完就返回）
warpgroup_commit_batch();    // 把刚发的这批打包
warpgroup_wait<0>();         // 等这批做完
```

这四句是 **WGMMA 的固定套路**，少一个就会在累加器没写完时去读它 —— 数据错且难查。
注意 `gemm()` 那行是**异步**的：它把指令发射出去就返回，真正的乘加在引擎里跑。
这就是 §4 能聊"让引擎并行"的前提。

### §3.4 代价：smem layout 必须是 GMMA 认识的

WGMMA 硬件按 descriptor 寻址 smem，而 descriptor 里只装得下几种固定的 stride 编码
—— 所以它对 smem 摆法有**编译期**硬性要求。`cute_mma_v1.cu` 打出四种官方原子：

```
Layout_K_SW128_Atom   K % 64 == 0   Sw<3,4,3> ...
Layout_K_SW64_Atom    K % 32 == 0   Sw<2,4,3> ...
Layout_K_SW32_Atom    K % 16 == 0   Sw<1,4,3> ...
Layout_K_INTER_Atom   K %  8 == 0   Sw<0,4,3> ...
```

两条规则：

1. **K 必须被 atom 的 K 长度整除**（SW128→64，SW64→32，SW32→16，INTER→8），
   否则编译期报 `block shape does not divide the target shape`。
2. **普通 row-major / padding 过的 layout 直接编译期拒绝**，报
   `Not a canonical GMMA_K Layout`。这就是 cute_04 §2 说"padding 在 SM90 是死路"
   的**真正原因** —— 不是 TMA 拒绝（TMA 全都能搬），是 **WGMMA 拒绝**。

> 这也解释了 cute_04 为什么要花一整章讲 swizzle：swizzle 不只是"消 bank conflict",
> 它是 GMMA（以及 TMA 的 SW 模式）**认识的标准摆法**。你绕不开它。

---

## §4 TMA：给 MMA 喂数据的那条通路

跑 `cute_mma_v2.cu`。同一个 GEMM（K 切成多个 tile，WGMMA 一路累加），
只搬数据那几行不同。对比两版 mainloop：

```
§4.1a 手写搬运:                 §4.1b TMA:
for i = tid; i < BM*BK; i+=128   if (one lane) {
    sA(...) = mA(...)               expect_tx(bar, 字节数);
    sB(...) = mB(...)               copy(tma_a.with(bar), gA, sA);
__syncthreads()                     copy(tma_b.with(bar), gB, sB)
    ← 128 线程各搬各的         }
    ← 全 block 栅栏            Bar::wait(bar, phase)
                                    ← 1 个 lane 描述整块,硬件搬
                                    ← mbarrier 按字节等
```

三个差别，也是 TMA 省下的三样东西：

1. **指令流**：`128线程 × N条 load` → `1 lane × 1条 TMA`。省下的发射槽让给 WGMMA。
2. **寄存器**：搬数据不再需要存地址和 in-flight 数据。WGMMA 的累加器已经占了
   每线程 32 个 float，这很关键。
3. **同步粒度**：`__syncthreads`（全 block 栅栏）→ `mbarrier`（按字节数等）。
   这是能做 producer/consumer、能做多 stage 流水线的**前提**（cute_04 §5）。

### §4.2 五个硬性条件

cute_04 §2 讲全了，这一章只在 `cute_mma_v2.cu` 里逐条标了位置。五个条件：

1. **src 必须是 `tma.get_tma_tensor(shape)` 的坐标 tensor** —— 不是普通 gmem tensor。
2. **descriptor 必须 host 侧用真实设备指针构造**。
3. **smem 必须 `__align__(128)`**。
4. **smem layout 必须带 PIPE mode** → `(BM,BK,PIPE)`。
5. **partition 必须用 `tma_partition`**，不是 `partition_S/D`。

> 提醒：`elect_one_sync()` 是**每个 warp 选一个 lane**，不是全 block 选一个。
> 跨多个 warpgroup 时一定要再限定 `(one && warp == leader)`，否则会有
> 多个 lane 同时发 TMA —— 详见 cute_04 §5.4 和练习。

> 诚实的话：这一版**没有变快**。搬完才算、算完才搬，两个引擎各闲一半。
> TMA 的价值要等它和多 stage 组合才兑现（cute_04 §5），cute_06 会铺满 grid。

---

## §5 Capstone：TMA + WGMMA 的完整 GEMM

跑 `cute_mma_capstone.cu`。前面 v0/v1/v2 分别解决"atom 是什么、WGMMA 怎么发、
TMA 怎么喂"。capstone 把它们拼起来，再加一层 grid：

```
grid = (N/BN, M/BM)
每个 CTA (128 线程) 负责 C 的一块 [BM×BN], 沿 K 循环:
    TMA 搬 A[BM×BK] + B[BN×BK] 进 smem
      ↓ mbarrier 等
    WGMMA 累加进寄存器
      ↓
    (下一轮)
最后写回 C
```

### §5.1 一个必须知道的坑：WGMMA 的 TiledMMA 不是"几个 warp 拼"

v0 §2.3 我们给 SM80 原子加 `Layout<Shape<_2,_2,_1>>` 拼出更大 tile。
**这一招在 WGMMA 上是错的**：WGMMA 原子本身就要 128 线程（一个 warpgroup），
所以 `make_tiled_mma` 的第二个参数在 SM90 上变成了"用几个 **warpgroup**"。
写 `Layout<Shape<_2,_1,_1>>{}` 会得 `size(mma)==256`（两个 warpgroup）——
你已经把它当"2个warp 128线程"用了，于是**一半的 C 没有线程去算，结果静默错**。
（我第一版就这么栽的，capstone 的代码注释里写了。）

那 BM=128 比原子 M=64 大一倍，谁去覆盖？**CuTe 自动重复原子**：
`partition_fragment_C(128×64 的 gC)` 会给出 `MMA_M = _2` ——
一个 warpgroup 沿 M 把同一条 WGMMA 发两次就够，不需要多要线程。

```
§5.2 正确性: 256³ / 512x256x512 和 CPU 逐点比, bad=0
§5.3 吞吐  : 4096³ 约 350~400 TFLOP/s (无流水线, 具体以你机器打印为准)
```

### §5.3 为什么离 cuBLAS 还远

本机 cuBLAS FP16 `4096³` 实测 ~878 TFLOP/s（~89% 峰值），capstone 只有它的一半上下。
三个原因，每个都是 cute_06 的一节：

| 差距 | 原因 | cute_06 解法 |
|---|---|---|
| 两个引擎各闲一半 | 搬完才算、算完才搬 | v3 多 stage 流水线 |
| 线程又搬又算串一起 | 同一批线程发 TMA 又等 WGMMA | v4 Warp Specialization |
| tile 小、一个 CTA 一个 warpgroup | `BM=128` 只有一个 WGMMA | v5 更大 tile + Block Cluster |

`cute_06` 的终点就是：把这三点全部解决，写出对标 cuBLAS 的 GEMM。

---

## §6 练习

跑 `make ex`，把 `exercises/ex.cu` 里的 TODO 填掉，再跑 `make ex`，
每题填对会打印 PASS。解答在 `exercises/solutions.md`。

| # | 主题 | 对应 |
|---|---|---|
| 1 | 从 TV 布局手算一个线程拿几个元素 | §1.3 |
| 2 | fragment 是寄存器还是描述符 | §3.2 |
| 3 | 手写一条 WGMMA 的四句（少一句会怎样） | §3.3 |
| 4 | 改错：WGMMA 的 TiledMMA 写成发 warp 数量 | §5.1 |
| 5 | 手写一段 TMA 搬运（五个条件含一处要修的） | §4.2 |
| 6 | 手写一个单 CTA 的 TMA + WGMMA GEMM | §5 |

---

## §7 代码地图 & 下一步

```
cute_05_mma_atom/
├── common.h               # 填矩阵 / CPU GEMM / 比对 / TFLOP/s
├── cute_mma_v0.cu         # §1-2  MMA Atom 解剖 (Ampere warp 级)
├── cute_mma_v1.cu         # §3    WGMMA (warpgroup 级, descriptor 读 smem)
├── cute_mma_v2.cu         # §4    TMA 数据通路, 和手写搬运并排
├── cute_mma_capstone.cu   # §5    单 CTA TMA+WGMMA, grid 铺开
├── exercises/             # §6    6 道练习 + solutions.md
├── README.md
└── Makefile               # make all / make run / make ex
```

**下一步 (cute_06)**：capstone 只剩没做重叠。cute_06 从 naive 一路上来——
v0 naive GEMM → v1 smem → v2 多 stage `cp.async` → **v3 TMA+WGMMA 多 stage
（cute_04 §5 骨架铺满 grid）** → **v4 Warp Specialization** → v5 Block Cluster。
终端是对标 cuBLAS。你已经拿到全部零件，cute_06 是把它们拧成一台机器。

---

> **记法**：`LN` / `TN` 里的 `N` 指"转置"（`transpose`）。A 行主、B 行主 = "TN"；
> A 转置、B 转置 = "NT"。CUTLASS 的 `gemm` API 第一个字符是 A 的 `trans`，
> 第二个是 B 的。本文全部用 TN（K 两边都连续，Tensor Core 最自然）。
