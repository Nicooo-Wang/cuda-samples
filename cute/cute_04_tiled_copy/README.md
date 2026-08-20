# Section 04: TMA 搬运与 Smem 摆法（SM90 Hopper）

> 前置：cute_01（Layout）、cute_02（Tensor）、cute_03（Copy Atom，了解即可）。
> 本章用 SM90（H200，`-arch=sm_90a`）。目标：**把数据搬进 smem、摆好、再搬出去**，
> 全程只让一个线程发指令。

## §0 路线图

cute_03 学会的是"每个线程搬自己那份"。本章的主角是 Hopper 新加的**搬运硬件 TMA**：
一个线程描述整块搬运，硬件自己去搬。围绕它的问题链：

> 没有 TMA 时怎么搬（§1）→ 有了 TMA 怎么搬最小版（§2）→
> 反方向的 store 和边界（§3）→ 搬进 smem 之后怎么摆（§4）→
> 怎么让搬和算重叠（§5）→ 全部用上，capstone 转置（§6）

| § | 讲什么 | 代码 |
|---|---|---|
| §1 | 手写搬运基准：每线程算地址，成本清单五条 | v0 |
| §2 | TMA load：descriptor / 坐标 tensor / mbarrier，五个新概念逐个拆 | v1 |
| §3 | TMA store（fence）与越界自动处理 | v2 |
| §4 | smem 摆法：bank 冲突 → padding 死路 → swizzle → TMA 的四种模式 | v3 |
| §5 | Multi-stage：单缓冲 → Double Buffer → Super Buffer → tma_partition | v4 |
| §6 | Capstone：转置（TMA load + swizzle + TMA store） | capstone |

**怎么用这一章**：每节先读 README 的概念部分，再跑对应的 `.cu`，对照输出。
代码里每个数字都能在 README 里找到解释。

> 本机 `nvidia-smi` 把卡标成 `L20X / 8.9`，但运行时是 **cc 9.0 / 132 SM / 150GB /
> clusterLaunch=1** 的真 Hopper。所有结论都用 `-arch=sm_90a` 实测得到。

---

## §1 没有 TMA 时怎么搬：手写基准

先看没有 TMA 时你会怎么写（`cute_tiled_v0.cu`，§1.1）。任务贯穿全章：

```
gmem 里一个 M x N 的 float 矩阵, row-major, stride = (N, 1)
每个 CTA 负责一个 CM x CN 的 tile:
  1) 把 tile 搬进 smem
  2) (在 smem 里做点什么)
  3) 搬回 gmem
```

手写搬运长这样：

```cpp
for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
    int r = i / CN;                       // 自己算地址
    int c = i % CN;
    smem[r * CN + c] = in[(row0 + r) * N + (col0 + c)];   // 自己发 load
}
__syncthreads();                          // 全 CTA 栅栏
```

这一版是**成本清单**，后面每一版都会拿它对照：

| 成本 | 手写（§1） |
|---|---|
| 发指令的线程数 | 128 个，每人发自己那几条 |
| 地址计算 | 每线程每趟各算一次（`i/CN`、`i%CN`、乘 stride） |
| 边界处理 | tile 不整除时要自己写 `if (r < M && c < N)` |
| 同步 | `__syncthreads()`，全 CTA 栅栏 |
| 寄存器 | 每线程几个寄存器存地址和中转值 |

§1.2 把同一件事换成 CuTe 坐标写法（`make_tensor` + `local_tile`）—— 成本清单一条没
少，但**地址算术收进 Layout 了**。这一小步是给 §2 铺路：TMA 用的正是这套坐标写法。

> 问：有没有办法让"算地址、发指令"这件事从 128 个线程身上卸下来？

---

## §2 TMA load：一个线程描述整块

Hopper 新增的 TMA（Tensor Memory Accelerator）是**独立硬件单元**，专做
gmem↔smem 整块搬运。`cute_tiled_v1.cu` 把 §1 的 gmem→smem 一步换成 TMA，
其余一字不差。

### §2.1 两步走：host 造 descriptor，kernel 发一条指令

TMA 天生分成 host / kernel 两步：

```cpp
// ── host 侧 ──────────────────────────────────────────────
auto mIn  = make_tensor(make_gmem_ptr(d_in), make_layout(make_shape(M, N), LayoutRight{}));
auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
auto tma  = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);
//                        ^^^^^^^^^^^^^  gmem 视图   smem 摆法
//        ^^^^ 这个对象里装着 128 字节的 TMA descriptor

// ── kernel 侧 ────────────────────────────────────────────
copy(tma.with(bar), per_cta.partition_S(gtile), per_cta.partition_D(sT));
//   ^^^^^^^^^^^^^  ^^^^ 一个线程 (thread 0) 发, 整块由硬件搬
```

**descriptor**（host 侧那 128 字节）里存了 gmem 基地址、形状、stride，以及 tile
形状和 smem 摆法。kernel 里每个 CTA 只需要说"我要第几块"—— 所以只有一个线程
发一条指令，其余 127 个线程从头到尾不碰搬运。

### §2.2 kernel 里的五个新东西（逐个拆）

| # | 新东西 | 为什么 |
|---|---|---|
| 1 | `__grid_constant__ const TmaLoad tma` | 传 cuTensorMap 的硬性要求 |
| 2 | `tma.get_tma_tensor(shape)` | 源是**坐标 tensor**，不是数据 tensor |
| 3 | `__shared__ uint64_t bar` | mbarrier，TMA 完成的通知机制 |
| 4 | `set_barrier_transaction_bytes(bar, N)` | "我要等 N 个字节到位" |
| 5 | `get_slice(0)` + `partition_S/D` | TMA 的"线程数"是 1，所以 slice 0 |

**1 — `__grid_constant__`**：descriptor 太大（128B），不能按普通参数传，必须是
`__grid_constant__ const`。

**2 — 坐标 tensor 是什么**（v1 §2.2 会打印）：普通 tensor 装数据，坐标 tensor 装
`(i,j)` 坐标对：

```
普通 gmem tensor:  gmem_ptr[32b](0x7f..) o (256,128):(128,1)   <- 有指针
坐标 tensor:       ArithTuple(_0,_0) o (256,128):(_1@1,_1@0)   <- 只有坐标
```

`local_tile` 切出来的 `gtile` 是一块坐标。硬件拿坐标去 descriptor 里换地址。
**为什么必须这样**：用普通 tensor 切片，tile 超出矩阵时会切出**越界的裸指针**，
一读就崩；而坐标可以越界 —— 硬件看到越界坐标就跳过。这是 §3.3 边界自动处理的
底层原因。

**3 — mbarrier（异步事务屏障）**：64 位，放在 smem 里。和 `__syncthreads` 的区别：
后者是全 CTA 栅栏，强制所有人停在同一行；mbarrier 是"等一个条件"—— 这里等的是
"TMA 搬完了 N 个字节"。只有 1 个线程发 TMA，所以初始化时 `initialize_barrier(bar, 1)`。

**4 — transaction bytes**：`set_barrier_transaction_bytes(bar, tma_transaction_bytes)`
告诉屏障"这一轮我要等这么多字节"。TMA 搬完会往屏障上"销账"，账齐了等待者放行。

**5 — get_slice(0)**：`make_tma_copy` 造出来的对象里，TMA 被描述成"1 个线程搬
CM*CN 个元素"的 TiledCopy（`ValLayout: (_1,1024)`）。所以 `get_slice(0)` 取第 0 个
（唯一一个）slice，`partition_S/D` 照常工作。

### §2.3 同步的三个动作和 phase

mbarrier 在 TMA load 里出现三次，各干一件事：

```cpp
initialize_barrier(bar, 1);              // 开张: 1 个参与者
set_barrier_transaction_bytes(bar, N);   // 记账: 这一轮要收 N 个字节
wait_barrier(bar, 0);                    // 等账收齐
```

mbarrier 没有"计数器归零"的概念，它用一个 **phase bit** 表示"第几轮"：

```
初始化后     phase = 0
收齐一轮后   自动翻成 1
再收齐一轮   翻回 0
...
```

所以 wait 时要传"我在等的这一轮的 phase"：**第 k 次使用传 k & 1**（v1 §2.3 用
同一个 barrier 连搬两块演示）。传错的症状是**死锁**，不是算错 —— 等一个永远不会
到来的翻转。

```
wait_barrier(bar, t & 1);   // 第 0 轮等 phase 0, 第 1 轮等 phase 1
```

### §2.4 成本清单对照

| 成本 | 手写（§1） | TMA（§2） |
|---|---|---|
| 发指令线程数 | 128 个 | **1 个** |
| 地址计算 | 每线程各算 | **host 侧 descriptor 算好** |
| 边界处理 | 自己写 predicate | **硬件自动**（§3.3） |
| 同步 | `__syncthreads` | **mbarrier 按字节数等** |
| 寄存器 | 存地址和中转值 | **几乎为 0** |

> 问：只有 load 一个方向？搬回去那一步呢？边界不整除怎么办？

---

## §3 TMA store 与边界

`cute_tiled_v2.cu` 把搬回 gmem 那一步也换成 TMA。**load 和 store 最重要的区别：
同步方向反了。**

```
TMA load   gmem -> smem   数据搬完之后才能用   -> 事后等: mbarrier
TMA store  smem -> gmem   数据要搬之前就写好   -> 事前挡: fence

load :  发起 --------> [硬件搬] --------> wait_barrier --> 读 smem
store:  写 smem --> fence --> 发起 --------> [硬件搬] --> (可选 wait)
```

### §3.1 store 的写法

```cpp
__syncthreads();      // 1) 等全 CTA 写完 smem
tma_store_fence();    // 2) 保证这些写对 TMA 硬件 (异步 proxy) 可见
                      //    少了这行是竞态: 可能搬出去半新半旧的数据
if (threadIdx.x == 0) {
    copy(tma_store, per.partition_S(sT), per.partition_D(gtile));
    //  ^^^^^^^^^^ 没有 .with(bar) —— store 不用 mbarrier
    tma_store_arrive();   // 提交这一批 store
}
tma_store_wait<0>();  // 等到 0 个未完成 (要复用 smem 时必须有)
```

`tma_store_fence()` 包装的是 `fence.proxy.async.shared::cta`：它建立的是**异步代理**
（TMA 硬件）视角的可见性。换 `__threadfence_block()` 不行 —— 那是普通代理视角，
保证不了 TMA 看到你的写。方向也反了：`partition_S` 作用在 smem 上，
`partition_D` 作用在 gmem 坐标上。

load / store 对照：

| | TMA load | TMA store |
|---|---|---|
| 方向 | gmem → smem | smem → gmem |
| 同步机制 | mbarrier（等字节数） | **fence（挡在发起之前）** |
| 同步时机 | 搬完之后等 | 发起之前挡 |
| copy 写法 | `copy(tma.with(bar), S, D)` | `copy(tma, S, D)` |
| partition_S | gmem 坐标 | smem |
| partition_D | smem | gmem 坐标 |

### §3.2 完整一趟

`load -> 计算 -> store` 串起来（v2 §3.2）。整个 kernel 里搬运指令就两条，都由
thread 0 发；其余 127 个线程做中间的"计算"（这里是一个平方，代表真实负载）。
注意：**"用了 TMA 线程就空了"这个说法不成立** —— 省掉的是"算地址、发搬运指令"
这件事，不是线程本身。

两个方向各要一个 descriptor：

```cpp
auto tma_load  = make_tma_copy(SM90_TMA_LOAD{},  mIn,  slay);
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mOut, slay);
```

一个 descriptor 绑死一个 gmem 张量 + 一个方向，不能共用。

### §3.3 边界：硬件自己兜

v0 手写碰到不整除必须自己写 predicate：

```cpp
if (row0 + r < M && col0 + c < N) smem[...] = in[...];
else                              smem[...] = 0;
```

TMA 不用写。v2 §3.3 用 M=40, N=36（右下角 CTA 只有 8x4 有效）实测：界内元素 =
原值，**界外元素被硬件填 0**，且 kernel 里零 predicate。这正是 §2.2 坐标 tensor
的直接好处：坐标越界合法，硬件跳过不读。

唯一的硬性要求在 gmem 一侧：**除最内层外，每一维的 stride 必须是 16 字节的整数
倍**（TMA 按 16 字节盒子访存）。不满足时先把矩阵 pad 到合适宽度。

> 问：数据搬进 smem 了。但**怎么摆**决定了读它的人快不快 —— 下节讲。

---

## §4 搬进 smem 之后怎么摆

`cute_tiled_v3.cu`。v1/v2 的 smem 都是 plain row-major —— 搬运没问题，但**读的人**
可能很慢。这一节回答"该摆成什么样"，以及"TMA 允许怎么摆"。

### §4.1 32 个 bank：为什么按列读慢 32 倍

smem 硬件按 4 字节轮流分给 32 个 bank：

```
float 下标 :   0    1    2  ...   31 |  32   33  ...
bank      :   0    1    2  ...   31 |   0    1  ...
              +------ 一轮 32 个 ------+   +- 绕回来 -+
```

规则只有一条：32 个 lane 落 32 个不同 bank → 一个周期；N 个 lane 落同一 bank →
拆成 N 次（N-way conflict）。

对 `(32,32):(32,1)` 的 tile：行 stride = 32 = bank 数，所以**下一行的同一列偏移
+32，bank 号纹丝不动**：

```
按行读 sT(0, 0..7):   偏移 0 1 2 3 4 5 6 7     bank 0 1 2 3 4 5 6 7   ✓
按列读 sT(0..7, 0):   偏移 0 32 64 96 ...      bank 0 0 0 0 ...      ✗ 32-way
```

什么时候按列读？转置、沿列方向的 reduction、以及某些 MMA 取数模式。

### §4.2 padding：能修，但 SM90 上是死路

把行 stride 32 改成 33：列偏移变 `0, 33, 66, ...`，每行错开 1 个 bank，冲突消除。
但三条代价：

| 代价 | 具体 |
|---|---|
| 多占 smem | cosize 1055 > size 1024（+3.0%），开数组要按 cosize |
| 破坏对齐 | 行首不再 128B 对齐，而 TMA 要求 128B 对齐 |
| **WGMMA 编译期拒绝** | SM90 Tensor Core 不接受这种 layout（cute_05 会亲手撞） |

第三条是硬墙：WGMMA 把 smem 地址和摆法编码进 descriptor，能表达的只有几种规范
形式，padding 不在其中。所以在 SM90 上消冲突只有一条路：**swizzle**。

### §4.3 swizzle 怎么映射

`Swizzle<B,M,S>` 不改 shape、不多占一个字节，只改"逻辑坐标 → 偏移"的映射函数，
做法是把偏移的某几个比特异或到另几个比特上：

```
B = 参与异或的比特数
M = 最低几位不动 (保护 2^M 个元素保持连续)
S = 异或的距离

swz(off) = off XOR ( ((off >> S) & mask_B) << M )
```

`Swizzle<5,0,5>` 手算 `(r=3, c=5)`：`off = 101`，取出高 5 位（=r=3），和低 5 位
异或：`c XOR r = 5 XOR 3 = 6` → `swz = 3*32+6 = 102`（v3 打印验证一致）。
一句话：**用行号去打乱列号**，`c_new = c XOR r`。

映射表（前几行）：

```
 r=1: 33 32 35 34 ...     两两交换
 r=2: 66 67 64 65 ...     每 2 个一组交换
 r=3: 99 98 97 96 ...     每 4 个一组倒转
```

第 0 列的偏移序列从 `0 32 64 96...`（全撞 bank 0）变成 `0 33 66 99...`（全不同）。
三个不变量（实测）：**cosize 不变**、**是双射**（不丢数据）、**行读仍不冲突**。

### §4.4 M 参数：消冲突和向量化的权衡

M 保护最低 M 位不参与异或，即 **2^M 个相邻元素保持连续**。实测（32x32 float）：

| swizzle | 列读最坏 | 行内最短连续 | 128-bit 向量 |
|---|---|---|---|
| plain | 32-way | 32 个 float | 可用 |
| `Sw<5,0,5>` | **1-way** | **1 个 float** | **编译失败** |
| `Sw<4,1,4>` | 2-way | 2 个 float | 编译失败 |
| `Sw<3,2,3>` | 4-way | **4 个 float** | **可用** |

最容易犯的错：挑冲突最少的 `Sw<5,0,5>`。它把冲突消得最干净，但 M=0 意味着每个
元素单独被打乱，128-bit 指令（一次搬 4 个连续 float）编译期就挂。规则：

> **先定 M = 你要用的向量宽度，再让 B、S 去消冲突。先保住向量化，再谈消冲突。**

### §4.5 交给 TMA 时：只有四种模式可选

§4.3/§4.4 的 `Swizzle<B,M,S>` 是**完全自由**的 —— 只要你自己写搬运代码。
但 descriptor 里存 swizzle 的只有**几个比特**，所以交给 TMA 的 layout 只有四种
（CuTe 封成四个原子，名字里的数字 = 一行占多少字节）：

```
float SW128 = Sw<3,4,3> o (8,32):(32,1)    一行 128 字节
float SW64  = Sw<2,4,3> o (8,16):(16,1)    一行  64 字节
float SW32  = Sw<1,4,3> o (8,8) :(8,1)     一行  32 字节
float INTER = Sw<0,4,3> o (8,4) :(4,1)     一行  16 字节
```

**M 全 = 4，S 全 = 3** —— 不是巧合，是硬件只支持这些。TMA 接受的 Swizzle 全集
（CuTe 源码 `copy_traits_sm90_tma_swizzle.hpp`）：

```
M = 4, S = 3, B = 0..3     <- 就是上面四个原子
M = 5, S = 2, B = 2
M = 6,        B = 2
```

手写的 `Sw<5,0,5>`（M=0）、`Sw<3,2,3>`（M=2）**一个都不在里面**，传给
`make_tma_copy` 直接编译期失败：

```
static assertion failed: "Unsupported layout swizzle."
static assertion failed: "Expected 128b=16B=(2^4)B to 512b=64B=(2^6)B base swizzle."
```

为什么 M 必须 ≥ 4：M=4 表示"最低 16 个元素不参与异或"。float 16 个 = 64 字节，
half 16 个 = 32 字节 —— TMA 一次访存的最小粒度就在这个量级，比它更细的打乱
硬件表达不了。§4.4 那条"先保向量宽度"的规则，在 TMA 这里是**强制**的。

### §4.6 用 TMA：官方原子，以及"逻辑连续 vs 物理字节序"

一个必须先说清的事实（v3 §4.6 打印验证）：**TMA 把 XOR 应用在物理地址上**。
SW128 搬完，smem 的裸字节序第 2 行是：

```
smem[32..63] = 132 133 134 135 128 129 130 131 140 141 142 143 ...
```

4 个一组重排 —— 但**用逻辑坐标 `sT(r,c)` 读，拿到的仍是正确数据**，因为 layout
里带着同一份 swizzle 映射，读取时自动逆映射。所以：

- 逻辑坐标读取 → 永远正确（这也是 §4.6 五种 layout 全部"TMA 正确"的原因）；
- 物理字节序 → 被 XOR，直接按地址读裸数组会看到乱序。

对比 §4.3/§4.4：**手写搬运时 swizzle 是逻辑层的重排（改 `s(r,c)` 的偏移）；
TMA 搬运时 swizzle 是硬件层的重排（物理字节序变，逻辑坐标不变）。** 两条路语义
不同 —— 这也是 TMA 只接受那四种的原因：能写进硬件的比特就那么几个。

descriptor 的 swizzle 字段的用途：① 决定硬件搬运的粒度/对齐（SW128 最宽，搬运
效率最高，所以工程上选能用的最宽的）；② 当 TMA 直接喂 WGMMA 时（cute_05/06），
保证 smem 摆成 WGMMA descriptor 认的规范形式 —— 那时它是一份**合同**。

怎么选模式：唯一硬约束是**内层维度必须被原子的内层长度整除**：

```
CN (float) | SW128(32) | SW64(16) | SW32(8) | INTER(4)
    32     |    可     |    可    |   可    |   可
    16     |   不可    |    可    |   可    |   可
     8     |   不可    |   不可   |   可    |   可
```

违反是编译期报错（`tile_to_shape: block shape does not divide...`）。
**实用规则：选能用的里面一行字节数最大的。** 本例 CN=32 float → SW128。

> 问：搬一块、摆一块都会了。但搬和算是**串行**的 —— 搬的时候计算单元闲着。
> 怎么让它们重叠？

---

## §5 Multi-stage：让搬运和计算重叠

`cute_tiled_v4.cu`。负载是一个真实的流水线形状：`A` 按列切成 NTILE 块，依次搬进
smem、全 CTA 读出来做 FMA 累加。**注意：这一版不需要 WGMMA**（那是 cute_05 的
事）—— 计算就用普通 FMA，只要"算"比"搬"慢，重叠的收益就显形。

规模：528 CTA × 128 tile × 8KB = 554MB（HBM 驻留，L2 装不下）—— 只有数据不在
L2 里，TMA 的延迟才真实，重叠才有意义。

### §5.1 单缓冲：两个引擎各闲一半

只有一个 smem buffer，每轮必须"等搬完 → 算 → 等读完"，时间线：

```
TMA   : [搬0]      [搬1]      [搬2]
计算  :      [算0]      [算1]      [算2]
```

两个引擎轮流干活，各闲一半。（v4 实测 0.146 ms，作为 1.00x 基准。）

### §5.2 Double Buffer：2 个 buffer 轮换

两个 buffer，`搬 k+1` 可以和 `算 k` 同时进行。需要**两组 barrier**：

```
full[s]   生产者→消费者: "buffer s 已装满"   按字节数等 (TMA 专用)
empty[s]  消费者→生产者: "buffer s 已用完"   按到达数等 (NTHR 个线程各 arrive 一次)
```

代码骨架：

```cpp
// prologue: 填满全部 STAGES 个 buffer (tile s -> buffer s)
// 之后 tile k 永远落在 buffer k % STAGES
for (int s = 0; s < STAGES; ++s) if (threadIdx.x == 0) {
    set_barrier_transaction_bytes(full[s], tx_bytes);
    copy(tma.with(full[s]), ..., sT(_, _, s));
}

for (int k = 0; k < NTILE; ++k) {
    int s = k % STAGES;
    int phase = (k / STAGES) & 1;          // buffer s 第几轮被用
    wait_barrier(full[s], phase);          // 等装满
    ...算...;
    arrive_barrier(empty[s]);              // 通知用完
    int knext = k + STAGES;
    if (knext < NTILE) {
        wait_barrier(empty[s], phase);     // 先等空, 再覆盖
        if (threadIdx.x == 0) copy(tma.with(full[s]), ..., sT(_, _, s));
    }
}
```

phase 公式从 §2.3 的 `k & 1` 变成 `(k / STAGES) & 1` —— 同一个 barrier 每隔
STAGES 轮用一次。**prologue 填满后不要"预推进"任何状态**（cute_06 用
`PipelineState` 时踩过的坑）：full/empty 都从 phase 0 开始，写错是死锁。

v4 实测：单缓冲 0.150 ms → Double Buffer 0.131 ms（**1.11x**）。

### §5.3 Super Buffer：更多 stage，以及 48KB 台阶

| STAGES | smem | 时间 (ms) | 相对单缓冲 |
|---|---|---|---|
| 1 | 8 KB | 0.150 | 1.00x |
| 2 | 16 KB | 0.131 | 1.11x |
| 3 | 24 KB | 0.130 | 1.12x |
| 4 | 32 KB | 0.132 | 1.10x |

stage 越多，TMA 提前搬得越远，容忍的延迟越长 —— 但 smem 线性涨、收益递减，
**stage 不是越多越好**。再往上还有一道硬台阶：**静态 `__shared__` 上限 48KB**
（0xc000），超过直接 ptxas 报 `uses too much shared data`。要吃到 H200 的 227KB
必须换**动态 smem**：

```cpp
extern __shared__ char smem[];                    // kernel 里
cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);  // host
k<<<grid, block, bytes>>>();                      // launch 时申请
```

这就是官方例子（`wgmma_tma_sm90.cu`）用 `SharedStorage` 结构体 + 动态 smem 的原因
—— SM90 GEMM 的 smem 动辄 100KB 以上。

### §5.4 tma_partition：cute_05/06 用的写法

v1–v4 用的都是 `make_tma_copy + get_slice(0) + partition_S/D`（把 TMA 当"1 线程的
TiledCopy"）。但 CUTLASS 的 GEMM 代码（cute_05 capstone、cute_06 v3+）用另一套：
`make_tma_atom + tma_partition`。两个必须知道的差异：

1. **smem layout 必须带 PIPE 维**（`(CM,CN,1)`），建 atom 时传切片 `slay(_,_,Int<0>{})`；
2. **partition 用 `tma_partition`**，得到 gmem 侧 `(TMA,)` 和 smem 侧 `(TMA,PIPE)`；
   发指令的 lane 用 `elect_one_sync()` 自己选。

```cpp
auto tma = make_tma_atom(SM90_TMA_LOAD{}, mA, slay3(_, _, Int<0>{}), tile_shape);

auto p  = tma_partition(tma, Int<0>{}, Layout<_1>{},
                        group_modes<0, 2>(sT), group_modes<0, 2>(gt));
auto tAg = get<0>(p);   // (TMA,)      gmem 侧
auto tAs = get<1>(p);   // (TMA, PIPE) smem 侧
```

**为什么 CUTLASS 用这套**：multicast（一个 CTA 搬给整簇）需要把"第几个 CTA"也
当成 slice 表达，TiledCopy 的线程映射不够用；`tma_partition` 把 slice 选择完全
交给你。**写法不同，搬的还是同一件事** —— 一个线程一条指令、硬件搬、barrier 等。

> 问：全部工具都有了，串起来做个完整的活 —— 转置。

---

## §6 Capstone：转置

`cute_tiled_capstone.cu`。转置是检验 smem 摆法最干净的任务：**读 gmem 合并、
写 gmem 合并，所有冲突都被挤到"读 smem"这一步**。六版，每版只换一件事：

```
t1  手写 + plain smem        32-way 冲突的起点
t2  手写 + SW128 smem        手写搬运也吃 swizzle 红利
t3  TMA load + plain smem    搬运换硬件, 转置逻辑不动
t4  TMA load + SW128 smem    §4 的写法
t5  TMA load + TMA store     本章终点: 两条搬运指令, 其余线程只做转置
t6  不整除的矩阵             硬件自动兜边界
```

实测（1024x1024 float，tile 32x32）：

```
t1  手写 + plain smem       列读 32-way   正确
t2  手写 + SW128 smem       列读 16-way   正确
t3  TMA load + plain smem   列读 32-way   正确
t4  TMA load + SW128 smem   列读 16-way   正确
t5  TMA load + TMA store    列读 16-way   正确
t6  越界 1000x640           正确
```

注意列读只从 32-way 降到 16-way：tile 32x32 只用到 SW128 原子 8 行模式的一半；
tile 更大（128x128，cute_05 之后）会降到 8-way。**plain 的 32-way 是实打实的最坏
情况，这才是 swizzle 要消的。**

### §6.1 t5 的转置怎么做

t5 的转置用**手写 + TMA store**，不是"转置视图 + TMA store"。为什么？（这是
§4.6 的直接应用 —— TMA store 按 smem **物理字节序**搬出，视图只改逻辑坐标
不改物理字节序，搬出去是原样的行，不是转置后的列。实测过，错的。）

所以"交给硬件"的部分是**搬运**，转置本身仍是普通代码：

```cpp
__shared__ float tmp[CM * CN];
for (int i = threadIdx.x; i < CM * CN; i += blockDim.x)
    tmp[i] = sT(i % CN, i / CN);     // 读: 转置着读
__syncthreads();                     // 读和写之间必须隔一道栅栏
for (int i = threadIdx.x; i < CM * CN; i += blockDim.x)
    sT(i / CN, i % CN) = tmp[i];     // 写: 原位置
```

不能原地 `sT(r,c) = sT(c,r)`：目标 `(r,c)` 的源 `(c,r)` 可能刚被别的线程改过。
先全 CTA 读进临时数组、隔一道 `__syncthreads`、再写 —— 这个两阶段模式以后
（cute_06 的 GEMM epilogue 之类）还会见到。

### §6.2 t6 越界版

`1000x640`（1000 % 32 = 8），grid (32,20)。最右一列 CTA 的界外部分由硬件 load
填 0，转置后再由 store 跳过不写。**全程零 predicate，结果正确。**

---

## §7 代码地图 + 练习 + 交接

| 文件 | 内容 | 对应 |
|---|---|---|
| `cute_tiled_v0.cu` | 手写搬运基准（每线程算地址）+ CuTe 坐标写法 | §1 |
| `cute_tiled_v1.cu` | TMA load：descriptor、坐标 tensor、mbarrier、phase | §2 |
| `cute_tiled_v2.cu` | TMA store（fence）、完整一趟、越界自动处理 | §3 |
| `cute_tiled_v3.cu` | bank 模型、padding、swizzle 映射、TMA 四种模式 | §4 |
| `cute_tiled_v4.cu` | Multi-stage：单缓冲 → Double → Super → tma_partition | §5 |
| `cute_tiled_capstone.cu` | 转置六版（含 TMA 两端 + 越界） | §6 |
| `exercises/ex.cu` | 8 道可自检练习 | §7 |

跑：`make run`（依次跑 v0..capstone），`make ex`（练习）。

## 练习

进 `exercises/`，把 `ex.cu` 里的 TODO 填完，`make run` 自动判 PASS/FAIL。
参考解答在 `exercises/solutions.md`。（8 道，后两题给足提示。）

**练习 1 — 数 bank ★☆☆**（§4.1）
`(32,32):(32,1)` 的 float tile，一个 warp 读第 5 列时最热的 bank 被请求几次？

**练习 2 — 手算 swizzle 映射 ★★☆**（§4.3）
`Swizzle<5,0,5>` 作用在 `(32,32):(32,1)` 上，手算 `swz_off(6, 3)`。按 §4.3 四步走。

**练习 3 — 选 M 保住向量化 ★★★**（§4.4）
128-bit atom 搬 32×32 float tile，`Sw<5,0,5>`/`Sw<4,1,4>`/`Sw<3,2,3>` 哪些能用？
注意冲突最少的那个未必能用。

**练习 4 — 选对 TMA 模式 ★★☆**（§4.6）
`CN=32` 的 float tile，SW128/64/32/INTER 哪些可用？选了会怎样？

**练习 5 — TMA 语义 ★★☆**（§2 §4.6）
(a) 手写 swizzle 和 TMA 的 swizzle 语义差在哪？(b) plain smem 交给 TMA 能搬对吗？

**练习 6 — 修一个转置 bug ★★★**（§6.1）
转置结果正确但慢。**只改 smem layout 一行**，让列读 ≤ 16-way。

**练习 7 — 手写一段 TMA 搬运 ★★★**（§2）
用 TMA 把 GM×GK 的一个 TM×TK tile 搬进 smem。五个 TODO 对应 §2.2 的五个新东西，
提示都在 v1 §2.1 的 `copy_tma_kernel` 里。**盖住参考实现默写一遍才是目的**。

**练习 8 — TMA Double Buffer ★★★**（§5.2）
把单缓冲 TMA 循环改成 2-stage。三个 TODO；**最易错的一行是 phase 公式** ——
`(k / STAGES) & 1` 写错成 `k & 1` 的症状是死锁（练习里跑 5 次专门抓它）。

### 交接

| 到 | 学什么 | 这里已经给了什么 |
|---|---|---|
| cute_05 | MMA / WGMMA / TMA 拼成算子 | §4.6 四种模式、§5.4 tma_partition 写法 |
| cute_06 | 完整 GEMM：多 stage 铺满 grid + Warp Spec + Cluster | §5 的 full/empty 双 barrier 骨架、§5.3 动态 smem |
| cute_07 | Persistent Block / tile 调度 | §5 的流水线心智模型 |

这三个后续章节正是 H200 五大特性（TMA、WGMMA、Warp Spec、Persistent、Cluster）
的补全：本章已经实际跑通 TMA + multi-stage，WGMMA 在 05 接上，Warp Spec 和
Persistent 在 06/07 各自主场。
