# Section 03: Copy Atom

## 本章要解决的问题

Section 02 结束时，我们能回答"每个线程负责哪些数据"了。但还差最后一步：**怎么把这些数据真的搬过来。**

最朴素的写法是逐元素赋值：

```cpp
dst[i] = src[i];
```

这行代码能跑，但它把两件事拧在了一起：**搬什么**（下标）和**怎么搬**（指令）。而 GPU 上"怎么搬"的选择多得惊人：

| 想要的效果 | 实际要发的指令 |
|---|---|
| 一次搬 4 个 float | `LDG.E.128` / `STG.E.128` |
| gmem → smem 不占寄存器 | `cp.async` (SM80+) |
| smem → 寄存器，喂给 Tensor Core | `ldmatrix` (SM75+) |
| gmem → smem 由硬件搬整块 | TMA (SM90+) |

这些指令**对数据的摆放方式各有硬性要求**：`LDG.E.128` 要求 16 字节对齐且连续；`ldmatrix` 要求 32 个线程按一个特定的诡异顺序各持 8 个 half。写错了不会报错，只会变慢，或者算出错误结果。

CuTe 的回答是：**把"一条搬运指令 + 它对数据摆放的要求"打包成一个对象，叫 Copy_Atom。**

```
Copy_Atom = 一条指令  +  这条指令要求的数据摆放方式
```

然后用 `make_tiled_copy` 把这个 atom "铺"到整个 thread block 上，得到 **TiledCopy** —— 一张"谁搬哪一份"的分工表。之后 `copy(tc, src, dst)` 一行就够了，指令选择、向量宽度、线程分工全都由类型系统决定。

本章讲清楚三件事：atom 内部是什么、怎么铺到线程上、以及**铺错了会怎样**（这部分最实用）。

---

## 1. 最朴素的 copy

`copy` 的输入输出都是 Tensor（Section 02），不是裸指针：

```cpp
float src[16], dst[16];
auto lay = make_layout(Int<16>{}, Int<1>{});
auto S = make_tensor(make_gmem_ptr(src), lay);
auto D = make_tensor(make_gmem_ptr(dst), lay);

copy(S, D);        // 就这一行
```

不指定任何 atom 时，CuTe 自己挑指令 —— 它会看 S/D 的 layout，能向量化就向量化（§3 讲判据）。

要求只有一条：**`size(S) == size(D)`**。形状可以不同：

```cpp
copy(S_4x8, D_8x4);    // OK, 都是 32 个元素，按各自的 colex 顺序对应
```

对应关系就是 Section 01 的 colex 序 —— `S(k)` 搬到 `D(k)`。

> 对应源码：`cute_copy_v0.cu` §1

---

## 2. Copy_Atom 内部是什么

一个 `Copy_Atom<Op, T>` 由两个参数决定：

- **`Op`** —— 用哪条指令（`UniversalCopy<uint128_t>`、`SM80_CP_ASYNC_CACHEGLOBAL<...>`、`SM75_U32x4_LDSM_N`…）
- **`T`** —— 按什么元素类型来数（`float`、`half_t`…）

它对外回答三个问题：

```
ThrID        这次原子操作要几个线程一起干？
NumValSrc    每个线程从 src 读几个 T？
NumValDst    每个线程往 dst 写几个 T？
```

**只有这三个都对上，指令才能发出去。** 对不上是编译期错误，不是运行时错误。

### 2.1 标量 vs 向量：宽度写在哪

```cpp
Copy_Atom<UniversalCopy<float>,     float>     // ThrID=1, NumVal=1
Copy_Atom<UniversalCopy<uint128_t>, float>     // ThrID=1, NumVal=4
Copy_Atom<UniversalCopy<uint128_t>, half_t>    // ThrID=1, NumVal=8
```

```
UniversalCopy<uint128_t> + float:      128 bit / 32 bit = 4 个 float
                                       ~~~~~~~~   ~~~~~~
                                       指令宽度   元素宽度

  thr0 一条指令搬 4 个 float:
     addr:  0    1    2    3
          +----+----+----+----+
          | a  | b  | c  | d  |     一条 LDG.E.128
          +----+----+----+----+
```

**关键：向量宽度写在 `UniversalCopy<>` 的模板参数里，不是 atom 的第二个参数。** 第二个参数只是"按什么类型数元素"。这是最容易搞混的一处 —— `Copy_Atom<UniversalCopy<float>, float>` 和 `Copy_Atom<UniversalCopy<uint128_t>, float>` 差了 4 倍带宽。

### 2.2 AutoVectorizingCopy：把宽度交给 CuTe

```cpp
Copy_Atom<AutoVectorizingCopy, float>     // NumVal = 1 ?
```

它的 `NumVal` 打印出来是 1，但这只是个**占位值**。真实宽度在 `copy()` 时按 src/dst 的 layout 现场推导（用 §3 的 `max_common_vector`）。

这也是 `copy(S, D)` 不写 atom 时的默认行为。`DefaultCopy` 则是假设对齐只有 8 bit 的保守版本。

**什么时候手写宽度、什么时候用 Auto？**

| | 用法 |
|---|---|
| `AutoVectorizingCopy` | layout 是编译期已知的常规情形，让 CuTe 推 |
| `UniversalCopy<uint128_t>` | 你确定能向量化，想让编译失败来兜底（推不出来就报错，而不是静默降级成标量） |

第二种更适合写性能敏感的 kernel：**宁愿编译失败，也不要静默变慢。**

### 2.3 显式传 atom 时的形状要求

```cpp
copy(Copy_Atom<UniversalCopy<uint128_t>, float>{}, S, D);
```

这时 S/D 必须是 **rank-1，且 `size` 恰好等于 `NumVal`**：

```
NumVal = 4:
  make_layout(Int<4>{}, Int<1>{})   size=4   -> OK
  make_layout(Int<8>{}, Int<1>{})   size=8   -> 编译失败
```

实践中你几乎不会手写这一层 —— `TiledCopy`（§4）会自动把 Tensor 切成正好 `NumVal` 大小的片。

> **一个真实的坑：宽向量 atom 属于 device 代码。**
>
> `UniversalCopy<uint128_t>` 会把 `float*` 重新解释成 `uint128_t*`。这在 host 代码里违反 C++ 的 **strict aliasing** 规则 —— `-O2` 以上编译器会认为"写 `uint128_t` 不可能影响 `float` 数组"，从而把这次写**整个优化掉**，结果静默出错（我在写这一章时就踩到了：`-O0/-O1` 结果正确，`-O2/-O3` 全是 0）。
>
> 所以宽向量搬运一律在 kernel 里做。本章 v0 只在 host 上演示标量和 `copy_if`，宽向量全部放在 v1 的 kernel 里。

### 2.4 copy_if：边界处理

真实 kernel 里 M/N 很少刚好被 tile 整除。`copy_if(pred, S, D)` 只在谓词为真的位置搬：

```cpp
auto pred = make_tensor<bool>(make_layout(Int<8>{}, Int<1>{}));
for (int i = 0; i < 8; ++i) pred(i) = (i % 2 == 0);
copy_if(pred, S, D);
```

```
pred :   1    0    1    0    1    0    1    0
src  : 100  101  102  103  104  105  106  107
dst  : 100   .   102   .   104   .   106   .        <- '.' 是原值，没被碰过
```

**这是处理"最后一个不完整 tile"的标准手段** —— 谓词写成"全局下标 < n"，越界的位置就自动挡住了。练习 4 会让你写一遍。

> 对应源码：`cute_copy_v0.cu` §2 §3 §4

---

## 3. 能不能向量化：max_common_vector

向量化的前提是 **src 和 dst 在内存里都有一段连续、且长度一致**。这个"共同的最大连续长度"就是 `max_common_vector(S, D)`。

```cpp
max_common_vector(连续32, 连续32)      // 32   -> 可以用 128bit
max_common_vector(连续32, stride-2)    //  1   -> 只能标量搬
```

判据只看两边**同时**连续多长：

```
S:  0  1  2  3  4  5  6  7      连续
D:  0  2  4  6  8 10 12 14      步长 2

     共同连续长度 = 1        ->  一次只能搬 1 个
```

这就是 `AutoVectorizingCopy` 内部选宽度的依据，也解释了 **Section 01 为什么反复强调 `coalesce`**：能塌缩成 `N:1` 才谈得上向量化。一个 layout 塌不掉，说明它有跨步或空洞，向量化自然无从谈起。

`max_common_layout(S, D)` 给出的是 layout 形式的同样信息。

### 3.2 recast：换元素类型看同一块内存

向量化的另一种写法 —— 把 Tensor 重新解释成更宽的类型：

```cpp
auto T = make_tensor(make_gmem_ptr(p),
                     make_layout(make_shape(Int<4>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{})));   // (4,8):(8,1)

recast<float4>(T);      // (4,2):(2,1)   8 列变 2 列，每列 4 个 float
```

```
原  T (4,8):(8,1)                recast<float4> (4,2):(2,1)
  8 个 float 一行                  2 个 float4 一行
  +--+--+--+--+--+--+--+--+        +--------+--------+
  |  |  |  |  |  |  |  |  |   ->   |float4  |float4  |
  +--+--+--+--+--+--+--+--+        +--------+--------+
```

要求：**最后一维必须连续、且长度能被整除**，否则编译失败。

> 对应源码：`cute_copy_v0.cu` §5 §6

---

## 4. make_tiled_copy：把 atom 铺到整个 block

一个 atom 只描述"一次操作"。真实 kernel 里有几百个线程，需要一张分工表。

```cpp
auto tc = make_tiled_copy(atom, thr_layout, val_layout);
```

三个参数各管一件事：

| 参数 | 含义 | 决定什么 |
|---|---|---|
| `atom` | 一次原子操作搬多少 | 发什么指令 |
| `thr_layout` | 线程怎么排 | **访存是否合并** |
| `val_layout` | 每线程一次拿几个、怎么排 | 能否凑出向量宽度 |

得到的 `TiledCopy` 有一个关键属性：

```
Tiler_MN = thr_shape * val_shape        (逐维相乘)
```

即"这个 TiledCopy 一次覆盖多大一块"。

### 4.1 标量版：32 线程 × 1 值

```cpp
auto atom = Copy_Atom<UniversalCopy<float>, float>{};
auto thr  = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
auto val  = make_layout(make_shape(Int<1>{}, Int<1>{}));
auto tc   = make_tiled_copy(atom, thr, val);
// Tiler_MN = (8,4)     线程数 = 32
```

`thr_layout = (8,4):(4,1)` 是 **row-major**，含义是"线程号沿列方向增长"：

```
        j=0   j=1   j=2   j=3        thr_layout (8,4):(4,1)
      +-----+-----+-----+-----+      格子里是线程号
i=0   | t0  | t1  | t2  | t3  |
      +-----+-----+-----+-----+      相邻线程 -> 相邻列 -> 相邻地址
i=1   | t4  | t5  | t6  | t7  |
      +-----+-----+-----+-----+
i=2   | t8  | t9  | t10 | t11 |
      +-----+-----+-----+-----+
  ...
i=7   | t28 | t29 | t30 | t31 |
      +-----+-----+-----+-----+
```

### 4.2 向量版：32 线程 × 4 值

只改两处 —— atom 换成 128 bit，`val_layout` 在**连续方向**给 4 个：

```cpp
auto atom = Copy_Atom<UniversalCopy<uint128_t>, float>{};
auto val  = make_layout(make_shape(Int<1>{}, Int<4>{}));    // 列方向 4 个
auto tc   = make_tiled_copy(atom, thr, val);
// Tiler_MN = (8,16)    线程数 = 32   <- 覆盖面积变成 4 倍
```

线程数没变，但覆盖面积翻了 4 倍：

```
        列 0-3      列 4-7      列 8-11     列 12-15
      +----------+----------+----------+----------+
i=0   |   t0     |   t1     |   t2     |   t3     |     每个格子 = 4 个连续 float
      +----------+----------+----------+----------+     = 一条 LDG.E.128
i=1   |   t4     |   t5     |   t6     |   t7     |
      +----------+----------+----------+----------+
  ...
```

**`val_layout` 必须放在连续方向。** 如果写成 `(4,1)`（行方向 4 个），那 4 个元素在内存里间隔一整行，凑不出 128 bit —— CuTe 会编译失败或退化成标量。

> 对应源码：`cute_copy_v1.cu` §1 §2

---

## 5. partition_S / partition_D：把 Tensor 切给当前线程

```cpp
auto thr = tc.get_slice(threadIdx.x);   // 取出"我"这一份
auto tS  = thr.partition_S(S);          // 我要读的
auto tD  = thr.partition_D(D);          // 我要写的
copy(tc, tS, tD);
```

### 5.1 返回的是 rank-3，不是 rank-1

这一点第一次见会困惑。`partition_S` 的结果形如：

```
((atom内部), rest_m, rest_n)
```

- **第 0 个 mode** —— 这个线程在**一次 atom 操作**里要搬的那几个（大小 = `NumVal`）
- **`rest_m` / `rest_n`** —— Tensor 比 `Tiler_MN` 大出来的部分，需要重复几轮

举例：`Tensor = 8×16`，`Tiler_MN = (8,4)`，那么 `rest_n = 16/4 = 4`：

```
Tensor 8x16,  Tiler 8x4

 +--------+--------+--------+--------+
 | tile 0 | tile 1 | tile 2 | tile 3 |     rest_n = 4
 +--------+--------+--------+--------+
   一个 TiledCopy 覆盖一个 tile，
   copy() 自动循环 4 轮把 4 个 tile 走完
```

**`copy(tc, tS, tD)` 会自动遍历 `rest_*` 这些 mode。** 所以 `Tiler_MN` 比 Tensor 小完全没问题 —— 不需要你手写外层循环。这也是为什么上面标量版（tiler 只有 8×4）能把整个 8×16 搬完。

### 5.2 观察实际地址：合并访存长什么样

标量版（`thr_layout` row-major，每线程 1 个值）每个线程拿到的偏移：

```
thr0 : 0   4   8  12          <- 4 轮，每轮跨一个 tile(4 列)
thr1 : 1   5   9  13
thr2 : 2   6  10  14
thr3 : 3   7  11  15
```

**同一轮里，thr0..thr3 拿的是 0,1,2,3 —— 连续地址。** 一个 warp 的 32 个线程覆盖一整段连续内存，合并成整齐的访存事务。

向量版（每线程 4 个值）：

```
thr0 : 0   1   2   3          <- 一条 128bit 指令
thr1 : 4   5   6   7
thr2 : 8   9  10  11
thr3 : 12  13  14  15
```

每个线程内部 4 个连续（凑成 128 bit），线程之间间隔 4，warp 整体仍然是一整段连续区间。**这是理想形态：既向量化，又合并。**

> 对应源码：`cute_copy_v1.cu` §1 §2

---

## 6. thr_layout 写错会怎样（本章最实用的一节）

把 `thr_layout` 从 row-major 改成 col-major，其他一个字不改：

```cpp
// 原来（正确）
auto thr_rm = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
// 改成（错误）
auto thr_cm = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
```

现在线程号沿**行**方向增长：

```
        j=0   j=1   j=2   j=3        thr_layout (8,4):(1,8)
      +-----+-----+-----+-----+
i=0   | t0  | t8  | t16 | t24 |      相邻线程 -> 相邻行 -> 地址差一整行!
      +-----+-----+-----+-----+
i=1   | t1  | t9  | t17 | t25 |
      +-----+-----+-----+-----+
i=2   | t2  | t10 | t18 | t26 |
      +-----+-----+-----+-----+
  ...
```

每个线程拿到的偏移变成：

```
thr0 :  0   4   8  12
thr1 : 16  20  24  28          <- 和 thr0 差了 16 = 一整行
thr2 : 32  36  40  44
thr3 : 48  52  56  60
```

**后果：**

| | row-major thr_layout | col-major thr_layout |
|---|---|---|
| 结果正确性 | 正确 | **同样正确** |
| 相邻线程地址间隔 | 1 | 16（一整行） |
| 一个 warp 的访存事务 | 少数几个连续段 | 32 个分散的段 |

**这是本章最需要记住的一点：`thr_layout` 写错不会报错，结果依然完全正确，只是慢。** 没有任何编译期或运行期信号会提示你 —— 只能靠看 layout 或用 profiler（`ncu` 的 `gld_efficiency` / sectors per request）发现。

> 判断方法很简单：**让 `thr_layout` 的 stride-1 那一维，对上 Tensor 的连续那一维。** row-major Tensor（stride `(N,1)`）配 row-major thr_layout（stride `(k,1)`）。

### 6.3 向量化需要编译期 stride

另一个真实的坑。下面这段会**编译失败**：

```cpp
// M, N 是运行时 int
auto lay = make_layout(make_shape(M, N), make_stride(N, 1));
auto S = make_tensor(make_gmem_ptr(src), lay);
copy(tc_vectorized, thr.partition_S(S), ...);
```

```
error: static assertion failed with
"Copy_Traits: src failed to vectorize into registers.
 Layout is incompatible with this CopyOp."
```

原因：CuTe 必须在**编译期**证明这 4 个元素连续，才敢发 128 bit 指令。运行时的 `N` 做不到这个证明。

修法是把尺寸变成编译期常量（模板参数）：

```cpp
template <int M, int N, class TiledCopy>
__global__ void kernel(const float* src, float* dst, TiledCopy tc) {
    auto lay = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    ...
}
```

**这就是 Section 01 说"能编译期确定就用 `Int<N>`"的实际后果** —— 不是风格建议，是能不能向量化的硬性前提。

> 对应源码：`cute_copy_v1.cu` §3 §4

---

## 7. Capstone：用 Copy_Atom 写高带宽 memcpy

`cute_copy_capstone.cu` 做四个版本，和 `cudaMemcpy` 对照。数据量 256 MB（读+写 537 MB）。

### 7.1 四个版本

**v1 naive** —— 每线程一个 float，grid-stride：

```cpp
for (int i = idx; i < n; i += stride) dst[i] = src[i];
```

**v2 vectorized** —— TiledCopy + 128 bit atom，每线程 4 个 float：

```cpp
auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>,
                          make_layout(Int<NTHR>{}, Int<1>{}),
                          make_layout(Int<VEC>{},  Int<1>{}));
auto thr = tc.get_slice(threadIdx.x);
copy(tc, thr.partition_S(S), thr.partition_D(D));
```

**v3 smem 中转** —— gmem → smem 用 `cp.async`，再 smem → gmem：

```cpp
auto atom_async = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{};
copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT));
cp_async_fence();       // 标记"这一批发完了"
cp_async_wait<0>();     // 等它落地
__syncthreads();        // smem 对全 block 可见
copy(tc_store, thr_store.partition_S(sT), thr_store.partition_D(gD));
```

`cp.async` 的意义是**搬运不经过寄存器、不占指令流**，发出去就能干别的事。这三行 fence/wait/sync 的配合是固定套路，Section 06 的 GEMM 会反复用到。

**v4 double buffer** —— 两块 smem 交替，搬下一块的同时写出这一块：

```
issue_load(tile 0, buf 0)
loop t:
    issue_load(tile t+1, buf ^ 1)     <- 先把下一块发出去
    cp_async_wait<1>()                <- 等最早那批落地（还有 1 批在飞）
    store(tile t, buf)                <- 写出这一块
```

`cp_async_wait<N>` 的 `N` 是"允许还有几批在飞"。发了下一块就是 `wait<1>`，最后一轮没有后续就是 `wait<0>`。

### 7.2 实际结果（H 系列，sm_90）

```
version                time(ms)         GB/s      ok
cudaMemcpy D2D            0.129         4177     yes
naive (scalar)            0.149         3594     yes
TiledCopy 128bit          0.128         4181     yes
smem staging              0.130         4143     yes
double buffer             0.132         4067     yes

相对 cudaMemcpy:
  naive (scalar)       0.86x
  TiledCopy 128bit     1.00x
  smem staging         0.99x
  double buffer        0.97x
```

**这个结果值得仔细读，因为它和直觉相反：**

1. **naive → vectorized：+16%。** 唯一的改动是 128 bit atom。标量版发 4 倍的指令数才搬同样的数据，指令发射成了瓶颈。

2. **vectorized 打平 `cudaMemcpy`（1.00x）。** 这是本章的主要结论：**用 CuTe 写的 memcpy 能达到 cuBLAS 级别的库函数水平**，而代码只有几行，且形状/类型全部参数化。

3. **smem 中转和 double buffer 反而更慢。** memcpy 是**纯带宽型**任务 —— 数据从 gmem 读一次、写一次，没有任何复用。多绕一跳 smem 只是增加延迟和同步开销，没有任何东西可以被隐藏。

**第 3 点是本章故意留的反面教材。** `cp.async` + double buffer 是 GEMM 的标准套路，但它的价值在于**掩盖计算的延迟**：GEMM 里一块数据要被复用 O(tile) 次，搬运和计算可以重叠。memcpy 没有计算，也就没有可重叠的对象。

> **工具是有适用范围的。** 看到一个技术在某处有效，不等于它到处有效 —— 要问"它掩盖的是什么开销，我这里有这个开销吗"。

### 7.3 尾块处理

`N` 不一定被 `TILE` 整除。capstone 里用的是最简单的办法 —— 尾块退回标量：

```cpp
int base = blockIdx.x * TILE;
if (base + TILE > n) {
    for (int i = base + threadIdx.x; i < n; i += NTHR) dst[i] = src[i];
    return;
}
```

代价是最后一个 block 慢一点，但只影响 1/grid 的工作量。更优雅的做法是 `copy_if` + 谓词（§2.4），练习 4 会让你写一遍。

---

## 8. 代码怎么读

三个程序是上面内容的可执行版本。**建议先读完本 README，再打开代码。**

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `cute_copy_v0.cu` | §1 §2 §3 | `copy(S,D)`、Copy_Atom 三件套、`copy_if`、`max_common_vector`、`recast` |
| `cute_copy_v1.cu` | §4 §5 §6 | `make_tiled_copy`、`partition_S/D` 的 rank-3 形状、thr_layout 写错的后果 |
| `cute_copy_capstone.cu` | §7 | 四版 memcpy + `cudaMemcpy` 对照 |

代码小节与本文的对应：

| 代码小节 | README |
|---|---|
| v0 §1 `copy(S,D)` | §1 |
| v0 §2 三件套 | §2 §2.1 §2.2 |
| v0 §3 显式 atom 的形状要求 | §2.3 |
| v0 §4 `copy_if` | §2.4 |
| v0 §5 `max_common_vector` | §3 |
| v0 §6 `recast` | §3.2 |
| v1 §1 标量 TiledCopy | §4.1 §5 |
| v1 §2 向量 TiledCopy | §4.2 §5.2 |
| v1 §3 thr_layout 写错 | §6 |
| v1 §4 编译期 stride | §6.3 |

```bash
make run              # 全部跑一遍
./cute_copy_v1        # 单独跑
make ex               # 做练习
```

> **多卡机器上跑 capstone 请指定一张空闲卡**，否则带宽数字会被邻居干扰：
> `CUDA_VISIBLE_DEVICES=<idle> ./cute_copy_capstone`

---

## 练习

按顺序做。答案填在 `exercises/ex.cu` 的 TODO 处，`make ex` 会自动检查并打印 PASS/FAIL。参考解答在 `exercises/solutions.md`。

### 练习 1 — Copy_Atom 三件套 ★☆☆
不运行代码，回答 `Copy_Atom<UniversalCopy<uint64_t>, half_t>` 的 `NumValSrc` 是多少？

把答案填进 `EX1_NUMVAL`，运行验证。（提示：`uint64_t` 是 64 bit，`half_t` 是 16 bit。）

### 练习 2 — 设计一个 TiledCopy ★★☆
128 个线程搬一块 **32×32 的 float**，要用满 128 bit 向量指令，且 `Tiler_MN` 恰好是 `(32,32)`。

按顺序推：

1. 总共 `32*32 = 1024` 个 float，128 个线程 → 每线程几个？
2. 一条 128 bit 指令只搬 4 个 float → 每线程要发几条？
3. 这些值怎么摆？`val_layout` 两维相乘要等于第 1 步的答案，**且连续方向必须是 4**。
4. 由 `Tiler_MN = thr_shape * val_shape` 反推 `thr_layout`（记得 row-major）。

### 练习 3 — 预测 partition_S ★★☆
`A = 16×16` float（row-major），`tc` = 32 线程 × 每人 4 个（128 bit，列方向），`thr_layout = (8,4):(4,1)`。

**先在纸上算** thr0 拿到的 4 个元素偏移分别是多少，填进 `EX3_OFFSETS` 再运行。

### 练习 4 — 用 copy_if 处理尾块 ★★☆
长度 **100** 的数组，用 32 线程 × 每人 4 个（一块 128）搬。100 不是 128 的倍数，最后一块越界。

用 `copy_if` + 谓词挡住越界位置。要求：前 100 个全部搬到，`100..127` 一个字节都不能被写。

提示：第 `i` 个元素的全局下标可以用指针差算出来 —— `base + (&tS(i) - (src + base))`。

### 练习 5 — max_common_vector ★★★
先预测下面三个的结果，再运行：

1. `max_common_vector(连续 32, 连续 32)`
2. `max_common_vector(连续 32, stride-4 的 32)`
3. `max_common_vector((4,8):(8,1), (4,8):(1,4))` ← row-major vs col-major

第 3 题想清楚：两个 layout 各自都"连续"吗？它们的**共同**连续长度是多少？

### 练习 6 — 修一个不合并的 bug ★★★
`ex6_kernel` 里的 `thr_layout` 写成了 col-major，结果正确但访存完全不合并。

改成 row-major，让 thr0..thr3 的首地址变成 `0,1,2,3`。

**这题的意义**：这是你在真实代码里最可能犯、且最难发现的错误。做完之后，回头看 §6 那张表。

### 练习 7 — 改 capstone ★★★
1. 把 `TILE` 从 `NTHR*4` 改成 `NTHR*2`、`NTHR*8`、`NTHR*16`，各跑一次记录带宽。
   `NTHR*2` 会发生什么？（先想，再编译 —— 错误信息本身就是答案。）
   剩下几个哪个最好？为什么不是越大越好？
2. 把 v2 的 atom 从 `uint128_t` 依次换成 `uint64_t`（每线程 2 个）和 `float`（每线程 1 个），
   记录三个带宽。相邻两档的提升幅度一样吗？说明瓶颈在哪一档发生了转移。
3. v4 的 `TILES_PER_BLOCK` 改成 2 和 8，观察变化。结合 §7.2 第 3 点解释你看到的现象。

> 跑之前记得 `CUDA_VISIBLE_DEVICES=<idle>`，否则邻居负载会把结论淹掉。

---

## 小结

| 概念 | 一句话 | 在后面章节的作用 |
|---|---|---|
| `Copy_Atom<Op,T>` | 一条指令 + 它要求的数据摆放 | 所有搬运的基本单元 |
| `ThrID` / `NumVal` | 几个线程协作 / 每人搬几个 | 对不上就编译失败 |
| 向量宽度 | 写在 `UniversalCopy<>` 里，不是第二个参数 | 4 倍带宽的差别 |
| `AutoVectorizingCopy` | 宽度交给 CuTe 现场推 | 常规情形的默认选择 |
| `max_common_vector` | src/dst 共同的最大连续长度 | 能否向量化的判据 |
| `copy_if` | 带谓词的搬运 | 尾块 / 边界处理 |
| `make_tiled_copy` | atom + thr_layout + val_layout | 一张"谁搬哪一份"的分工表 |
| `Tiler_MN` | `thr_shape * val_shape` | 一次覆盖多大一块 |
| `partition_S/D` | 返回 `((atom内),rest_m,rest_n)` | `copy()` 自动遍历 rest |
| **thr_layout 写错** | **结果正确但不合并，无任何报错** | 最常见的性能 bug |
| 编译期 stride | 向量化的硬性前提 | `Int<N>` 不是风格问题 |
| `cp.async` | 搬运不占寄存器和指令流 | GEMM 流水线的基础 |

**下一章：Section 04 — TiledCopy。** 本章的 `make_tiled_copy` 只是入门用法。Section 04 会讲多级分块（block tile → warp tile → thread tile）、`ldmatrix` 这类给 Tensor Core 喂数据的特殊 atom，以及 swizzle 如何在 layout 层面消掉 bank conflict。
