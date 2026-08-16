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

本章讲清楚四件事：atom 内部是什么、**这些东西该写在 host 还是 kernel 里**（§4，决定你的代码长什么样）、怎么铺到线程上、以及**铺错了会怎样**（§7，这部分最实用）。

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

## 4. 谁在 host 上做，谁在 kernel 里做

这一节讲的是**写法惯例**，不是某个 API。它决定了你的 CuTe 代码长什么样，所以放在动手之前。

先看 CUTLASS 官方 `examples/cute/tutorial/tiled_copy.cu` 的 kernel 签名：

```cpp
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  Tensor thr_tile_S = thr_copy.partition_S(tile_S);
  ...
}
```

`Tensor` 和 `TiledCopy` 都是**参数**。它们在 `main()` 里构造好，整个 kernel 里没有一次 `make_layout`。`sgemm_sm80.cu` 更极端 —— 连 smem layout、TiledMMA、`cp.async` 的 atom 全是 host 构造后传进去的：

```cpp
// host 侧
auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{}, ...);
TiledMMA  mmaC  = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{}, ...);

gemm_device<<<dimGrid, dimBlock, smem_size>>>(prob_shape, cta_tiler,
                                              A, dA, sA, copyA, ...,
                                              C, dC, sC, mmaC, ...);
```

### 4.1 这个分工是什么

```
┌─ host ────────────────────────────────────┐
│  数据长什么样      make_layout / make_tensor
│  用什么指令搬      Copy_Atom
│  谁搬哪一份        make_tiled_copy / make_tiled_mma
│  smem 怎么摆       tile_to_shape / Swizzle
└───────────────────┬───────────────────────┘
                    │  按值传参
┌─ kernel ──────────▼───────────────────────┐
│  我这个 block 负责哪块    local_tile(blockIdx)
│  我这个线程负责哪几个    get_slice(threadIdx)
│  搬                       copy() / gemm()
└───────────────────────────────────────────┘
```

一句话：**host 描述，kernel 索引。** kernel 里出现的运行时量只有 `blockIdx` 和 `threadIdx`。

### 4.2 为什么按值传 Tensor 和 TiledCopy 不心疼

因为静态 layout 是**空类型** —— 形状信息全在类型里，运行时零字节：

```
Layout<Shape<_8,_16>,Stride<_16,_1>>         sizeof = 1    <- 空类型
Copy_Atom<UniversalCopy<float>, float>       sizeof = 1    <- 空类型
TiledCopy (128bit, 32 线程)                  sizeof = 1    <- 空类型
Tensor<gmem_ptr<float>, 静态 layout>         sizeof = 8    <- 只有那个指针
```

传一个 `TiledCopy` 传的是零字节；传一个静态 `Tensor` 只传了里面的指针。对比一下运行时的：

```
Layout<Shape<int,int>,Stride<int,int>>       sizeof = 16
Tensor<gmem_ptr<float>, 动态 layout>         sizeof = 24
```

`sizeof == 1` 是 C++ 对空类型的规定（对象必须有地址），实际做参数时被完全优化掉。

### 4.3 为什么不在 kernel 里构造

在 kernel 里写 `make_tiled_copy(...)` 能编译、也能跑对 —— 都是编译期的东西，编译器会消掉。但有三个实际问题：

| | host 构造 | kernel 内构造 |
|---|---|---|
| 形状能不能 `print` 出来看 | 能，`print(tc)` 一行 | 要 `if (thread0()) print(...)` 再跑一遍 |
| 换一组参数试 | 改 host 一行，kernel 不动 | 每种组合一份 kernel |
| 静态检查 | `static_assert(size(copy) == size(mma))` 写在 kernel 开头，参数一进来就查 | 没有可查的对象 |

第 2 点最实在。capstone 里四个版本共用同一个 kernel 模板，换的只是 host 传进去的 layout；如果 layout 写死在 kernel 里，就得复制四份 kernel。**这也是 Section 04 之后能"只改一个 smem layout 就对比三种方案"的前提。**

> **反过来说，什么必须在 kernel 里？** 依赖 `blockIdx` / `threadIdx` 的一切：`local_tile`、`get_slice`、`partition_S/D`、`copy`。以及 `__shared__` 数组本身 —— 但它的 **layout** 仍然可以从 host 传进来（`sgemm_sm80.cu` 就是这么做的，capstone v3/v4 也一样）。

### 4.4 一个例外：CuTe 不管你怎么分配内存

`__shared__ float raw[...]` 的**大小**必须是编译期常量，所以它写在 kernel 里。惯例是按 layout 的 `cosize` 开：

```cpp
template <int TILE, class SLayout, ...>
__global__ void k(..., SLayout slay, ...) {
    __shared__ float raw[cosize_v<SLayout>];        // 大小由 layout 决定
    auto sT = make_tensor(make_smem_ptr(raw), slay);  // layout 从 host 来
    ...
}
```

**用 `cosize` 而不是 `size`**：带 padding 的 layout（Section 04 会讲）最后一个元素的偏移超出 `size`，按 `size` 开会越界。这里 layout 无 padding，两者相等。

> 对应源码：`cute_copy_v1.cu` §1（打印上面那张 sizeof 表）

---

## 5. make_tiled_copy：把 atom 铺到整个 block

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

### 5.1 标量版：32 线程 × 1 值

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

### 5.2 向量版：32 线程 × 4 值

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

## 6. partition_S / partition_D：把 Tensor 切给当前线程

```cpp
auto thr = tc.get_slice(threadIdx.x);   // 取出"我"这一份
auto tS  = thr.partition_S(S);          // 我要读的
auto tD  = thr.partition_D(D);          // 我要写的
copy(tc, tS, tD);
```

### 6.1 返回的是 rank-3，不是 rank-1

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

### 6.2 观察实际地址：合并访存长什么样

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

## 7. thr_layout 写错会怎样（本章最实用的一节）

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

### 7.3 向量化要求 stride 静态，不是 shape 静态

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

很容易由此得出结论"要向量化就得把整个 layout 写成编译期的"。**这个结论是错的**，而且错得有代价 —— 它会让你把本该动态的矩阵尺寸也塞进模板参数，每个尺寸编译一份 kernel。

真正的要求只有一条：**被向量化那一维的 stride 是静态的。** shape 可以是运行时值。同一个 128 bit TiledCopy 沿列方向搬，三种写法的结果：

| gmem layout | 结果 |
|---|---|
| `make_stride(N, 1)` 全动态 | **编译失败** |
| `make_stride(N, Int<1>{})` 只有末维 stride 静态 | 通过 |
| `make_stride(Int<N>{}, Int<1>{})` 全静态 | 通过 |

中间那行是关键：`shape` 是运行时的 `(M, N)`，照样能发 `LDG.E.128`。

道理也说得通 —— CuTe 要证明的是"这 4 个元素在内存里相邻"，这件事只由**那一维的 stride 等于 1** 决定，跟一共有多少行多少列无关。

> **官方 example 正是这么写的。** `tiled_copy.cu` 用 `make_shape(256, 512)`（运行时 `int`）却能做 128 bit 搬运，因为 `make_layout` 默认是 **LayoutLeft（列主序）**，第 0 维 stride 恰好是静态的 `_1`：
>
> ```
> make_layout(make_shape(M, N))  ==  (256,512):(_1,256)
>                                            ~~~~ 静态
> ```
>
> 它的 `val_layout` 是 `(4,1)` —— 沿 **M** 方向，也就是 stride 为 `_1` 的那一维。换成 `(1,4)` 沿 N 就会编译失败。

**实践建议：尺寸该动态就动态，但连续那一维的 stride 一定写成 `Int<1>{}`。** CUTLASS 的 GEMM 就是这个风格 —— `M/N/K` 全是运行时 `int`，而 `dA = make_stride(ldA, Int<1>{})` 把连续维钉成静态。

这仍然印证 Section 01 说的"能编译期确定就用 `Int<N>`"，只是要精确到位：**关键的那个编译期常量是 stride，不是 shape。**

> 对应源码：`cute_copy_v1.cu` §4 §5

---

## 8. Capstone：用 Copy_Atom 写高带宽 memcpy

`cute_copy_capstone.cu` 做四个版本，和 `cudaMemcpy` 对照。数据量 256 MB（读+写 537 MB）。

### 8.1 四个版本

**v1 naive** —— 每线程一个 float，grid-stride：

```cpp
for (int i = idx; i < n; i += stride) dst[i] = src[i];
```

**v2 vectorized** —— TiledCopy + 128 bit atom，每线程 4 个 float。按 §4 的分工，host 构造、kernel 只索引：

```cpp
// host
auto mS = make_tensor(make_gmem_ptr(d_src),
                      make_layout(make_shape(N), make_stride(Int<1>{})));   // N 是运行时的
auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                          make_layout(Int<NTHR>{}, Int<1>{}),
                          make_layout(Int<TILE/NTHR>{}, Int<1>{}));

// kernel
auto gS = local_tile(mS, Shape<Int<TILE>>{}, make_coord(blockIdx.x));   // 我这个 block 的那块
auto thr = tc.get_slice(threadIdx.x);                                   // 我这个线程的那份
copy(tc, thr.partition_S(gS), thr.partition_D(gD));
```

注意 `mS` 的 shape 是运行时的 `N`，只有 stride 是 `Int<1>{}` —— 按 §7.3，这就够向量化了。

**v3 smem 中转** —— gmem → smem 用 `cp.async`，再 smem → gmem。smem 的 **layout 也从 host 传进来**，kernel 里只按 `cosize` 开数组：

```cpp
// host
auto slay = make_layout(Int<TILE>{}, Int<1>{});

// kernel
__shared__ float raw[cosize_v<SLayout>];
auto sT = make_tensor(make_smem_ptr(raw), slay);

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

v4 的 smem layout 多了一维，用来选缓冲区：

```cpp
auto slay_2buf = make_layout(make_shape(Int<TILE>{}, Int<2>{}));   // (TILE, PIPE)
...
copy(tc_load, ..., thr_load.partition_D(sT(_, buf)));              // sT(_, buf) 取第 buf 块
```

**这个 PIPE 维就是多 stage 流水线的最小形态。** Section 06 的 GEMM 里它会变成 `(BM, BK, PIPE)`，`PIPE` 取 3~5，而取哪一块的写法完全一样 —— `sA(_,_,pipe_index)`。v4 只是 `PIPE = 2` 的特例。

### 8.2 实际结果（H 系列，sm_90）

```
version                time(ms)         GB/s      ok
cudaMemcpy D2D            0.128         4183     yes
naive (scalar)            0.149         3601     yes
TiledCopy 128bit          0.128         4186     yes
smem staging              0.129         4156     yes
double buffer             0.132         4064     yes

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

### 8.3 尾块处理

capstone 取的 `N` 恰好被 `TILE` 整除（`static_assert` 卡住了这一点），所以四个版本都不用管边界 —— 这是为了让代码只讲一件事。

真实场景里 `N` 不会这么配合，有三种处理方式：

| 办法 | 代价 | 用在哪 |
|---|---|---|
| 尾块退回标量循环 | 最后一个 block 慢，只影响 1/grid | 最省事 |
| `copy_if` + 谓词（§2.4） | 多算一个谓词 tensor | CUTLASS 的标准做法 |
| 分配时向上取整到 `TILE` 的倍数 | 多占一点显存 | 你能控制分配时 |

第二种最通用，练习 4 会让你写一遍。官方 `tiled_copy_if.cu` 用的也是它，配合 `make_identity_tensor` 拿到全局坐标来判断越界。

---

## 9. 代码怎么读

三个程序是上面内容的可执行版本。**建议先读完本 README，再打开代码。**

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `cute_copy_v0.cu` | §1 §2 §3 | `copy(S,D)`、Copy_Atom 三件套、`copy_if`、`max_common_vector`、`recast` |
| `cute_copy_v1.cu` | §4 §5 §6 §7 | host/kernel 分工、`make_tiled_copy`、`partition_S/D` 的 rank-3 形状、thr_layout 写错的后果 |
| `cute_copy_capstone.cu` | §8 | 四版 memcpy + `cudaMemcpy` 对照 |

代码小节与本文的对应：

| 代码小节 | README |
|---|---|
| v0 §1 `copy(S,D)` | §1 |
| v0 §2 三件套 | §2 §2.1 §2.2 |
| v0 §3 显式 atom 的形状要求 | §2.3 |
| v0 §4 `copy_if` | §2.4 |
| v0 §5 `max_common_vector` | §3 |
| v0 §6 `recast` | §3.2 |
| v1 §1 host 描述 / kernel 索引、sizeof 表 | §4 |
| v1 §2 标量 TiledCopy | §5.1 §6 |
| v1 §3 向量 TiledCopy | §5.2 §6.2 |
| v1 §4 thr_layout 写错 | §7 |
| v1 §5 向量化要求 stride 静态 | §7.3 |

> **v0 是个例外：它在 host 上跑 CuTe。** 那是为了单独讲 atom 的属性和 `copy_if` 的语义，不涉及线程分工 —— 用 host 代码最省事。从 v1 起全部是 kernel，遵循 §4 的分工。

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

**这题的意义**：这是你在真实代码里最可能犯、且最难发现的错误。做完之后，回头看 §7 那张表。

### 练习 7 — 向量化到底要求什么 ★★☆
`ex7_kernel` 收一个 host 传进来的 Tensor，用 128 bit atom 沿连续方向搬。

三个候选 layout 里选出**能编译且能向量化**的那些，填进 `EX7_MASK`（bit0/1/2）：

```cpp
0: make_layout(make_shape(M, N),             make_stride(N, 1))
1: make_layout(make_shape(M, N),             make_stride(N, Int<1>{}))
2: make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}))
```

**先想清楚再填**：CuTe 需要在编译期证明的是"这 4 个元素相邻"，这件事由 shape 决定还是由 stride 决定？

填完之后把你选中的每个 layout 真的传进 kernel 跑一遍 —— 检查会告诉你哪个编译不过。

### 练习 8 — 把 layout 从 kernel 里搬到 host ★★☆
`ex8_kernel` 是按"kernel 内构造"的老写法写的：`make_layout` / `make_tiled_copy` 全在 kernel 里，尺寸靠模板参数 `<M, N>` 传。

把它改成 §4 的分工：

1. kernel 签名改成收 `TensorS S, TensorD D, TiledCopy tc`，去掉模板参数 `<M, N>`；
2. 在 `ex8()` 里构造 layout / Tensor / TiledCopy，传进去；
3. kernel 里只留 `get_slice` + `partition` + `copy`。

改完后检查会验证两件事：结果仍然正确，且 `sizeof(TiledCopy) == 1`（说明你传的确实是空类型）。

**这题的意义**：这是本章最该形成的肌肉记忆。做完之后你写任何 CuTe kernel 都会先问"这个东西该在 host 还是 kernel"。

### 练习 9 — 改 capstone ★★★
1. 把 `TILE` 从 `NTHR*4` 改成 `NTHR*2`、`NTHR*8`、`NTHR*16`，各跑一次记录带宽。
   `NTHR*2` 会发生什么？（先想，再编译 —— 错误信息本身就是答案。）
   剩下几个哪个最好？为什么不是越大越好？
2. 把 v2 的 atom 从 `uint128_t` 依次换成 `uint64_t`（每线程 2 个）和 `float`（每线程 1 个），
   记录三个带宽。相邻两档的提升幅度一样吗？说明瓶颈在哪一档发生了转移。
3. v4 的 `TILES_PER_BLOCK` 改成 2 和 8，观察变化。结合 §8.2 第 3 点解释你看到的现象。
4. 现在四个版本共用 host 侧的 `mS`/`mD`/`tc_vec`。试着把 v2 的 layout 改回写在 kernel 里
   （尺寸走模板参数），然后再想加一档 `uint64_t` 的对比 —— 需要改几处？
   这就是 §4.3 那张表第 2 行的实际体感。

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
| **host 描述，kernel 索引** | layout/atom/TiledCopy 在 host 构造，kernel 只用 blockIdx/threadIdx | CuTe 代码的基本骨架 |
| 静态 layout 是空类型 | `sizeof == 1`，按值传参零成本 | 上一条之所以可行 |
| `make_tiled_copy` | atom + thr_layout + val_layout | 一张"谁搬哪一份"的分工表 |
| `Tiler_MN` | `thr_shape * val_shape` | 一次覆盖多大一块 |
| `partition_S/D` | 返回 `((atom内),rest_m,rest_n)` | `copy()` 自动遍历 rest |
| **thr_layout 写错** | **结果正确但不合并，无任何报错** | 最常见的性能 bug |
| 向量化的前提 | **连续那一维的 stride 静态**（shape 可以动态） | 尺寸不必塞进模板参数 |
| `cp.async` | 搬运不占寄存器和指令流 | GEMM 流水线的基础 |
| smem 的 PIPE 维 | `(TILE, PIPE)`，`sT(_, i)` 取第 i 块 | 多 stage 流水线的写法 |

**下一章：Section 04 — TiledCopy。** 本章的 `make_tiled_copy` 只是入门用法。Section 04 会讲多级分块（block tile → warp tile → thread tile）、`ldmatrix` 这类给 Tensor Core 喂数据的特殊 atom，以及 swizzle 如何在 layout 层面消掉 bank conflict。
