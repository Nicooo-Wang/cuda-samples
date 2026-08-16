# cute_03 练习参考解答

先自己做，卡住了再看。每题都标注了对应的 README 小节。

---

## 练习 1 — Copy_Atom 三件套 ★☆☆

**答案：`NumValSrc = 4`**

```cpp
constexpr int EX1_NUMVAL = 4;
```

算法就一步除法：

```
UniversalCopy<uint64_t> + half_t:   64 bit / 16 bit = 4 个 half
                          ~~~~~~~     ~~~~~~   ~~~~~~
                          元素类型    指令宽度  元素宽度
```

`ThrID = 1` —— `UniversalCopy` 系列都是单线程指令，不需要多线程协作。

**要点**：向量宽度由 `UniversalCopy<>` 的模板参数决定，第二个参数只负责"按什么类型数元素"。同一个 `uint64_t`：

| 元素类型 | NumVal |
|---|---|
| `float` (32b) | 2 |
| `half_t` (16b) | 4 |
| `int8_t` (8b) | 8 |

> README §2.1

---

## 练习 2 — 设计一个 TiledCopy ★★☆

```cpp
auto thr_layout = make_layout(make_shape (Int<16>{}, Int<8>{}),
                              make_stride(Int<8>{},  Int<1>{}));   // 128 线程, row-major
auto val_layout = make_layout(make_shape (Int<2>{},  Int<4>{}));   // 每线程 8 个
```

**推导过程：**

```
step 1  总量: 32*32 = 1024 个 float,  1024 / 128 线程 = 每线程 8 个
step 2  一条 128bit 指令搬 4 个 float  ->  每线程发 2 条
step 3  这 8 个怎么摆? 两维相乘 = 8, 且连续(列)方向必须是 4
        ->  val_layout = (2,4)
step 4  Tiler_MN = thr_shape * val_shape (逐维相乘)
        (32,32) = (thr_m, thr_n) * (2,4)
        ->  thr_m = 32/2 = 16     thr_n = 32/4 = 8
        ->  thr_layout = (16,8),  128 个线程  ✓
step 5  stride 用 row-major (8,1): 让相邻线程落在相邻列 = 相邻地址
```

**验证：**

```
线程数   = 16*8 = 128        ✓
Tiler_MN = (16*2, 8*4) = (32,32)   ✓
```

**常见错误：**

| 写法 | 线程数 | Tiler_MN | 问题 |
|---|---|---|---|
| `thr(16,8)` + `val(1,4)` | 128 | `(16,32)` | 每线程只 4 个，盖不满 32 行 |
| `thr(32,8)` + `val(1,4)` | **256** | `(32,32)` | 形状对了但线程数超了 |
| `thr(16,8)` + `val(4,2)` | 128 | `(64,16)` | 连续方向只有 2，凑不出 128bit |
| `thr(8,16)` + `val(4,1)` | 128 | `(32,16)` | 连续方向 1，退化成标量 |

第 3、4 行是同一个错误的两种形式：**`val_layout` 的连续（列）方向必须 ≥ 4**，否则 4 个 float 在内存里不相邻，128 bit 指令无从组装。写成连续方向 < 4 时 CuTe 会直接编译失败：

```
"TiledCopy uses too few vals for selected CopyAtom"
```

> README §5 §6.2

---

## 练习 3 — 预测 partition_S ★★☆

**答案：`{0, 1, 2, 3}`**

```cpp
constexpr int EX3_OFFSETS[4] = {0, 1, 2, 3};
```

**推导**：`val_layout = (1,4)` 表示每线程在**列**方向连续拿 4 个。thr0 位于 `thr_layout` 的 `(0,0)`，所以它的起点是 `A(0,0) = 0`，然后沿列方向走 4 个：

```
A = 16x16 row-major, 所以列方向 stride = 1

thr0 拿 (0,0) (0,1) (0,2) (0,3)
偏移    0     1     2     3          <- 4 个连续 -> 一条 LDG.E.128
```

**顺带算一下 thr1**：`thr_layout = (8,4):(4,1)` 是 row-major，thr1 在 `(0,1)`。但每个线程占 4 列，所以 thr1 的列起点是 `1*4 = 4`：

```
thr1 : 4  5  6  7
thr2 : 8  9 10 11
thr3 : 12 13 14 15
thr4 : 16 17 18 19       <- thr_layout 第 0 维走完 4 个后换行, 16 = 一整行
```

**要点**：`partition_S` 返回的是 rank-3 `((atom内), rest_m, rest_n)`。上面列的是第 0 个 mode（atom 内部那 4 个）。本题 `Tiler_MN = (8,16)`，Tensor 是 16×16，所以 `rest_m = 2, rest_n = 1` —— `copy()` 会自动跑 2 轮。

> README §6 §6.2

---

## 练习 4 — 用 copy_if 处理尾块 ★★☆

```cpp
for (int i = 0; i < int(size(tS)); ++i) {
    pred(i) = (base + int(&tS(i) - (src + base))) < n;
}
```

也可以写得更直白 —— `&tS(i)` 是绝对地址，直接和 `src + n` 比：

```cpp
pred(i) = (&tS(i) - src) < n;
```

**为什么要用指针差算下标？** 因为 `tS` 是 `partition_S` 之后的**视图**，它的 `i` 是"我这个线程的第 i 个值"，不是全局下标。两者之间的映射由 `thr_layout`/`val_layout` 决定，手推很容易错。用指针差 `&tS(i) - src` 直接问"这个元素在原数组的什么位置"，是最稳的做法。

**验证的两个方向都重要：**

```
前 100 个   -> 必须全部搬到      (谓词不能太严)
100..127    -> 一个都不能被写    (谓词不能太松)
```

只检查第一个方向的话，`pred(i) = true` 也能"通过" —— 但那就写越界了。

**这就是 capstone §8.3 尾块处理的正确做法。** capstone 里为了简单用了"退回标量循环"，代价是最后一个 block 慢；`copy_if` 能保持向量化的同时挡住越界。

> README §2.4 §8.3

---

## 练习 5 — max_common_vector ★★★

```cpp
constexpr int EX5_A = 32;   // 连续 / 连续
constexpr int EX5_B = 1;    // 连续 / stride-4
constexpr int EX5_C = 1;    // row-major / col-major
```

**逐题分析：**

**① 连续 vs 连续 → 32**

两边都是 `32:1`，整段共同连续，所以是 32。够发 128 bit（只需要 4）。

**② 连续 vs stride-4 → 1**

```
S:  0  1  2  3  4  5  6  7 ...      连续
D:  0  4  8 12 16 20 24 28 ...      步长 4

共同连续长度 = 1
```

dst 每两个元素间隔 4，一次只能搬 1 个 —— **只能标量**。注意这里不是"取两者的最小值"，而是"两边**同时**连续多长"。

**③ row-major vs col-major → 1**

这题最容易答错。两个 layout **各自**看都是"紧密无空洞"的（`size == cosize == 32`），但它们的**元素遍历顺序不同**：

```
src (4,8):(8,1)   colex 展开 k=0,1,2,3 -> 偏移 0, 8, 16, 24      (跨行)
dst (4,8):(1,4)   colex 展开 k=0,1,2,3 -> 偏移 0, 1,  2,  3      (连续)
```

`copy` 是按 colex 顺序逐个对应的（§1）。第 k 个元素在 src 里跳 8，在 dst 里跳 1 —— 步长不一致，所以共同连续长度只有 1。

**要点**：**"每个 layout 各自连续" ≠ "可以向量化"。** 向量化要求两边在**同一个遍历顺序下**同时连续。这正是 Section 01 §5 讲 `coalesce` 时那个反直觉结论的实际后果 —— row-major 的 `(4,8):(8,1)` 塌不成 `32:1`，而 col-major 的 `(4,8):(1,4)` 可以。

> README §3，Section 01 §5

---

## 练习 6 — 修一个不合并的 bug ★★★

```cpp
// 错误：col-major，相邻线程差一整行
auto thr_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));

// 正确：row-major，相邻线程差 1
auto thr_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
```

**改动只有 stride：`(1,8)` → `(4,1)`。**

**修好前后对比：**

```
col-major (1,8):                    row-major (4,1):
  thr0 -> 0                           thr0 -> 0
  thr1 -> 16                          thr1 -> 1
  thr2 -> 32                          thr2 -> 2
  thr3 -> 48                          thr3 -> 3
  相邻线程差 16(一整行)               相邻线程差 1
  32 线程 -> 32 个分散的段            32 线程 -> 一整段连续
```

**判断规则（记住这一条就够）：**

> 让 `thr_layout` 的 **stride-1 那一维**，对上 Tensor 的**连续那一维**。

本题 Tensor 是 `(16,16):(16,1)` —— 连续维是第 1 维（列）。所以 `thr_layout` 的 stride 也要让第 1 维为 1，即 `(4,1)`。

如果 Tensor 是 col-major `(16,16):(1,16)`，那连续维是第 0 维，`thr_layout` 就该反过来写成 `(1,8)`。**没有"永远用 row-major"的规则，只有"对上 Tensor 的连续维"。**

**为什么这题重要**：改错了

- 不会编译失败
- 不会运行报错
- 结果完全正确
- 只是慢（可能慢几倍）

发现它只有两条路：**读 layout 推地址**，或者 **profiler**（`ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld`，或看 `gld_efficiency`）。

> README §7

---

## 练习 7 — 向量化到底要求什么 ★★☆

```cpp
constexpr int EX7_MASK = 0b110;   // bit1 和 bit2 可以, bit0 不行
```

| layout | 能否用 128 bit atom |
|---|---|
| `make_stride(N, 1)` 全动态 | **不能**，编译失败 |
| `make_stride(N, Int<1>{})` 末维 stride 静态 | 能 |
| `make_stride(Int<16>{}, Int<1>{})` 全静态 | 能 |

**要点：决定能否向量化的是 stride，不是 shape。**

CuTe 要在编译期证明的命题是"这 4 个元素在内存里相邻"。这件事只由**被向量化那一维的 stride 等于 1** 决定 —— 一共有多少行多少列跟它无关。所以第 2 行的 shape 是运行时的 `(M,N)`，照样能发 `LDG.E.128`。

第 1 行失败的报错：

```
"Copy_Traits: src failed to vectorize into registers.
 Layout is incompatible with this CopyOp."
```

注意这里的 `1` 是个运行时 `int`，CuTe 无法在编译期知道它等于 1；写成 `Int<1>{}` 才是编译期的"1"。**这两个字面上都是 1，类型完全不同** —— 这是本题唯一的考点。

> **为什么这题重要**：很容易从"全动态编译失败"推出"要向量化就得把尺寸做成模板参数"，然后给每个矩阵尺寸编译一份 kernel。CUTLASS 的做法恰恰相反 —— `M/N/K` 全是运行时 `int`，只有连续维的 stride 钉成 `Int<1>{}`：
>
> ```cpp
> auto dA = make_stride(ldA, Int<1>{});    // sgemm_sm80.cu
> ```
>
> 官方 `tiled_copy.cu` 也一样：`make_shape(256, 512)` 是运行时的，但默认 LayoutLeft 让第 0 维 stride 恰好是静态 `_1`，于是它的 `val_layout` 取 `(4,1)` 沿 M 方向。换成 `(1,4)` 沿 N 就会失败。

> README §7.3

---

## 练习 8 — 把 layout 从 kernel 里搬到 host ★★☆

kernel 改成收 Tensor 和 TiledCopy，一行 layout 都不留：

```cpp
template <class TensorS, class TensorD, class TiledCopy>
__global__ void ex8_kernel(TensorS S, TensorD D, TiledCopy tc, int* out_sizeof) {
    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(S), thr.partition_D(D));
    if (threadIdx.x == 0) out_sizeof[0] = int(sizeof(TiledCopy));
}
```

host 侧构造，注意 stride 用真实的 `LD = 20`：

```cpp
auto lay8 = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<LD>{}, Int<1>{}));
auto tc8  = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                            make_layout(make_shape(Int<8>{}, Int<4>{}),
                                        make_stride(Int<4>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<4>{})));
ex8_kernel<<<1, 32>>>(make_tensor(make_gmem_ptr(d_s), lay8),
                      make_tensor(make_gmem_ptr(d_d), lay8), tc8, d_sz);
```

**原来为什么错**：kernel 里写死了 `make_stride(Int<N>{}, Int<1>{})`，即行 stride 16。而真实缓冲区每行 20 个 float（16 有效 + 4 填充）。于是第 1 行往后每一行都读偏了 4 个位置，把填充值 `-7` 搬了过去。

这是个真实场景 —— 带 leading dimension 的矩阵（cuBLAS 的 `ldA`）、行对齐到 128 字节的缓冲区、某个更大矩阵的子块，全都是 `stride != 宽度`。

**要点：layout 是数据的性质，不是 kernel 的性质。**

```
谁知道 stride 是 20？    分配内存的那一方 —— host
谁把它写死了？          kernel
结果                     换个 stride 就得改 kernel（甚至改模板参数列表）
```

把描述权交给 host 之后，同一个 kernel 能吃任何 stride、任何形状、任何 TiledCopy 配置。capstone 的四个版本共用一套 kernel 模板，就是靠这一点。

`sizeof(TiledCopy) == 1` 这一项在改之前也会 PASS —— 它验证的不是你改没改，而是让你确认**这样传参真的不花钱**：静态 layout 是空类型，形状信息全在类型里。

> README §4 §4.2 §4.3

---

## 练习 9 — 改 capstone ★★★

### ① TILE 大小

实测（sm_90，256 MB 数据，`NTHR = 256`）：

```
TILE = NTHR*2  ->  编译失败!
TILE = NTHR*4  ->  4182 GB/s      <- 最优
TILE = NTHR*8  ->  3842 GB/s      (-8%)
TILE = NTHR*16 ->  2692 GB/s      (-36%)
```

**第一行本身就是个知识点。** `TILE = NTHR*2` 意味着 `VEC = 2`，每线程只有 2 个 float，凑不出 128 bit：

```
"TiledCopy uses too few vals for selected CopyAtom"
```

要用 `uint128_t` atom，`TILE/NTHR` 必须 ≥ 4。

**为什么再大反而明显变慢？** 128 bit 是**单条访存指令的硬件上限**，每线程 8 个只是发 2 条指令，不会更宽。而 TILE 变大的代价是实打实的：

- block 数量减少 → SM 上驻留的 warp 变少 → 延迟没东西可掩盖
- 每线程的寄存器需求上升 → occupancy 下降

`TILE = NTHR*16` 掉 36% 就是这个原因。所以 **每线程 4 个 float = 128 bit 是甜点**：刚好用满单条指令宽度，又不牺牲并发度。

### ② 换成 uint64_t

实测：

```
float     (每线程 1 个, 32bit)  ->  2685 GB/s     0.64x
uint64_t  (每线程 2 个, 64bit)  ->  3835 GB/s     0.92x
uint128_t (每线程 4 个,128bit)  ->  4186 GB/s     1.00x   <- 打平 cudaMemcpy
```

**收益递减非常明显：**

```
1 -> 2 个:  +43%      指令发射还是瓶颈
2 -> 4 个:  + 9%      已经接近 DRAM 带宽墙
4 -> 8 个:  - 8%      纯亏（见 ①）
```

规律：**瓶颈在 1 个/线程时是指令发射，到 4 个/线程时已经变成 DRAM 带宽。** 打到带宽墙之后再优化指令毫无意义，反而要开始为 occupancy 付代价。

### ③ TILES_PER_BLOCK

实测：

```
TPB = 2  ->  4095 GB/s
TPB = 4  ->  4064 GB/s
TPB = 8  ->  4002 GB/s
```

单调下降，且全部低于纯向量版的 4182。

**解释**：double buffer 想掩盖的是 **gmem load 的延迟**。它需要有"别的事"可以在等待时做。但 memcpy 里唯一的另一件事就是 store，而 store 本身也在抢同一条 DRAM 带宽 —— **没有空闲的执行单元可以利用**。

所以增加 TPB 只是让每个 block 干更多活、block 总数变少，并发度下降，反而略慢。

**这三题共同的结论**：memcpy 是**纯带宽型**任务。一旦用 128 bit 把指令开销压下去、访存合并好，就已经贴着 DRAM 上限（这台机器约 4.2 TB/s）了。剩下的所有"高级技巧"（smem 中转、流水线、更大 tile）都只能带来负收益 —— 它们各自都要付出 occupancy 或同步的代价，却没有任何延迟可以掩盖。

**这些技巧的正确用武之地是 GEMM** —— 那里一块数据被复用 O(tile) 次，搬运可以和计算重叠，`cp.async` + double buffer 才能真正发挥作用。Section 06 会看到同样的技术带来 2-3 倍的提升。

### ④ 把 layout 挪回 kernel 里要付什么代价

现在四个版本共用 host 侧的 `mS` / `mD` / `tc_vec`，加一档 `uint64_t` 对比只需改 host **一处**：

```cpp
auto tc_64 = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, float>{}, thr_lay,
                             make_layout(Int<2>{}, Int<1>{}));
// kernel 一个字不改, 直接换参数调用
```

如果 layout 和 TiledCopy 写死在 kernel 里（尺寸走模板参数），同一件事要改：

1. kernel 里的 atom 类型
2. kernel 里的 `val_layout`
3. 因为 `VEC` 变了，`TILE/NTHR` 的关系也变 → 模板参数列表跟着变
4. 每个组合都是一份独立的 kernel 实例

**这就是 §4.3 那张表第 2 行的实际体感。** 把"数据长什么样"从 kernel 里拿出去之后，kernel 变成了一段**只管索引的纯逻辑**，参数空间的探索成本从"改代码"降到"改一行 host 声明"。

Section 04 会重度依赖这一点 —— 那里要"只换一个 smem layout 就对比 plain / padded / swizzle 三种方案"，kernel 完全不动。

> README §4.3 §8.2 §8.3
