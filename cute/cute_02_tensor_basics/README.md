# Section 02: Tensor Basics

## 本章要解决的问题

Section 01 讲的 Layout 只是一个**函数** —— 输入坐标，输出整数。它不知道数据在哪，也碰不到数据。

```cpp
auto A = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
A(1, 3);        // 11 —— 一个整数，不是元素值
```

真正写 kernel 时我们要回答两个 Layout 单独回答不了的问题：

1. **这些偏移是相对谁的？** —— 需要一个指针。
2. **这块数据要分给谁处理？** —— 需要把大 Tensor 切成块、再分给线程。

本章就补上这两块：**Tensor**（指针 + Layout）和**两个切分算子**（`local_tile` / `local_partition`）。学完之后，你就有了写 GPU kernel 所需的完整数据视图工具。

---

## 1. Tensor = 指针 + Layout

```cpp
float* p = ...;
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
auto T = make_tensor(make_gmem_ptr(p), layout);

T(1, 3) = 42.0f;   // 等价于 p[layout(1,3)] = p[11] = 42.0f
```

Layout 负责算偏移，指针负责提供基址。打印出来能看到两部分：

```
T = gmem_ptr[32b](0x7ffc...) o (_4,_8):(_8,_1)
    ~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~
    指针(带元素位宽)        o   Layout
```

`o` 就是 composition 的记号 —— **Tensor 本身就是"指针 ∘ Layout"这个复合**。

### 指针必须显式包装

```cpp
make_tensor(p, layout)                    // 编译错误
make_tensor(make_gmem_ptr(p), layout)     // 正确
make_tensor(make_smem_ptr(p), layout)     // shared memory
make_tensor(make_rmem_ptr(p), layout)     // register
```

**为什么不能直接传裸指针？** 因为 CuTe 需要在**类型**里知道这块内存住在哪 —— gmem / smem / rmem 的可用指令完全不同（TMA 只能对 gmem、`ldmatrix` 只能对 smem）。后面 Section 03/04 的 Copy 会根据这个类型自动选指令。裸指针丢掉了这个信息，所以 CuTe 直接拒绝。

> 这是初学最容易撞的第一个编译错误，错误信息还很难读（一堆 `tensor_impl.hpp` 的模板报错）。记住：**看到 `cannot be used to initialize an entity of type` 指向 `tensor_impl.hpp`，八成是忘了 `make_*_ptr`。**

### 常用查询

```cpp
size(T)      // 32     元素总数
rank(T)      // 2      mode 个数
shape(T)     // (_4,_8)
stride(T)    // (_8,_1)
T.layout()   // 取出 Layout 本身，可用 Section 01 的全部运算
T.data()     // 取出指针
print_tensor(T)   // 打印实际数值（不只是 layout）
```

### Tensor 是 view，不是容器

这一点决定了整章的用法。**Tensor 不拥有数据，只是一个视图。**

```cpp
float v[16] = {0,1,2,...};
auto T = make_tensor(make_gmem_ptr(v), make_layout(Int<16>{}, Int<1>{}));
auto p = local_partition(T, make_layout(Int<4>{}, Int<1>{}), 3);
p(0) = 999.0f;
// v[3] 现在是 999  —— 直接写穿到底层数组
```

实测确认：改 `p(0)` 之后 `v[3]` 从 `3` 变成 `999`。

**所有切分操作（切片、`local_tile`、`local_partition`）返回的都是 view，零拷贝。** 这就是为什么在 kernel 里可以放心地层层切分——没有任何内存搬运发生，全是编译期的下标变换。

---

## 2. 切片：`_` 通配符

```cpp
auto row2 = T(2, _);    // 第 2 行
auto col3 = T(_, 3);    // 第 3 列
```

`_`（即 `cute::_`）表示"这一维全要"。实测结果：

```
T      = (_4,_8):(_8,_1)      基址 0x...390
T(2,_) = (_8):(_1)            基址 0x...3d0     row2(0)=16, row2(7)=23
T(_,3) = (_4):(_8)            基址 0x...39c     col3(0)=3,  col3(3)=27
```

两点值得注意：

- **rank 降了**（2 → 1）：固定的那一维消失了。
- **基址变了**：`T(2,_)` 的指针从 `...390` 挪到 `...3d0`，差 `0x40 = 64` 字节 = 16 个 float = `T(2,0)` 的偏移。

这正好解答了 Section 01 练习 2 遗留的问题：**Layout 表达不了起点偏移，但 Tensor 可以 —— 偏移记在指针上。** 切片时 Layout 只保留形状，起点变化全部落到指针里。

---

## 3. `local_tile`：把大 Tensor 切成块

### 要解决什么

GPU 上一个 CTA（thread block）通常只负责整个矩阵的一小块。`local_tile` 就是"给我第 (i,j) 块"。

```cpp
auto blk = local_tile(BT, tile_shape, block_coord);
```

- `BT` — 大 Tensor
- `tile_shape` — 每块多大
- `block_coord` — 要第几块

### 实测

8×8 的 `BT`（值 0-63，row-major），切成 4×4 的块：

```
local_tile(BT, (4,4), (0,0)).layout = (_4,_4):(_8,_1)    首元素 = 0
local_tile(BT, (4,4), (0,1)).layout = (_4,_4):(_8,_1)    首元素 = 4
local_tile(BT, (4,4), (1,1)).layout = (_4,_4):(_8,_1)    首元素 = 36
```

**三个块的 layout 完全一样，只有首元素不同。** 再次印证 §2 的结论：**块的位置记在指针上，不在 Layout 里。** 这是好事——同一个 layout 意味着后续对每块的处理代码完全一致。

块 (1,1) 的内容（对应原矩阵 row 4-7 × col 4-7）：

```
36 37 38 39
44 45 46 47
52 53 54 55
60 61 62 63
```

### 背后就是 Section 01 的 composition

`local_tile` 建立在 `zipped_divide` 上，而后者是 composition 的封装：

```
zipped_divide(BT, (4,4)) = ((_4,_4),(_2,_2)):((_8,_1),(_32,_4))
                            ~~~~~~~  ~~~~~~~
                            块内坐标  块号
```

**这是个嵌套 layout**（Section 01 §3.4/§3.5 讲的那种），语义非常清楚：

- 第 0 个 mode `(4,4):(8,1)` —— 块内怎么走
- 第 1 个 mode `(2,2):(32,4)` —— 块与块之间怎么走（下一块 row 方向跳 32 = 4 行 × 行距 8；col 方向跳 4）

`local_tile(BT, tile, coord)` 就是"取出这个嵌套 layout，把块号固定成 `coord`，剩下块内部分"。理解了这一点，`local_tile` 就不再是黑盒。

### divide 家族的区别

同样的切分，三种打包方式：

```
zipped_divide(BT,(4,4)) = ((_4,_4),(_2,_2)):((_8,_1),(_32,_4))    块内和块号各自成组
tiled_divide (BT,(4,4)) = ((_4,_4),_2,_2)  :((_8,_1),_32,_4)      块号摊开
flat_divide  (BT,(4,4)) = (_4,_4,_2,_2)    :(_8,_1,_32,_4)        全摊平
```

日常用 `local_tile` 就够，知道底下是 `zipped_divide` 即可。

### tiler 可以只切一部分维度

```cpp
local_tile(BT, make_shape(Int<2>{}, Int<8>{}), make_coord(1, 0))
// -> (_2,_8):(_8,_1)   首元素 16
```

切成 "2 行 × 整行 8 列"，取第 1 块 = row 2-3。**用来按行条带切分很方便。**

---

## 4. `local_partition`：把块分给线程

### 要解决什么

一个 CTA 拿到一块数据后，还要分给块内的若干线程。`local_partition` 回答"我这个线程负责哪些元素"。

```cpp
auto mine = local_partition(T, thr_layout, thread_idx);
```

关键参数是 **`thr_layout`（线程布局）**：它描述线程如何排列，**直接决定了数据的分配模式**。

### 实测：thr_layout 决定一切

16 个元素分给 4 个线程，`thr_layout = 4:1`：

```
tid0:  0  4  8 12
tid1:  1  5  9 13
tid2:  2  6 10 14
tid3:  3  7 11 15
```

**每个线程拿到的是跨步的 4 个，不是连续的 4 个。** 这个默认行为很重要——同一轮里 4 个线程访问下标 0,1,2,3，**跨线程连续 = 合并访存**，正是 GPU 想要的。

这也直接印证了 Section 01 §4 的 complement：

```
local_partition(V, 4:1, 0).layout = ((_4)):((_4))
complement(4:1, 16)               = _4:_4
```

**partition 得到的 layout 正是 thr_layout 的 complement。** 所以 §4 学的那个"补集"不是理论，它就是 partition 的实现。

### 陷阱：thr_layout 写错会静默出错

把 `thr_layout` 换成 `4:4`：

```
tid0:  0  4  8 12
tid1:  0  4  8 12     <- 和 tid0 一模一样!
tid2:  0  4  8 12
tid3:  0  4  8 12
```

**四个线程拿到完全相同的元素**——12 个元素从没被碰过，4 个被重复写 4 次。而且**不报错、不警告**，跑起来只是结果不对。

原因：`thr_layout = 4:4` 意味着"线程 t 的起点是 `4t`"，但 4 个线程的起点 0,4,8,12 恰好和 complement 给出的 value 步长撞在一起，覆盖范围退化了。

**记住：`thr_layout` 的 stride 通常应该是 1**（线程连续编号）。这是 partition 最常见的错误来源，且极难调试——所以本章练习专门有一题练这个。

### 2D partition

线程也可以是二维排列：

```cpp
auto thr2d = make_layout(make_shape(Int<2>{}, Int<4>{}),
                         make_stride(Int<4>{}, Int<1>{}));   // 2x4 = 8 个线程
local_partition(BT, thr2d, tid)
// tid=0 -> (_4,_2):(_16,_4)  首元素 0
// tid=1 -> (_4,_2):(_16,_4)  首元素 1
// tid=2 -> (_4,_2):(_16,_4)  首元素 2
```

同样是"layout 相同、首元素不同"的模式。相邻 tid 的首元素差 1，说明沿最后一维（col）连续排布——依然是合并访存友好的。

---

## 5. 标准范式：tile → partition

真实 kernel 里这两个算子几乎总是**连用**，构成 CuTe 的核心数据流：

```cpp
// 1) CTA 层：从全局数据取出本 block 负责的块
auto blk = local_tile(gA, cta_tile_shape, make_coord(blockIdx.x, blockIdx.y));

// 2) Thread 层：从块中取出本线程负责的元素
auto mine = local_partition(blk, thr_layout, threadIdx.x);

// 3) 干活 —— 直接对 mine 读写，下标都算好了
for (int i = 0; i < size(mine); ++i) mine(i) = ...;
```

实测（8×8 数据，取块 (1,0)，再分给 2×2 共 4 个线程）：

```
1) local_tile(BT,(4,4),(1,0)) -> (_4,_4):(_8,_1)   首元素 32
2) local_partition(blk,(2,2):(2,1),0) -> (_2,_2):(_16,_2)
   tid0 拿到: 32 48 34 50
```

**注意这个两级结构对应 GPU 的两级并行**（grid 里的 block、block 里的 thread），而两级都只是 view 变换，没有任何数据搬运。

后面章节会在这个骨架上继续叠：Section 03/04 在中间插入 gmem→smem 的 Copy，Section 05/06 把最内层换成 MMA。**骨架不变。**

---

## 6. 寄存器 Tensor：`make_fragment_like`

前面的 Tensor 都指向已有内存。计算时还需要**新开一块**放中间结果（累加器等），且要放在寄存器里：

```cpp
auto frag = make_fragment_like(mine);   // 形状照抄 mine，但重新分配、紧密排布
```

对比两个类似函数（`mine` 的 layout 是 `(2,2):(16,2)`）：

```
make_tensor_like(mine)   = (_2,_2):(_2,_1)
make_fragment_like(mine) = (_2,(_2)):(_1,(_2))
```

两者都**丢掉了原来的 stride `(16,2)`**（那是指向大数组的跨步），换成紧密排布——因为新数据是独立的一小块，没必要跨步。`make_fragment_like` 保证第一维 stride 为 1，最利于向量化和寄存器分配。

**用法：** 需要一个和某个 view 同形状的临时缓冲时用它。Section 05 的 MMA 累加器就是这么来的。

---

## 7. 代码怎么读

| 文件 | 对应章节 | 内容 |
|---|---|---|
| `cute_tensor_v0.cu` | §1 §2 | Tensor 创建、查询、切片、view 语义验证 |
| `cute_tensor_v1.cu` | §1 | GPU 上的 gmem / smem Tensor，多线程协作 |
| `cute_tensor_v2.cu` | §3 §4 §5 | `local_tile`、`local_partition`、thr_layout 陷阱、两级范式 |
| `cute_tensor_capstone.cu` | 综合 | GEMV，对比 baseline / naive / smem / partition 四版 |

```bash
make run    # 全部跑一遍
make ex     # 做练习
```

---

## 练习

题目按顺序做，答案填在 `exercises/ex.cu` 的 TODO 处，框架自动判分。参考解答见 `exercises/solutions.md`。

### 练习 1 — Tensor 基础 ★☆☆
`float v[24]`，值 0-23。创建一个 `(4,6)` row-major 的 Tensor `T`。

1. `T(2,3)` 的值是多少？
2. `size(T)`、`rank(T)` 各是多少？
3. `T(1,_)` 的 shape 和 stride 是什么？它的第一个元素是多少？
4. `T(_,2)` 呢？

### 练习 2 — view 语义 ★☆☆
接上题。通过 `T(3,5) = -1.0f` 修改后，`v[23]` 变成多少？为什么？

再想一步：如果对 `T` 做切片得到 `auto r = T(3,_)`，然后 `r(5) = -2.0f`，`v[23]` 会变吗？先答再验证。

### 练习 3 — local_tile ★★☆
`(8,12)` row-major 的 Tensor `BT`（值 0-95）。用 `local_tile` 切成 `(4,4)` 的块。

1. 一共有几块？（提示：算 shape 相除）
2. `local_tile(BT, (4,4), (1,2))` 的首元素是多少？先手算再验证。
3. 这个块的 layout 是什么？和 `(0,0)` 块的 layout 相同吗？为什么？
4. 用 `zipped_divide(BT, (4,4))` 打印完整的嵌套 layout，说明第 1 个 mode 的 stride 各自代表什么。

### 练习 4 — thr_layout 决定分配模式 ★★☆
32 个元素的一维 Tensor，8 个线程。**先预测再验证**：分别用下面三个 `thr_layout`，写出 tid=0 和 tid=1 各拿到哪些下标。

1. `thr = 8:1`
2. `thr = 8:4`
3. `thr = 4:1`（只有 4 个线程）

对每一个回答：这个分配是"跨线程合并"还是"每线程连续"？哪个有问题（重复/漏掉）？

### 练习 5 — 两级范式 ★★★
`(8,8)` 的 Tensor（值 0-63）。要求：

1. 用 `local_tile` 取出块 `(0,1)`（4×4）。首元素应该是多少？
2. 用 `local_partition` 把这个块分给 `2×2` 布局的 4 个线程。
3. 写出 tid=3 拿到的全部元素值。先手算再验证。

提示：先算 `local_tile` 的首元素，再想 partition 在这个基础上怎么走。

### 练习 6 — make_fragment_like ★★☆
`mine` 的 layout 是 `(2,2):(16,2)`。

1. `make_fragment_like(mine)` 的 layout 是什么？先预测。
2. 为什么它的 stride 和 `mine` 不同？
3. `size()` 和 `cosize()` 分别是多少？和 `mine` 比有什么区别，说明了什么？

### 练习 7 — 改 capstone ★★★
`cute_tensor_capstone.cu` 里 `gemv_cute_partition` 用的 `thr_layout` 是一维的。

1. 跑一遍记录各版本耗时。
2. 把 GEMV 的 shape 从 `(4096,512)` 改成 `(512,4096)`（瘦长变宽扁），重新跑。哪个版本受影响最大？为什么？
3. `gemv_cute_smem` 把向量 `x` 放进 shared memory。当 N 从 512 涨到 4096 时，smem 够用吗？算一下需要多少字节，和 H100/H200 每 block 的 smem 上限（228KB）比较。

---

## 小结

| 概念 | 一句话 | 关键点 |
|---|---|---|
| `make_tensor(ptr, layout)` | Tensor = 指针 + Layout | 指针必须 `make_gmem_ptr` 等包装 |
| view 语义 | 所有切分都零拷贝 | 写 view 会穿透到底层数组 |
| `T(i,_)` | 切片 | rank 降低，起点记在指针上 |
| `local_tile` | 取第 (i,j) 块 | 各块 layout 相同，只有指针不同 |
| `local_partition` | 分给线程 | thr_layout 决定分配模式，stride 通常取 1 |
| tile → partition | 标准两级范式 | 对应 grid/block 两级并行 |
| `make_fragment_like` | 新开紧密排布的缓冲 | 用于累加器等寄存器数据 |

**下一章：Section 03 — Copy Atom。** 到目前为止我们只是在"描述"数据视图，还没有真正搬运数据。Copy Atom 是 CuTe 里数据搬运的最小单元，它会根据指针类型（gmem/smem）自动选择合适的硬件指令。
