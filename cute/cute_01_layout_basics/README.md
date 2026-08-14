# CuTe 教程 01: Layout 基础

## 📖 教程概述

本教程介绍 CuTe 的核心概念：**Layout**。Layout 是 CuTe 中最基础也最重要的抽象，它描述了如何将多维逻辑坐标映射到一维线性内存地址。

## 🔰 前置知识：CuTe 核心概念速览

### 什么是 CuTe？

**CuTe (CUDA Templates)** 是 CUTLASS 3.x 的核心抽象层，提供编译期的张量和布局抽象。

**设计理念：**
- 🎯 **类型系统描述数据**：用模板描述形状和访问模式
- ⚡ **编译期计算**：让编译器在编译时完成计算，而非运行时
- 🚀 **零开销抽象**：高层抽象不牺牲性能

### Int&lt;N&gt; vs int：为什么用编译期常量？

```cpp
int n = 8;              // 运行时变量 - 编译器不知道具体值
Int<8> n_compile{};     // 编译期常量 - 编译器可以激进优化
```

**Int&lt;N&gt; 的优势：**
- ✅ 编译器可以展开循环（loop unrolling）
- ✅ 优化寄存器分配
- ✅ 消除分支判断
- ✅ 生成立即数指令（PTX 汇编）
- ✅ 类型系统携带信息（Int&lt;8&gt; ≠ Int&lt;16&gt;）

### Shape 和 Stride：数据的形状与步长

**Shape（形状）** - 描述每个维度有多少元素
```cpp
Shape = (8)        // 一维: 8 个元素
Shape = (4, 8)     // 二维: 4 行 8 列
Shape = (2, 4, 8)  // 三维: 2×4×8
```
💡 类比：Shape 就像房子的**户型图** - 告诉你有几间房，每间多大

**Stride（步长）** - 描述维度步进时地址跳多远
```cpp
Stride = (1)       // 一维: 连续存储
Stride = (8, 1)    // 二维 row-major: 换行跳8，换列跳1
Stride = (1, 4)    // 二维 column-major: 换行跳1，换列跳4
```
💡 类比：Stride 就像**门牌号规则** - 从一个房间到另一个房间，门牌号加多少

**Layout = Shape + Stride**
```cpp
Layout = (Shape, Stride)

// 定义映射函数: 坐标 (i, j, k) → 线性偏移
offset = i * stride[0] + j * stride[1] + k * stride[2] + ...
```

### 可视化示例

**4×8 矩阵的两种布局：**

```
Row-major: Shape=(4,8), Stride=(8,1)
   内存: [0,1,2,3,4,5,6,7,8,9,10,11,...]
   
   逻辑视图:
      0  1  2  3  4  5  6  7
   0: 0  1  2  3  4  5  6  7
   1: 8  9 10 11 12 13 14 15
   2:16 17 18 19 20 21 22 23
   3:24 25 26 27 28 29 30 31
   
   坐标(1,3) → offset = 1*8 + 3*1 = 11 ✓

Column-major: Shape=(4,8), Stride=(1,4)
   同样的内存，不同的逻辑视图:
      0  1  2  3  4  5  6  7
   0: 0  4  8 12 16 20 24 28
   1: 1  5  9 13 17 21 25 29
   2: 2  6 10 14 18 22 26 30
   3: 3  7 11 15 19 23 27 31
   
   坐标(1,3) → offset = 1*1 + 3*4 = 13 ✓
```

**关键洞察：** 改变 Layout 就能改变数据访问模式，无需移动数据！

## 🎯 学习目标

- 理解 Layout 的组成：Shape 和 Stride
- 掌握 Layout 的代数运算（composition, complement, coalesce）
- 学会用 Layout 描述不同的数据访问模式
- 在 GPU 上使用 CuTe Layout 实现高效算法

## 📚 教程结构

### v0: Layout 基础 (`cute_layout_v0.cu`)
**核心概念：Shape 和 Stride**

- 一维 Layout：连续存储
- 二维 Layout：row-major vs column-major
- Layout 的 `size()` 和 `cosize()`
- 自定义步长的 Layout

**关键知识点：**
```cpp
// Layout = (Shape, Stride)
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),    // 4行8列
                         make_stride(Int<8>{}, Int<1>{}));   // 行步长8，列步长1

int offset = layout(i, j);  // 计算线性偏移: i * 8 + j * 1
```

### v1: Layout 代数运算 (`cute_layout_v1.cu`)
**核心概念：Layout 的组合和变换**

- `composition(L1, L2)`: 复合映射
- `complement(L, size)`: 计算补集维度
- `coalesce(L)`: 合并连续维度
- 嵌套 Layout：层次化索引

**应用场景：**
- 线程到数据的映射
- Warp 和 Block 的层次结构
- Tile 的组织

### v2: 操作实际数据 (`cute_layout_v2.cu`)
**核心概念：Layout 改变访问模式**

- 在 CPU 上用 Layout 操作数据
- 矩阵转置：逻辑转置 vs 物理转置
- Strided 访问：每隔一列采样
- Layout 的零拷贝能力

**关键洞察：**
```cpp
// 转置只需改变 Layout，无需移动数据！
auto layout_transposed = make_layout(make_shape(Int<N>{}, Int<M>{}),
                                    make_stride(Int<1>{}, Int<N>{}));
```

### capstone: GPU 矩阵转置 (`cute_layout_capstone.cu`)
**综合应用：GPU 上的 Layout + Tensor**

实现三个版本的矩阵转置：
1. **Naive**: 全局内存直接转置
2. **Tiled**: 使用 shared memory + CuTe Tensor
3. **Padded**: 添加 padding 避免 bank conflict

**性能对比 (2048×2048, H200):**
```
Naive:  1117.91 GB/s (基准)
Tiled:  1198.32 GB/s (1.1x)
Padded: 2467.12 GB/s (2.2x)  ← padding 消除了 bank conflict
```

**核心代码：**
```cpp
// 在 GPU shared memory 上创建 CuTe Tensor
__shared__ Element smem[TILE_M * (TILE_N + PADDING)];

auto smem_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                              make_stride(Int<TILE_N + PADDING>{}, Int<1>{}));

auto tile = make_tensor(make_smem_ptr(smem), smem_layout);

// 使用 Tensor 索引访问
tile(local_i, local_j) = input[...];
```

## 🚀 编译和运行

```bash
# 编译所有版本
make all

# 运行所有版本
make run

# 运行单个版本
./cute_layout_v0
./cute_layout_v1
./cute_layout_v2
./cute_layout_capstone

# 清理
make clean
```

## 💡 关键收获

1. **Layout 是映射函数**：从 N 维逻辑坐标到 1 维线性偏移
2. **Layout = (Shape, Stride)**：Shape 定义维度大小，Stride 定义步长
3. **零拷贝变换**：很多操作可以通过改变 Layout 而非移动数据完成
4. **编译期优化**：CuTe 的 `Int<N>` 让编译器在编译期完成大量计算
5. **抽象与性能**：Layout 的抽象不影响性能，反而让优化更清晰

## 🔗 与 CUDA 概念的对应

| CUDA 概念 | CuTe Layout 表达 |
|-----------|------------------|
| row-major | `(_M, _N):(_N, _1)` |
| column-major | `(_M, _N):(_1, _M)` |
| strided array | 自定义 stride |
| bank conflict | padding stride |
| thread mapping | `complement()` |

## 📌 常见问题

**Q: 为什么要用 `Int<N>{}` 而不是普通的 `int`？**  
A: `Int<N>` 是编译期常量，让编译器可以在编译期展开循环、优化寄存器分配，生成更高效的代码。

**Q: `size()` 和 `cosize()` 的区别？**  
A: `size()` 是逻辑元素总数（Shape 的乘积），`cosize()` 是实际占用的地址空间大小（最大偏移+1）。对于有 padding 的 Layout，两者不同。

**Q: 什么时候用逻辑转置，什么时候用物理转置？**  
A: 如果转置后只用一次且可以 fuse 到下一个操作，用逻辑转置（零拷贝）。如果要多次访问，用物理转置（连续访问更快）。

## 📖 下一步

完成本教程后，你应该：
- ✅ 理解 Layout 的数学含义
- ✅ 会创建和操作 Layout
- ✅ 能在 GPU 上使用 Layout

**下一个教程：cute_02_tensor_basics**
- Tensor 的创建和分区
- `local_tile` 和 `local_partition`
- 不同内存空间的 Tensor（gmem/smem/rmem）
- Tensor 的切片和重组

## 📚 参考资料

- [CuTe 官方文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [CuTe 论文: CuTe: A Generic C++ Compiler for Mixed-Precision Tensor Computations](https://developer.nvidia.com/blog/cutlass-3-0-faster-kernels-for-small-problem-sizes/)
