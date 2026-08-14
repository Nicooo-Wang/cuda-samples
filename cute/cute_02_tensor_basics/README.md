# CuTe 教程 02: Tensor 基础

## 📖 教程概述

本教程介绍 CuTe 的核心数据结构：**Tensor**。在 cute_01 中我们学习了 Layout（映射函数），现在我们将 Layout 与指针结合，创建可以直接操作数据的 Tensor。

## 🎯 学习目标

- 理解 Tensor = 指针 + Layout
- 创建不同内存空间的 Tensor（gmem/smem/rmem）
- 使用 Tensor 索引访问数据
- 掌握 local_partition 实现线程到数据的映射
- 在 GPU 上使用 Tensor 实现实际算法

## 🔰 前置知识

**核心公式：**
```cpp
Tensor = 指针 + Layout

// 创建 Tensor
auto tensor = make_tensor(ptr, layout);

// 访问元素（自动计算偏移）
tensor(i, j) = value;  // 等价于 ptr[layout(i, j)] = value
```

**为什么需要 Tensor？**

❌ **传统方式（手动计算偏移）：**
```cpp
float* data = ...;
data[i * stride_i + j * stride_j] = value;  // 容易出错
```

✅ **CuTe Tensor：**
```cpp
auto tensor = make_tensor(data, layout);
tensor(i, j) = value;  // 简洁、安全、零开销
```

## 📚 教程结构

### v0: CPU 上的 Tensor (`cute_tensor_v0.cu`)
**核心概念：Tensor 的创建和基本操作**

- Tensor 的属性：size, shape, stride, data
- 索引访问：`tensor(i, j, k)`
- 零开销抽象验证
- 不同 Layout 的 view（共享数据）
- 基础切片：`tensor(i, _)` 和 `tensor(_, j)`

**关键代码：**
```cpp
// 创建 Tensor
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                         make_stride(Int<8>{}, Int<1>{}));
auto tensor = make_tensor(data, layout);

// 访问元素
tensor(i, j) = value;

// 查询属性
int total = size(tensor);      // 逻辑元素总数
auto s = shape(tensor);        // 形状
auto st = stride(tensor);      // 步长
float* p = data(tensor);       // 底层指针

// 切片
auto row2 = tensor(2, _);      // 第2行
auto col3 = tensor(_, 3);      // 第3列
```

### v1: GPU 上的 Tensor (`cute_tensor_v1.cu`)
**核心概念：gmem 和 smem Tensor**

- Global memory Tensor：普通 device 指针
- Shared memory Tensor：`make_smem_ptr()`
- 多线程协作访问 Tensor
- 数据流水线：gmem → smem → gmem

**关键代码：**
```cpp
// Shared memory Tensor
__shared__ float smem[M * N];

auto layout = make_layout(...);
auto tensor = make_tensor(make_smem_ptr(smem), layout);

// 多线程协作访问
tensor(row, col) = value;  // 每个线程访问不同位置
```

**Kernel 示例：**
1. 基础 gmem Tensor
2. Smem Tensor 创建和访问
3. 多线程协作初始化
4. 完整数据流水线（含计算）

### v2: local_partition (`cute_tensor_v2.cu`)
**核心概念：线程到数据的映射**

**什么是 Partition？**

Partition 回答："每个线程应该处理哪些数据？"

```cpp
// 原始 Tensor: 64 个元素
auto tensor = make_tensor(..., make_layout(Int<64>{}));

// 线程布局: 8 个线程
auto thr_layout = make_layout(Int<8>{});

// Partition: 每个线程得到自己的数据视图
auto my_data = local_partition(tensor, thr_layout, thread_id);
// my_data: size = 8 (64 / 8)
```

**工作原理：**
- 将数据平均分配给所有线程
- 返回每个线程的 view（不拷贝数据）
- 不同线程的 view 访问不同数据（无重叠）

**对比传统方式：**

❌ **传统 CUDA：**
```cpp
int tid = threadIdx.x;
for (int i = tid; i < N; i += blockDim.x) {
    data[i] = ...;  // 手动步进
}
```

✅ **CuTe Partition：**
```cpp
auto my_data = local_partition(tensor, thr_layout, tid);
for (int i = 0; i < size(my_data); ++i) {
    my_data(i) = ...;  // 自动映射
}
```

**Kernel 示例：**
1. 1D partition 基础
2. 2D tensor partition
3. Partition 映射关系演示
4. 向量加法应用

### capstone: 矩阵-向量乘法 (GEMV) (`cute_tensor_capstone.cu`)
**综合应用：y = A * x**

实现四个版本：
1. **Baseline**: 传统 CUDA
2. **CuTe Naive**: 直接用 Tensor
3. **CuTe + Smem**: 缓存向量 x
4. **CuTe + Partition**: 使用 partition 分配行

**核心优化：**
- Shared memory 缓存频繁访问的数据（向量 x）
- Partition 简化线程到矩阵行的映射
- 零开销：CuTe 版本性能 ≥ 手写 CUDA

**关键代码：**
```cpp
// 使用 partition 加载 x 到 shared memory
auto my_x = local_partition(tx_smem, thr_layout, tid);
auto my_x_gmem = local_partition(tx_gmem, thr_layout, tid);
for (int j = 0; j < size(my_x); ++j) {
    my_x(j) = my_x_gmem(j);
}

// 使用 partition 分配矩阵行
auto my_rows = local_partition(tA, thr_layout, tid);
auto my_y = local_partition(ty, thr_layout, tid);

for (int row_idx = 0; row_idx < size(my_y); ++row_idx) {
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += my_rows(row_idx, j) * tx_smem(j);
    }
    my_y(row_idx) = sum;
}
```

**性能对比 (4096×512, H200):**
```
Baseline       1.000x
CuTe Naive     ~1.00x  (零开销！)
CuTe + Smem    ~1.2x   (缓存优化)
CuTe + Part    ~1.2x   (简化映射，性能相当)
```

## 🚀 编译和运行

```bash
# 编译所有版本
make all

# 运行所有版本
make run

# 运行单个版本
./cute_tensor_v0
./cute_tensor_v1
./cute_tensor_v2
./cute_tensor_capstone

# 清理
make clean
```

## 💡 关键收获

1. **Tensor = 指针 + Layout**
   - Layout 定义映射，指针指向数据
   - 组合起来就可以用坐标访问数据

2. **统一接口**
   - gmem/smem/rmem 使用相同的 API
   - 只有创建方式不同（`make_smem_ptr` 等）

3. **local_partition 自动映射**
   - 回答"每个线程处理哪些数据"
   - 返回 view，零拷贝
   - 代码更清晰，不易出错

4. **零开销抽象**
   - 编译后和手写 CUDA 一样快
   - 甚至可能更快（编译器优化空间更大）

5. **可组合**
   - 可以基于一个 Tensor 创建新的 view
   - Partition、切片都是 view 操作
   - 灵活且高效

## 📌 Tensor vs Layout

| | Layout | Tensor |
|---|--------|--------|
| 定义 | (Shape, Stride) | 指针 + Layout |
| 作用 | 坐标→偏移的映射 | 可访问的数据 |
| 操作 | layout(i, j) → offset | tensor(i, j) → 值 |
| 是否包含数据 | ❌ | ✅ |

## 🔧 常见操作速查

```cpp
// 创建
auto tensor = make_tensor(ptr, layout);
auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

// 查询
size(tensor)        // 元素总数
shape(tensor)       // 形状
stride(tensor)      // 步长
data(tensor)        // 底层指针

// 访问
tensor(i, j, k)     // 索引访问

// 切片
tensor(i, _)        // 第 i 行
tensor(_, j)        // 第 j 列

// Partition
local_partition(tensor, thr_layout, thread_id)
```

## 🤔 常见问题

**Q: make_smem_ptr 的作用是什么？**  
A: 将 `__shared__` 指针包装成 CuTe 能识别的类型，告诉 CuTe 这是 shared memory。这样 CuTe 可以做特定优化。

**Q: Partition 会拷贝数据吗？**  
A: 不会！Partition 返回的是 view，和原 Tensor 共享数据。只是改变了索引方式。

**Q: 为什么 CuTe 版本性能和手写一样？**  
A: 因为 Layout 的所有计算都在编译期完成。编译后的汇编代码和手写的几乎相同。

**Q: 什么时候用 Partition？**  
A: 当需要将数据分配给多个线程时。它自动计算每个线程负责哪些数据，避免手动步进。

## 📖 下一步

完成本教程后，你应该：
- ✅ 理解 Tensor 的本质
- ✅ 能在 GPU 上创建和使用 Tensor
- ✅ 掌握 local_partition 的用法
- ✅ 能用 Tensor 实现实际算法

**下一个教程：cute_03_copy_atom**
- Copy Atom 的概念
- 不同内存层次间的高效数据搬运
- Copy_Atom 的线程布局
- 使用 Copy 实现优化的 memcpy
