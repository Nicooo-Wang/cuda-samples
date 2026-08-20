# CuTe 教程系列

从零开始学习 CuTe (CUDA Templates)，CUTLASS 3.x 的核心抽象层。

## 📚 教程列表

| 教程 | 主题 | 难度 |
|------|------|------|
| [cute_01](cute_01_layout_basics/) | Layout 基础 | ⭐ |
| [cute_02](cute_02_tensor_basics/) | Tensor 基础 | ⭐ |
| [cute_03](cute_03_copy_atom/) | Copy Atom | ⭐⭐ |
| [cute_04](cute_04_tiled_copy/) | Swizzle / TMA / Multi-stage | ⭐⭐⭐ |
| [cute_05](cute_05_mma_atom/) | MMA Atom / WGMMA | ⭐⭐⭐ |
| [cute_06](cute_06_gemm_basic/) | 完整 GEMM (TMA+WGMMA+WS+Cluster) | ⭐⭐⭐⭐ |
| [cute_07](cute_07_persistent_scheduling/) | Persistent Block / Tile 调度 | ⭐⭐⭐⭐ |

## 🎯 学习路径

建议按顺序学习，每个教程都基于前面的知识。

**H200 五大特性覆盖**（本机 sm_90a 全部实测跑通）：

| 特性 | 章节 |
|---|---|
| TMA | cute_04 §2-5, cute_05 §4, cute_06 §4 |
| WGMMA | cute_04 §4.2 (编译期拒绝), cute_05 §3, cute_06 §4 |
| Warp Specialization | cute_06 §5 (手写; cute_04 §5 埋下动机) |
| Block Cluster | cute_06 capstone |
| Persistent Block | cute_07 |

## 🏁 系列终点

cute_06 capstone 的手写 TMA+WGMMA GEMM 实测 2048³ ≈ 430 TFLOP/s
（cuBLAS ≈ 878），cute_07 把 persistent + 调度讲完。你手里已经有
CUTLASS 的全部核心概念。
