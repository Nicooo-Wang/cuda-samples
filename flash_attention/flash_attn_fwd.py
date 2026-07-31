"""
Flash Attention 前向传播（仅 causal mask）—— 用于学习 Triton 的精简示例。

本文件是从一份完整的 Flash Attention 实现（含反向）里【只剥离正向传播部分】得到的：
  - 保留了 _attn_fwd_inner（online softmax 内循环）和 attn_fwd（前向 kernel）；
  - 用一个普通函数 flash_attention_fwd() 启动 kernel，不再用 torch.autograd.Function；
  - 正确性对照 PyTorch 的 scaled_dot_product_attention(is_causal=True)。
  - 删除了所有反向（attn_backward_*）、Delta、autograd 相关代码。

前向主要参考两篇原始论文的伪代码：
  https://arxiv.org/abs/2205.14135
  https://arxiv.org/abs/2307.08691

学习要点（这份代码想让你掌握的 Triton 技巧）：
  - 调用子 kernel（attn_fwd 里两次调用 _attn_fwd_inner）；
  - 用更快的 tl.exp2() 代替 tl.exp()（代价是反向求导要多乘一个 ln2，本示例无反向所以无影响）；
  - Flash Attention 特有的并行策略：序列维为主并行轴，batch×head 为次轴；
  - tl.static_assert() 编译期断言；
  - 多轴 launch grid 以及【轴顺序为什么重要】（同一段序列的 PID 落到同一个 SM）；
  - 用近似常数（rln2）代替每次现算；
  - 把“对角线块”和“对角线下方块”分两次调用 inner kernel 处理（causal mask 的分块技巧）。

不支持（为保持示例简洁而砍掉）：
  - 非 causal 的 mask / 无 mask；
  - fp32 以外的数据类型（不做 mixed precision / fp16 / fp8）；
  - dropout；
  - 反向传播 / autograd。
"""

import math

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# import os
# os.environ["TRITON_INTERPRET"] = "1"  # 想逐行调试 Triton 时取消注释（用 Python 解释器跑，慢但可断点）


@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N, offsets_KV_N,
    N: tl.constexpr, Dh: tl.constexpr,
):
    """
    箭头表示本 pid 的 for 循环方向；每条箭头是一个不同的 PID
                N of K & V
                ------------>
                ------------>
    N of Q      ------------>
                ------------>
                ------------>
    但如果考虑 causal mask，其实更像这样：
                N of K & V
                >
                --->
    N of Q      ------>
                --------->
                ------------>
    更进一步：对角线上的块在【第二次】调用本 kernel 时单独处理（带三角 mask）：
                N of K & V
                x
                   x
    N of Q            x
                         x
                            x
    而第一次调用处理对角线【下方】所有不需要三角 mask 的块：
                N of K & V

                -->
    N of Q      ----->
                -------->
                ----------->
    """
    if DIAGONAL:
        # 处理位于对角线上、存在“被 mask / 未被 mask”过渡的块
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        # 告诉编译器 lo 是 BLOCK_SIZE_QO 的倍数，便于优化
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else:
        # 对角线下方的块：causal mask 全为 1，不需要三角处理
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    # 沿 K & V 的 N 维逐块循环，边循环边把 O 累加器更新出来
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        # 告诉编译器 start_KV 是 BLOCK_SIZE_KV 的倍数
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        # 笼统的经验：凡是动态变量（相对静态 constexpr 而言）都可以试试 tl.multiple_of()

        # 计算 (Q @ K^T) / sqrt(Dh)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)  # (Dh, BLOCK_SIZE_KV)
            # 序列 mask：把块里超过 N 的不存在的 token 置成零向量
        S = tl.dot(Q, K_T) * softmax_scale  # (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
            # 被 mask 的 token 会在 S 的底边和右边形成零行/零列

        if DIAGONAL:  # 当前块包含对角线
            # causal mask：下三角（含对角线）为 True
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            # 把上三角（不含对角线）加 -inf
            S += tl.where(causal_mask, 0, -1.0e6)  # (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        # 注意：原先贴着 S 右边的被 mask token 现在大多变成了 -inf；
        #       贴着底边的还是以 0 为主、只是靠右部分混进了些 -inf（最底下那行除外，全是 0）

        # 求本块的最大值，并与之前所有块的最大值比较，得到更新后的最大值
        M_new = tl.maximum(M, tl.max(S, axis=1))  # (BLOCK_SIZE_QO,)
            # 底部被 mask 的行最大值会是 0（因为它们只有 0 和 -inf）
        # 用新最大值做 safe softmax 的平移
        S -= M_new[:, None]  # (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
            # 对被 mask 的不存在 token，减 0 等于没变

        # 对每个 safe 后的点积求 exp，作为 softmax 的分子
        P = tl.exp2(S)  # (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
            # 用底数 2 而不是 e：更快，且 softmax 对底数更换保持不变；
            # 代价是反向求导会稍微复杂（要乘 ln2），本示例没有反向所以无所谓。
            # 对底部被 mask 的不存在 token：2^0 = 1

        # 对注意力分数按行求和
        L_new = tl.sum(P, axis=1)  # (BLOCK_SIZE_QO,)
            # 对被 mask 的不存在 token：把一堆 1 和一些 -inf 相加；
            # 最底下那行全是 1，和最大，等于 BLOCK_SIZE_KV
        # 这个 alpha 是用来修正上一轮 L 的缩放因子
        alpha = tl.exp2(M - M_new)  # (BLOCK_SIZE_QO,)
            # 对被 mask 的不存在 token：2^(1-1)... 这里 M 初值很大负数，但逻辑上无影响
        # 用 alpha 修正上一轮 L，再加上本轮 L
        L = L * alpha + L_new  # (BLOCK_SIZE_QO,)
            # 每个被 mask 的不存在 token 的 L_i 会逐渐趋近于某个值

        # 计算 O = P @ V + O * alpha
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)  # (BLOCK_SIZE_KV, Dh)
        # 用可能更新的最大值修正之前的 O
        O = O * alpha[:, None]  # (BLOCK_SIZE_QO, Dh)
        # 把本轮 P@V 累加进 O
        O = tl.dot(P, V, acc=O)  # (BLOCK_SIZE_QO, Dh)
            # 注意：这里在【还没除以 softmax 分母 l_i】的情况下就做了 V 的投影，
            #  因为在这种分块累加的语境下，"先乘 V 再统一除"与"先除再乘 V"满足结合律。
            # acc=O 表示把结果累加进 O。
            # 底部被 mask 的不存在 token 会让 O 底部出现一堆错误值，但之后 store 时会被 mask 掉。

        # 把旧最大值更新为新最大值，准备进入下一轮 for
        M = M_new

        # 推进指针
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M  # 这三个留到后面用（前向只用 O；L、M 用来算 LSE）


# autotune：自动挑出最划算的元参数。这里故意只留单个配置，便于学习时编译快、行为确定；
# 想要更好性能，可以把下面每个列表里被注释掉的值放开，让它真正搜索。
@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]  # , 32, 64, 128]
        for BLOCK_SIZE_KV in [16]  # , 32, 64, 128]
        for num_stages in [3]  # , 5, 7]
        for num_warps in [4]  # , 8, 16]
    ],
    key=["Dh"],
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr,               # 每个 shape (B, H, N, Dh)
    O_ptr,                             # shape (B, H, N, Dh)，最终输出写这里
    LSE_ptr,                           # shape (B, H, N)，先存每行最大值，后存 logsumexp（反向才需要）
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B,  # batch 维度（相对其它维度，batch 大小可以更灵活）
    # 元参数（编译期决定）
    H: tl.constexpr, N: tl.constexpr,
    Dh: tl.constexpr,  # 应总是 2 的幂
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
    # 为了后面用 tl.exp2（比 tl.exp 快），需要把 softmax_scale 乘上 ln2 的倒数
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    """
    证明 e^x = 2^(x * rln2)：
      e^x = (2^(log_2(e)))^x        （因为 a = 2^log_2(a)）
           = 2^(x * log_2(e))        （幂的乘积法则）
           = 2^(x * 1/log_e(2))      （log_2(e) = 1/log_e(2)）
           = 2^(x * rln2)
    """

    # 编译期断言（相比普通 assert，static_assert 在编译期生效）
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
        # 原版 triton 教程有这条断言，具体原因不重要，留着不碍事

    # 本 pid 处理序列长度里的哪一个 Q 块
    block_index_QO = tl.program_id(0)
    # 本 pid 处理哪一个 head、哪一个 batch（每个 program 绑定一个 batch 的一个 head）
    index_BH = tl.program_id(1)
    index_B = index_BH // H
    index_H = index_BH % H

    # 通过 batch、head 索引，定位到 Q/K/V/O 里那个 (N, Dh) 的块
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    # N 维偏移按 pid 切分；Dh 维则整块放进 SRAM
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    # 为每个 tensor 构造各自的偏移
    Q_offsets = (offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh)
        # (BLOCK_SIZE_QO, Dh)
    # 加载 K 的同时做转置（而不是另写一个转置 kernel）
    K_T_offsets = (offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N)
        # (Dh, BLOCK_SIZE_KV)
    V_offsets = (offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh)
        # (BLOCK_SIZE_KV, Dh)

    # 加载本 pid 要用的 Q 块；它在整个 inner 循环里都会留在 SRAM
    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.)  # (BLOCK_SIZE_QO, Dh)
        # 序列 mask：把块里超过 N 的不存在 token 置成零向量

    # 预分配中间值与输出
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)  # 运行最大值，每行一个；大负数会被 tl.max 忽略
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)   # 运行累加和，每行一个；初值 1 是因为 e^0=1
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)               # 输出累加器，O 的一组行

    # 先算“密集块”（mask 全为 1 的块）—— causal 下就是对角线下方的块
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False,  # DIAGONAL：对角线块在下面那次调用里单独处理
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )

    # 再算对角线上的块（带三角 causal mask）
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True,  # DIAGONAL：对角线块做三角 mask
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )

    # 最后除以 softmax 分母。注意我们已经先把 O 乘了 V，所以这一步相对于朴素 softmax 是“乱序”的
    O = O / L[:, None]  # (BLOCK_SIZE_QO, Dh) / (BLOCK_SIZE_QO, 1)
        # 能乱序是因为：分块视角下这里不再是矩阵乘，而是逐个点积；
        # 而“点积后逐元素除”与“先逐元素除再做点积”在这种粒度下满足结合律。
        # 底部被 mask 的不存在 token：O 是一堆无意义的值、L 大致趋近某值，相除不会出问题，store 时会被 mask 掉。

    # 算 logsumexp（LSE），反向才需要。前向输出 O 本身不需要它，这里保留计算过程作为学习要点：
    #   把 max 和 sum 合在一起存，仍能正确还原 softmax，靠的是指数运算：
    #     softmax(x_i) = exp(x_i - m_i) / l_i = exp(x_i - m_i) / exp(log(l_i)) = exp(x_i - m_i - log(l_i))
    LSE = M + tl.math.log2(L)  # (BLOCK_SIZE_QO,)
        # 底部被 mask 的不存在 token：M 是一堆 0、L 大致趋近某值，LSE 底部会是一堆 log_2(...)，反正不用

    # 写回 DRAM
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)  # (BLOCK_SIZE_QO,)
        # mask 防止把底部没用的 LSE 值写出去
    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])  # (BLOCK_SIZE_QO, Dh)
        # mask 防止把底部不存在 token 对应的无用 O 值写出去


def flash_attention_fwd(q, k, v, scale):
    """启动前向 kernel 并返回输出 O（不做 autograd，所以不能 .backward()）。"""
    assert q.shape == k.shape == v.shape
    assert q.shape[-1] <= 128, f'本示例只支持 head_dim <= 128，得到的是 {q.shape[-1]}'
    B, H, N, Dh = q.shape
    assert q.device == k.device and q.device == v.device
    assert q.dtype == k.dtype == v.dtype == torch.float32

    O = torch.empty_like(q)  # 输出（pre-head 拼接与混合之前）
    # kernel 会写 LSE，所以这里仍要分配（但前向输出 O 用不到它，留着只是为了 kernel 签名）
    LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    # 轴顺序很关键：它决定了哪些 program 最终共享同一块 SRAM。
    grid = lambda args: (
        triton.cdiv(N, args["BLOCK_SIZE_QO"]),  # 主并行轴：沿序列长度切
        B * H,                                  # 次并行轴：batch × head
    )
    # 序列维 axis 在前、BH 并行 axis 在后：因为前者我们希望落在同一个 SM 上彼此靠近。
    r"""
    假设 launch grid 为 (3, 2)，有 3 个 SM、每个能容纳 2 个 PID，则 PID 分布大致是：
        [0, 0] \ SM0
        [1, 0] /
        [2, 0] \ SM1
        [0, 1] /
        [1, 1] \ SM2
        [2, 1] /
    """

    attn_fwd[grid](
        q, k, v, O, LSE,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, N, Dh,
    )
    return O


######### 正确性测试（仅前向）#########
def test_flashattention_fwd(B, H, N, Dh, device=DEVICE, atol=5e-3):
    # 造数据
    q = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device)
    k = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device)
    v = torch.randn((B, H, N, Dh), dtype=torch.float32, device=device)
    sm_scale = 1.0 / math.sqrt(Dh)

    # 本实现
    tri_out = flash_attention_fwd(q, k, v, sm_scale)
    # 参考实现：PyTorch 内置的 causal SDPA
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 对比
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
    print(f"passed fwd  (B={B}, H={H}, N={N}, Dh={Dh})")


if __name__ == "__main__":
    # 几组 shape：前三个 N 是 BLOCK_SIZE 倍数（无块边界 masking），
    # 最后一个 N=69 故意不是 16 的倍数，用来验证序列 masking 正确。
    test_flashattention_fwd(1, 1, 128, 32)
    test_flashattention_fwd(1, 1, 128, 64)
    test_flashattention_fwd(1, 1, 128, 128)
    test_flashattention_fwd(32, 8, 69, 128)
    print("全部前向测试通过 ✅")
