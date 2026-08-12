#!/usr/bin/env python3
"""教学版 Transformer Block（三）：Expert Parallel(EP) 版 Top-k MoE。

与 mha_moe.py（单 rank）平行、自包含、可独立运行，两文件之间零依赖：

    # 示例：world_size=2，每 rank 一张 GPU
    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 transformer_block/mha_moe_ep.py
    # 也可 --nproc_per_node=4（E_local=2）或 =8（E_local=1）

与单 rank 版的差别（也是本文件的教学重点）：
  * 专家按 rank 切分：rank r 持有全局专家 [r*E_local : (r+1)*E_local]，本 rank
    只实例化这 E_local 个 Expert；router(gate) 在各 rank 间复制。
  * token 经 NCCL all_to_all_single 跨 rank 分发到专家所在的 rank，算完再原路
    回收，最后在本地按 router 权重加权求和 —— 正好补上 mha_moe.py 里点名的"无 all-to-all"。
  * aux loss 用跨 rank 汇总后的全局统计量算（all_reduce）。

输入约定：纯 EP 下，前序（复制型）层给出相同激活，所以每个 rank 的输入 x 完全
一致；EP 前向在【每个 rank 上都重建出完整 [B,T,C] 输出】（不是分片），残差/下一层
直接可用。

核心 shape：
    x (输入，各 rank 一致) : [B, T, C]
    每 rank flat tokens     : [N, C]，N = B*T
    dispatch 发送/接收       : 每 rank 发出 N*k 条、收回 N*k 条；接收量随负载变化
    block 输出（各 rank 一致）: [B, T, C]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """模型超参数（运行时的 B、T 在 forward 时给定）。num_experts 需能被 world_size 整除。"""

    d_model: int = 64        # C：每个 token 的隐藏维度（输入/输出特征维）
    num_heads: int = 4       # H：注意力头数，要求 C % H == 0
    d_ff: int = 256          # 每个专家 FFN 的中间层维度
    num_layers: int = 2      # 堆叠几个 block
    num_experts: int = 8     # E：MoE 全局专家总数，要求 E % world_size == 0
    top_k: int = 2           # 每个 token 路由到几个专家，要求 1 <= k <= E


class MultiHeadCausalSelfAttention(nn.Module):
    """手写多头因果自注意力（与 mha_moe.py 中实现一致，独立重复一份）。
    纯 EP 下注意力是复制的：各 rank 权重、输入相同 → 输出相同。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # 断言 C 能被 H 整除，保证每个头维度 D 为整数
        assert cfg.d_model % cfg.num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.H = cfg.num_heads                  # 头数 H
        self.D = cfg.d_model // cfg.num_heads   # 每个头的维度 D = C / H
        self.scale = self.D ** -0.5             # 注意力缩放因子 1/sqrt(D)，稳定梯度

        # QKV 投影：一次性算出 q/k/v，输出通道 3*C；不带 bias
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # 输出投影：把多头拼接后的 C 维映射回 C 维；不带 bias
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        # 输入  x  : [B, T, C]（B=批次，T=序列长，C=隐藏维，各 rank 一致）
        # 输出 out : [B, T, C]
        B, T, C = x.shape                      # 拆出批次 B、序列长 T、隐藏维 C

        # 一次线性投影得到 q/k/v 拼接张量: [B, T, 3C]
        qkv = self.qkv_proj(x)
        # reshape 成 [B, T, 3, H, D] 再转置到 [3, B, H, T, D]
        # dim0=q/k/v，dim1=批次，dim2=头，dim3=序列，dim4=头内维
        qkv = qkv.reshape(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        # 沿 dim=0 拆出三个张量，各自 [B, H, T, D]
        q, k, v = qkv.unbind(dim=0)

        # 注意力分数 = q @ k^T * scale，每头独立，shape [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 因果掩码：下三角为 True（只看历史），上三角屏蔽未来；shape [T, T]
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        # 把上三角（未来位置）置为当前 dtype 的极小值，softmax 后趋近 0
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        # 沿 key 序列方向 softmax，得到注意力概率分布 [B, H, T, T]
        probs = F.softmax(scores, dim=-1)

        # 概率加权 value 得到上下文 [B, H, T, D]，转置并把多头拼回 [B, T, C]
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, C)
        # 输出投影回 [B, T, C]
        out = self.o_proj(ctx)

        if trace:
            # 仅 rank0 打印（注意力复制，各 rank 形状一致）
            print("    [Attention]  (replicated)")
            print(f"      q / k / v            {[B, self.H, T, self.D]}")
            print(f"      attn scores / probs  {[B, self.H, T, T]}")
            print(f"      output               {list(out.shape)}")

        return out                              # [B, T, C]


class Expert(nn.Module):
    """单个专家：结构等同普通 Dense FFN，但参数与其它专家互相独立（逐字重复自 mha_moe.py）。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # 升维投影：C -> d_ff（FFN 中间层），不带 bias
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        # 降维投影：d_ff -> C，不带 bias
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x : [..., C]（任意前导维，最后维是 C）
        # 输出   : [..., C]
        # up_proj 升维 -> GELU 激活 -> down_proj 降维，即标准两层 FFN
        return self.down_proj(F.gelu(self.up_proj(x)))


class TopKRouter(nn.Module):
    """门控路由（EP 版）。与单 rank 版同构，但 aux loss 需要跨 rank 统计量，
    所以这里只返回路由决策 + 完整概率，全局 aux 在 ExpertParallelMoE 里算。

    forward(flat):  flat [N, C]
        -> (topk_p [N, k], topk_i [N, k], probs [N, E] fp32)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # top_k 必须落在 [1, num_experts] 区间内
        assert 1 <= cfg.top_k <= cfg.num_experts, "top_k 必须在 [1, num_experts] 内"
        self.E = cfg.num_experts                # 全局专家数 E
        self.k = cfg.top_k                      # 每个 token 选 k 个专家
        # 门控线性层：C -> E，输出每个 token 对 E 个专家的 logits
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

    def forward(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 输入 flat : [N, C]（N = B*T 个 token）
        # 输出      : topk_p [N, k]（归一化权重）、topk_i [N, k]（专家号）、probs [N, E]（fp32）
        # 门控打分：每个 token 得到对 E 个专家的 logits [N, E]
        logits = self.gate(flat)
        # router 概率用 fp32 算更稳，梯度照样回传到 gate。
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        # 取概率最大的 k 个：topk_p 是概率值，topk_i 是对应全局专家号；均 [N, k]
        topk_p, topk_i = torch.topk(probs, k=self.k, dim=-1)
        # 选中的 k 个概率在本 token 内重新归一化，作为加权权重。
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # 权重转回输入 dtype；专家号、完整概率保持原样供 aux loss 使用
        return topk_p.to(flat.dtype), topk_i, probs


class ExpertParallelMoE(nn.Module):
    """Expert Parallel 版 Top-k MoE。

    本 rank 只持有 E_local = E // world_size 个专家（全局专家 [offset:offset+E_local]）。
    每个 rank 对自己的 N 个 token 独立做门控决策，再通过 all_to_all 把 token 送到
    对应专家所在的 rank，算完原路收回，本地加权求和。最终每个 rank 都得到完整 [B,T,C]。

    前向十步（注释里的 ① ~ ⑩ 与下方代码一一对应）：
      ① 本地路由（gate → softmax → topk → renorm）；
      ② 算出每条 (token,slot) 的目的 rank 与目的【本地】专家号；
      ③ 按目的 rank 排序，得到发送顺序 order 与每 rank 发送量 send_counts；
      ④ all_to_all 交换计数向量 → recv_counts；
      ⑤ 重排 token / 本地专家号到发送缓冲（按目的 rank 分段）；
      ⑥ all_to_all 发送 token 与本地专家号（变长分段）；
      ⑦ 本地专家计算：按 recv 的本地专家号分组，【写回原位】（保持顺序以便回传对齐）；
      ⑧ all_to_all 把输出原路送回（分段方向取反）；
      ⑨ 撤销排序 order，按 (token,slot) 乘上权重 topk_p 求和；
      ⑩ 用全局统计量算 aux loss。
    """

    def __init__(self, cfg: Config, rank: int, world_size: int) -> None:
        super().__init__()
        # 约束：全局专家数必须能被 world_size 整除，才能均匀切分到各 rank
        assert cfg.num_experts % world_size == 0, (
            f"num_experts({cfg.num_experts}) 必须能被 world_size({world_size}) 整除"
        )
        self.E = cfg.num_experts                # 全局专家总数 E
        self.k = cfg.top_k                      # 每 token 选 k 个专家
        self.rank = rank                        # 本进程的 rank
        self.world_size = world_size            # 进程数（GPU 数）
        self.E_local = cfg.num_experts // world_size   # 本 rank 持有的专家数
        self.offset = rank * self.E_local       # 本 rank 持有的全局专家起始号

        # 门控路由：各 rank 复制一份（权重相同），对本地 token 独立做决策
        self.router = TopKRouter(cfg)
        # 只实例化本 rank 负责的 E_local 个专家。
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(self.E_local)])

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入  x : [B, T, C]（各 rank 一致）
        # 输出 out : [B, T, C]（各 rank 一致）；aux_loss : 标量
        B, T, C = x.shape                       # 批次 B、序列长 T、隐藏维 C
        N = B * T                               # 本 rank 的 token 总数
        k, W, E_local = self.k, self.world_size, self.E_local  # 别名，简化书写

        # 把 [B, T, C] 展平成 [N, C]，路由/分发都在 token 维度上做
        flat = x.reshape(N, C)                  # [N, C]

        # ① 本地路由（每个 rank 对自己的 N 个 token 独立做门控决策）
        topk_p, topk_i, probs = self.router(flat)          # [N,k], [N,k], [N,E](fp32)

        # ② 每条 (token,slot) 的目的 rank 与目的本地专家号
        #    约定：专家按 rank 连续切分，rank r 持有全局专家 [r*E_local:(r+1)*E_local]
        # 全局专家号除以 E_local 即得它所在的 rank（整除商） [N, k]
        dest_rank = topk_i // E_local           # [N, k]
        # 全局专家号对 E_local 取余，即该专家在目的 rank 内的本地编号 [N, k]
        local_expert = topk_i % E_local         # [N, k]

        # ③ 按 dest_rank 排序 N*k 条 (token,slot)，得到发送顺序与每 rank 发送量
        # 把 [N, k] 的目的 rank 展平为 [N*k]，第 p 行对应第 p 条 (token,slot)
        dest_flat = dest_rank.reshape(-1)       # [N*k]
        # 稳定排序：返回使 dest_flat 升序的下标序列，stable 保证同 rank 内 (token,slot) 顺序不变
        order = torch.argsort(dest_flat, stable=True)      # [N*k]
        # 统计发给每个 rank 的条数：send_counts[r] = 本 rank 要发到 rank r 的条数
        send_counts = torch.bincount(dest_flat, minlength=W).to(torch.int64)  # [W]

        # ④ all_to_all 交换计数向量（1D，每 rank 一个元素）→ 本 rank 从各 rank 的接收量
        # 预留接收缓冲；recv_counts[r] 将等于 rank r 的 send_counts[本rank]，即本 rank 从 r 收到的条数
        recv_counts = torch.empty_like(send_counts)        # [W]
        # 1D 等分 all_to_all：每段 1 个元素，第 r 段发给 rank r
        dist.all_to_all_single(recv_counts, send_counts)
        splits = send_counts.tolist()          # 发给各 rank 的条数（Python list，长度 W）
        rsplits = recv_counts.tolist()         # 从各 rank 收到的条数（长度 W）
        n_recv = int(recv_counts.sum().item()) # 本 rank 总接收条数（随负载变化）

        # ⑤ 重排 token 与本地专家号到发送缓冲（按 dest_rank 分段）
        # 把每个 token 复制 k 份（连续重复），第 p 行 = flat[p // k]；即 (token,slot) 的 token 特征
        base_tok = flat.repeat_interleave(k, dim=0)        # [N*k, C]
        # 同样按 (token,slot) 顺序取本地专家号 [N*k]
        base_lex = local_expert.reshape(-1)    # [N*k]
        # 用 order 重排，使发送缓冲按目的 rank 分段（分段大小 = splits）
        send_tok = base_tok[order].contiguous()            # [N*k, C]
        send_lex = base_lex[order].contiguous()            # [N*k]

        # ⑥ all_to_all 发送 token 与本地专家号（变长分段）
        # 接收缓冲：n_recv 条 token 特征
        recv_tok = flat.new_zeros(n_recv, C)   # [n_recv, C]
        # 接收缓冲：n_recv 条本地专家号
        recv_lex = send_lex.new_zeros(n_recv)  # [n_recv]
        # token 跨 rank 交换：发送按 splits 分段，接收按 rsplits 分段
        dist.all_to_all_single(
            recv_tok, send_tok, output_split_sizes=rsplits, input_split_sizes=splits
        )
        # 本地专家号同样跨 rank 交换，使 recv_tok 与 recv_lex 行行对齐
        dist.all_to_all_single(
            recv_lex, send_lex, output_split_sizes=rsplits, input_split_sizes=splits
        )

        # ⑦ 本地专家计算：按 recv_lex 分组；【写回原位】保持 recv 顺序，回传才能对齐
        # 结果缓冲：与 recv_tok 同序（写回原位），shape [n_recv, C]
        recv_out = recv_tok.new_empty(n_recv, C)
        # 遍历本 rank 持有的每个本地专家
        for lid in range(E_local):
            # 找出 recv 中所有路由到本地专家 lid 的行下标 [m]
            sel = (recv_lex == lid).nonzero(as_tuple=False).squeeze(1)
            # 该专家有 token 命中才计算，否则跳过
            if sel.numel():
                # 用专家 lid 处理这批 token，结果写回原位置（保持顺序以便回传对齐）
                recv_out[sel] = self.experts[lid](recv_tok[sel])

        # ⑧ 原路送回：input_split=recv_counts（发回来源 rank），output_split=send_counts
        # 回传缓冲：按本 rank 发送顺序排列，长度 N*k（本 rank 最初发出了 N*k 条）
        out_back = flat.new_zeros(N * k, C)    # [N*k, C]
        # 反向 all_to_all：recv_out 按 rsplits 发回各来源 rank，out_back 按 splits 接收
        dist.all_to_all_single(
            out_back, recv_out, output_split_sizes=splits, input_split_sizes=rsplits
        )

        # ⑨ 撤销排序 order：out_back[j] 对应 send_tok[j]=base_tok[order[j]]，
        #    所以 out_ordered[order] = out_back 还原成按 (token,slot) 的顺序，再加权求和。
        # 按 (token,slot) 原始顺序的结果缓冲 [N*k, C]
        out_ordered = flat.new_zeros(N * k, C)
        # 用 order 做逆排列：把 out_back 放回它来源的 (token,slot) 位置
        out_ordered[order] = out_back
        # 重排成 [N, k, C]，乘上每条 (token,slot) 的权重 [N, k, 1]，再沿 k 维求和
        combined = (
            out_ordered.view(N, k, C) * topk_p.unsqueeze(-1)
        ).sum(dim=1)                           # [N, C]：每 token 的 k 个专家输出加权融合
        # 展平回 [B, T, C]
        out = combined.reshape(B, T, C)

        # ⑩ 全局 aux loss：路由统计跨 rank 汇总（all_reduce）
        # 统计本 rank 上每个全局专家被选中的次数 [E]（共 N*k 次选择），转 fp32
        local_counts = torch.bincount(topk_i.reshape(-1), minlength=self.E).to(probs.dtype)
        # 跨 rank 求和 → 全局每个专家的总命中次数 [E]
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        # 全局负载比例 = 命中次数 / 全局总选择次数 (W*N*k)；均匀时每个专家 ≈ 1/E
        g_load = local_counts / (W * N * k)    # [E]
        # 本 rank 上每个专家的平均路由概率（沿 token 维求均值）[E]
        local_meanp = probs.mean(dim=0)
        # 跨 rank 求和再除以 W → 全局平均路由概率 [E]
        dist.all_reduce(local_meanp, op=dist.ReduceOp.SUM)
        g_meanp = local_meanp / W              # [E]
        # 负载均衡损失：E * Σ(平均概率 ⊙ 负载比例)，均匀分布时取最小值；标量
        aux_loss = self.E * torch.sum(g_meanp * g_load)

        if trace and self.rank == 0:
            # 只在 rank0 打印 trace，避免多 rank 重复输出
            print("    [Expert-Parallel MoE]")
            print(
                f"      rank0 持有专家          [{self.offset}:{self.offset + E_local}] "
                f"(E={self.E}, E_local={E_local}, world_size={W})"
            )
            print(f"      flat tokens (每 rank)  [{N}, {C}]，每 token 选 k={k} 个专家")
            print(f"      send_counts -> 各 rank {splits}   (本 rank 共发出 {N * k} 条)")
            print(f"      recv_counts <- 各 rank {rsplits}   (本 rank 共接收 {n_recv} 条)")
            print(f"      aux_loss (全局)        {aux_loss.item():.4f}")

        # 返回 [B, T, C] 输出与标量 aux_loss
        return out, aux_loss


class Block(nn.Module):
    """Pre-LN Decoder Block：复制型注意力子层 + EP-MoE 子层，各一次残差。"""

    def __init__(self, cfg: Config, rank: int, world_size: int) -> None:
        super().__init__()
        self.rank = rank                        # 记录本 rank，用于 trace 门控
        # 第一个 LayerNorm：注意力子层前的归一化（Pre-LN）
        self.norm1 = nn.LayerNorm(cfg.d_model)
        # 复制型多头自注意力（各 rank 权重相同）
        self.attn = MultiHeadCausalSelfAttention(cfg)
        # 第二个 LayerNorm：MoE 子层前的归一化（Pre-LN）
        self.norm2 = nn.LayerNorm(cfg.d_model)
        # EP 版 MoE 子层：专家分布在本 rank
        self.moe = ExpertParallelMoE(cfg, rank, world_size)

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入  x : [B, T, C]；输出 x : [B, T, C]，aux : 标量
        # 注意力是复制的：各 rank 都算、结果相同，所以只在 rank0 打 trace。
        # 残差1：x + Attention(LayerNorm(x))
        x = x + self.attn(self.norm1(x), trace=trace and self.rank == 0)
        # MoE 子层：对归一化后的输入做 EP 前向，返回输出与 aux loss
        moe_out, aux = self.moe(self.norm2(x), trace=trace)
        # 残差2：x + MoE(LayerNorm(x))
        x = x + moe_out
        return x, aux                           # [B, T, C]，标量


class TransformerStack(nn.Module):
    """堆叠 num_layers 个 EP Block；各层 aux_loss（已全局化）求和返回。"""

    def __init__(self, cfg: Config, rank: int, world_size: int) -> None:
        super().__init__()
        # 堆叠 num_layers 个 Block，每个都传入 rank/world_size
        self.blocks = nn.ModuleList(
            [Block(cfg, rank, world_size) for _ in range(cfg.num_layers)]
        )
        # 最终归一化层，用在所有 block 之后
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入 x : [B, T, C]；输出 (norm(x) [B,T,C], total_aux 标量)
        # 标量 0 张量，用于累加各层 aux loss（与 x 同 device/dtype）
        total_aux = x.new_zeros(())
        # 逐层前向，trace 仅在第 0 层打开
        for i, blk in enumerate(self.blocks):
            x, aux = blk(x, trace=trace and i == 0)
            # 累加本层 aux loss（已是全局量）
            total_aux = total_aux + aux
        # 返回最终归一化输出与累计 aux loss
        return self.norm(x), total_aux


def _init_distributed() -> tuple[int, int, int]:
    """初始化 NCCL 进程组（由 torchrun 注入 RANK/WORLD_SIZE/LOCAL_RANK 环境变量）。"""
    # 初始化 NCCL 后端进程组（GPU 间集合通信）
    dist.init_process_group(backend="nccl")
    # 本进程的全局 rank
    rank = dist.get_rank()
    # 进程总数（= GPU 数）
    world_size = dist.get_world_size()
    # 本机内的 local rank，默认取环境变量，否则按 GPU 数取模
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    # 绑定当前进程到对应 GPU，后续 tensor 默认落在这张卡上
    torch.cuda.set_device(local_rank)
    # 返回全局 rank、进程数、本机 rank
    return rank, world_size, local_rank


def main() -> None:
    # 没有环境变量 RANK 说明不是 torchrun 启动 → 报错并给出用法
    if "RANK" not in os.environ:
        raise SystemExit(
            "本脚本是多进程版本，必须用 torchrun 启动（由它注入 RANK/WORLD_SIZE 等）。示例：\n"
            "  CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 "
            "transformer_block/mha_moe_ep.py"
        )
    # 初始化分布式，拿到 rank/world_size/local_rank
    rank, world_size, local_rank = _init_distributed()
    # 本进程对应的 GPU 设备
    device = torch.device(f"cuda:{local_rank}")
    # 是否为主进程（rank0），用于单点打印日志
    is_root = rank == 0

    def log(msg: str) -> None:
        # 只有 rank0 打印，避免多进程重复输出
        if is_root:
            print(msg)

    # 默认超参配置
    cfg = Config()
    # 再次校验专家数能被 world_size 整除（与 __init__ 内的断言呼应）
    assert cfg.num_experts % world_size == 0, (
        f"num_experts({cfg.num_experts}) 必须能被 world_size({world_size}) 整除"
    )

    # 固定随机种子，保证各 rank 初始化权重一致（复制型参数需相同）
    torch.manual_seed(0)
    # 构建模型并搬到本 GPU；各 rank 各持一份（专家部分互不相同）
    model = TransformerStack(cfg, rank, world_size).to(device)
    # 统计本 rank 持有的参数量
    n_params = sum(p.numel() for p in model.parameters())

    # 输入 shape：B=批次，T=序列长
    B, T = 2, 8
    # 固定输入种子，保证各 rank 输入完全相同（纯 EP 前提）
    torch.manual_seed(1234)                    # 每个 rank 同样的输入
    # 随机生成输入 [B, T, C]，落在本 GPU
    x = torch.randn(B, T, cfg.d_model, device=device)

    # 以下为 rank0 打印的运行信息头
    log("=" * 76)
    log(f"Expert-Parallel Top-k MoE  |  world_size={world_size}, "
        f"per-rank device={device}")
    log(
        f"config: C={cfg.d_model}, H={cfg.num_heads}, E={cfg.num_experts}, "
        f"E_local={cfg.num_experts // world_size}, top_k={cfg.top_k}, "
        f"layers={cfg.num_layers}"
    )
    log(f"parameters (每 rank) : {n_params:,}")
    log(f"input  x : {tuple(x.shape)}   (B, T, C)")
    log("-" * 76)

    # 前向：y 为 [B, T, C] 输出，aux 为标量负载均衡损失；trace 打开第 0 层
    y, aux = model(x, trace=True)

    # 以下为 rank0 打印的结果信息
    log("-" * 76)
    log(f"output y : {tuple(y.shape)}   (B, T, C)")
    log(f"aux_loss (load balance, 全局) = {aux.item():.4f}")
    log(f"y[0, 0, :4] = {[round(v, 4) for v in y[0, 0, :4].tolist()]}")
    log("")

    # 销毁进程组，释放 NCCL 资源
    dist.destroy_process_group()


if __name__ == "__main__":
    # 入口：直接运行时调用 main（实际需经 torchrun 启动）
    main()
