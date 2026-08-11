#!/usr/bin/env python3
"""教学版 Transformer Block（二）：手写 Multi-Head Causal Self-Attention + Top-k MoE。

与 mha_ffn.py 平行、自包含、可独立运行，两文件之间零依赖：

    .venv/bin/python transformer_block/mha_moe.py

唯一区别：前馈子层从 Dense FFN 换成 Top-k Mixture of Experts —— 每个 token
由 router 选出 top_k 个专家，分别计算后按归一化权重加权合并。

核心 shape 与 mha_ffn.py 一致：
    x (输入)     : [B, T, C]
    q / k / v    : [B, H, T, D]，C = H * D
    attn scores  : [B, H, T, T]
    block 输出    : [B, T, C]
此外 MoE 多返回一个 aux_loss（Switch Transformer 风格负载均衡辅助损失，标量）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """模型超参数（运行时的 B、T 在 forward 时给定）。"""

    d_model: int = 64        # C：每个 token 的隐藏维度
    num_heads: int = 4       # H：注意力头数，要求 C % H == 0
    d_ff: int = 256          # 每个专家 FFN 的中间层维度
    num_layers: int = 2      # 堆叠几个 block
    num_experts: int = 4     # E：MoE 专家数
    top_k: int = 2           # 每个 token 路由到几个专家，要求 1 <= k <= E


class MultiHeadCausalSelfAttention(nn.Module):
    """手写多头因果自注意力（与 mha_ffn.py 中实现一致，独立重复一份）。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.H = cfg.num_heads
        self.D = cfg.d_model // cfg.num_heads
        self.scale = self.D ** -0.5

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                           # [B, T, 3C]
        qkv = qkv.reshape(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)                      # 各 [B, H, T, D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)                # [B, H, T, T]

        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, C)
        out = self.o_proj(ctx)                           # [B, T, C]

        if trace:
            print("    [Attention]")
            print(f"      q / k / v            {[B, self.H, T, self.D]}")
            print(f"      attn scores / probs  {[B, self.H, T, T]}")
            print(f"      output               {list(out.shape)}")

        return out


class Expert(nn.Module):
    """单个专家：结构等同普通 Dense FFN，但参数与其它专家互相独立。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class TopKMoE(nn.Module):
    """手写 token-level Top-k MoE（不调用任何第三方 MoE 库）。

    每个 token 的处理：
      1. router 给 E 个专家各打一个分 -> softmax 成概率；
      2. 选概率最大的 k 个专家；
      3. 把这 k 个专家的输出按"归一化后的 router 概率"加权求和。

    这里用 Python 专家循环，逻辑透明；不含容量上限 / token drop / all-to-all，
    因此适合学习与正确性验证，不是高性能实现。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert 1 <= cfg.top_k <= cfg.num_experts, "top_k 必须在 [1, num_experts] 内"
        self.E = cfg.num_experts
        self.k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat = x.reshape(B * T, C)                       # [N, C]，N = B*T

        logits = self.router(flat)                       # [N, E]
        # router 概率用 fp32 算更稳，梯度照样回传到 router。
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_p, topk_i = torch.topk(probs, k=self.k, dim=-1)   # 均为 [N, k]
        # 选中的 k 个概率在本 token 内重新归一化，作为加权权重。
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # 把每个专家的加权输出累加回对应 token；一个 token 被路由到 k 个专家，
        # 就会被累加 k 次，正好实现"加权求和"。
        combined = torch.zeros_like(flat)
        for eid, expert in enumerate(self.experts):
            tok, slot = torch.where(topk_i == eid)       # 该专家负责的 (token, 槽位)
            if tok.numel() == 0:
                continue
            eo = expert(flat.index_select(0, tok))       # [n, C]
            g = topk_p[tok, slot].to(eo.dtype)           # [n]
            combined.index_add_(0, tok, eo * g.unsqueeze(-1))

        out = combined.reshape(B, T, C)

        # Switch Transformer 风格负载均衡辅助损失：理想时各专家负载都接近 1/E。
        # mean_prob：平均每个 token 给该专家的概率；load：实际分到的 token 比例。
        counts = torch.bincount(topk_i.reshape(-1), minlength=self.E)
        load = counts.float() / float(flat.shape[0] * self.k)
        mean_prob = probs.mean(dim=0)
        aux_loss = self.E * torch.sum(mean_prob * load)

        if trace:
            print("    [Top-k MoE]")
            print(f"      flat tokens          {list(flat.shape)}")
            print(f"      router logits        {list(logits.shape)}")
            print(f"      top-k expert ids     {list(topk_i.shape)}")
            print(f"      tokens per expert    {counts.tolist()}")
            print(f"      normalized load      {[round(v, 3) for v in load.tolist()]}")

        return out, aux_loss


class Block(nn.Module):
    """Pre-LN Decoder Block：注意力子层 + MoE 子层，各一次残差。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.moe = TopKMoE(cfg)

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), trace=trace)
        moe_out, aux = self.moe(self.norm2(x), trace=trace)
        x = x + moe_out
        return x, aux


class TransformerStack(nn.Module):
    """堆叠 num_layers 个 Block；把各层 aux_loss 求和返回。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self, x: torch.Tensor, trace: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_aux = x.new_zeros(())
        for i, blk in enumerate(self.blocks):
            x, aux = blk(x, trace=trace and i == 0)
            total_aux = total_aux + aux
        return self.norm(x), total_aux


def main() -> None:
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TransformerStack(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model, device=device)

    print("=" * 64)
    print("MHA + Top-k MoE  |  device =", device)
    print(
        f"config: C={cfg.d_model}, H={cfg.num_heads}, "
        f"E={cfg.num_experts}, top_k={cfg.top_k}, layers={cfg.num_layers}"
    )
    print(f"parameters: {n_params:,}")
    print(f"input  x : {tuple(x.shape)}   (B, T, C)")
    print("-" * 64)

    y, aux = model(x, trace=True)

    print("-" * 64)
    print(f"output y : {tuple(y.shape)}   (B, T, C)")
    print(f"aux_loss (load balance) = {aux.item():.4f}")
    print(f"y[0, 0, :4] = {[round(v, 4) for v in y[0, 0, :4].tolist()]}")
    print()


if __name__ == "__main__":
    main()
