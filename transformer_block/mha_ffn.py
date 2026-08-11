#!/usr/bin/env python3
"""教学版 Transformer Block（一）：手写 Multi-Head Causal Self-Attention + Dense FFN。

本文件自包含、可独立运行，与同目录其它脚本零依赖：

    .venv/bin/python transformer_block/mha_ffn.py

它只做一件事：初始化一个由若干 block 堆叠的 Transformer，喂一个 dummy 的
[B, T, C] 隐藏张量，跑一次前向，打印每一步的关键 shape。

注意：这里故意不接 token embedding 和 lm_head，把注意力集中到 block 本身。
真实 LM 里，block 前面会接 embedding(V->C)、后面会接 lm_head(C->V)；但对 block
内部计算而言，[B,T,C] 是 embedding 查出来的、还是随机生成的，没有任何区别。

核心 shape：
    x (输入)     : [B, T, C]
    q / k / v    : [B, H, T, D]，其中 C = H * D
    attn scores  : [B, H, T, T]
    block 输出    : [B, T, C]
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
    d_ff: int = 256          # FFN 中间层维度
    num_layers: int = 2      # 堆叠几个 block


class MultiHeadCausalSelfAttention(nn.Module):
    """手写多头因果自注意力（不调用 nn.MultiheadAttention）。

    数学过程：
        Q = X·Wq,  K = X·Wk,  V = X·Wv        每个 head 各一套 W
        A = softmax( Q·K^T / sqrt(D) + 因果掩码 )
        Y = Concat(head_1, ..., head_H)·Wo

    工程上把 Wq/Wk/Wv 合并成一个 qkv_proj，减少 Linear 调用次数。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.H = cfg.num_heads
        self.D = cfg.d_model // cfg.num_heads
        self.scale = self.D ** -0.5                      # 1/sqrt(D)

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                           # [B, T, 3C]
        # 拆成 3 份、再分头：[B,T,3C] -> [B,T,3,H,D] -> [3,B,H,T,D]
        qkv = qkv.reshape(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)                      # 各 [B, H, T, D]

        # 每个 head 内做 QK^T：[B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 因果掩码：第 i 个 query 只能看第 0..i 个 key，看不到未来 token。
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)                # [B, H, T, T]

        # 用概率对 V 加权求和：[B,H,T,T] @ [B,H,T,D] -> [B,H,T,D]
        ctx = torch.matmul(probs, v)

        # 多头拼回隐藏维：[B,H,T,D] -> [B,T,H,D] -> [B,T,C]
        ctx = ctx.transpose(1, 2).reshape(B, T, C)
        out = self.o_proj(ctx)                           # [B, T, C]

        if trace:
            print("    [Attention]")
            print(f"      q / k / v            {[B, self.H, T, self.D]}")
            print(f"      attn scores / probs  {[B, self.H, T, T]}")
            print(f"      output               {list(out.shape)}")

        return out


class DenseFFN(nn.Module):
    """逐 token 的两层前馈：FFN(x) = down( GELU( up(x) ) )。

    FFN 不在 token 间做任何混合；它对 [B, T] 中每个位置独立地用同一组参数做
    升维 -> 非线性 -> 降维。token 之间的信息交换全在 attention 里完成。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        h = F.gelu(self.up_proj(x))                      # [B, T, d_ff]
        out = self.down_proj(h)                          # [B, T, C]
        if trace:
            print("    [Dense FFN]")
            print(f"      expanded             {list(h.shape)}")
            print(f"      projected            {list(out.shape)}")
        return out


class Block(nn.Module):
    """Pre-LN Decoder Block（两次残差）：
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = DenseFFN(cfg)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), trace=trace)
        x = x + self.ffn(self.norm2(x), trace=trace)
        return x


class TransformerStack(nn.Module):
    """把 num_layers 个 Block 堆起来，末尾加一个 LayerNorm。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            # 只展开第一层，避免多层重复刷屏。
            x = blk(x, trace=trace and i == 0)
        return self.norm(x)


def main() -> None:
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TransformerStack(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model, device=device)

    print("=" * 64)
    print("MHA + Dense FFN  |  device =", device)
    print(
        f"config: C={cfg.d_model}, H={cfg.num_heads}, "
        f"D={cfg.d_model // cfg.num_heads}, d_ff={cfg.d_ff}, "
        f"layers={cfg.num_layers}"
    )
    print(f"parameters: {n_params:,}")
    print(f"input  x : {tuple(x.shape)}   (B, T, C)")
    print("-" * 64)

    y = model(x, trace=True)

    print("-" * 64)
    print(f"output y : {tuple(y.shape)}   (B, T, C)")
    print(f"y[0, 0, :4] = {[round(v, 4) for v in y[0, 0, :4].tolist()]}")
    print()


if __name__ == "__main__":
    main()
