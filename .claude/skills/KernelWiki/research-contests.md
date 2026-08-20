# Research Results: GPU Mode NVFP4 Hackathon + FlashInfer MLSys 2026 Contest

> Raw research data collected 2026-04-16. To be processed into source/wiki pages during implementation.

---

# 1. GPU Mode Blackwell NVFP4 Kernel Hackathon

## 1.1 Overview

Four-part performance challenge hosted by NVIDIA + GPU MODE (Nov 2025 - Feb 2026), targeting B200 GPUs. Individual competition. Sesterce provided DGX B200 compute. Submissions via [Popcorn CLI](https://github.com/gpu-mode/popcorn-cli).

**Reference Kernels Repository**: `gpu-mode/reference-kernels` → `/problems/nvidia/` contains: `nvfp4_gemv/`, `nvfp4_gemm/`, `nvfp4_dual_gemm/`, `modal_nvfp4_dual_gemm/`, `nvfp4_group_gemm/`.

## 1.2 NVFP4 Data Format

- 4-bit floating-point (E2M1): 1 sign, 2 exponent, 1 mantissa
- Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (positive and negative)
- Block scaling: every 16 FP4 elements share one FP8 E4M3 scale factor
- Two-level scaling: per-block E4M3 + per-tensor FP32 global scale
- Dequantization: `x_hat_i = s_global * s_block * deq_FP4(q_i)`
- E4M3 block scale (vs UE8M0 in MXFP4): non-power-of-two scales, smaller block (16 vs 32)

## 1.3 Timeline and Prizes

| Problem | Dates |
|---|---|
| Kernel 1: NVFP4 Batched GEMV | Nov 10 -- Nov 28, 2025 |
| Kernel 2: NVFP4 GEMM | Nov 29 -- Dec 19, 2025 |
| Kernel 3: NVFP4 Gated Dual GEMM | Dec 20, 2025 -- Jan 16, 2026 |
| Kernel 4: NVFP4 Grouped GEMM | Jan 17 -- Feb 13, 2026 |

Prizes per problem: 1st DGX Spark + GTC, 2nd RTX 5090 + GTC, 3rd RTX 5080.
Grand Prize: Dell Pro Max with GB300. Weighted scoring: 10%/20%/30%/40% for problems 1-4.

## 1.4 The Four Problems

### Problem 1: NVFP4 Batched GEMV
- Inputs: Matrix `a` (M×K×L, NVFP4), vector `b` (1×K×L, NVFP4), scales `sfa` (M×K/16×L, FP8), `sfb` (1×K/16×L, FP8)
- Output: `c` (M×1×L, FP16)
- Nature: Memory-bound (low arithmetic intensity)
- Speed of light: ~8.6us for largest config
- Benchmark configs: (M:7168, K:16384, L:1), (M:4096, K:7168, L:8), (M:7168, K:2048, L:4)

### Problem 2: NVFP4 GEMM
- Standard NVFP4 block-scaled GEMM
- Compute-bound, uses tensor cores
- Top: Simon 10.807us, yue 10.914us, currybab 10.931us (geometric mean)

### Problem 3: NVFP4 Gated Dual GEMM
- Two GEMMs (gate + up) → SiLU → element-wise multiply
- Standard MLP gate-up pattern in modern LLMs
- Key: fusion of GEMM + SiLU + multiply

### Problem 4: NVFP4 Grouped GEMM
- Multiple GEMMs with variable M, shared N and K
- Directly relevant to MoE inference
- Heaviest weight (40%) in grand prize

## 1.5 Key Optimization Techniques

### Memory-Bound (Problem 1 GEMV)

**PTX-level control**:
- Raw PTX `cvt.rn.f16x2.e2m1x2`, `ld.global` instead of C intrinsics
- PTX byte unpacking `mov.b32 {tmp0, tmp1, tmp2, tmp3}` instead of bitwise extraction

**Cache policy differentiation**:
- Matrix A (streamed once): `L1::no_allocate` to avoid L1 pollution
- Vector B (reused): `L1::evict_last` to keep hot
- Rank 1 used different `ld.global` qualifiers per K-dimension variant

**Register budgeting**:
- Rank 1: `-maxrregcount=32`; Rank 3: `-maxrregcount=45`
- Lower registers → higher occupancy → critical for memory-bound

**Wider vectorized loads**:
- 128-bit (`ld.global.v2.u64`) and 256-bit (`ld.global.v4.u64`)
- Only effective when PTX byte unpacking avoids bitwise overhead

**Per-K specialization**:
- Separate kernel compilations per K-dimension with full loop unrolling
- Different optimal configs for K=1024, K=3584, K=8192

**Data reuse**:
- Rank 2 shared B vector reads across BLOCK_M rows per thread block

### Compute-Bound (Problems 2-4 GEMM)

- Warp specialization with dedicated TMA and tensor core warps
- CUTLASS `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` schedule
- TMA for async bulk loads to shared memory
- TMEM for MMA results (128×512 matrix, 32-bit elements)
- 128-byte alignment for TMA

## 1.6 Participant Blogs

1. **[Yue's Journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)**: Problem 1. CuTe DSL (100us) → optimized CUDA (22.392us). Steps: coalesced access (2000→443us), hardware intrinsics (443→39us), PTX assembly (39→27us), ILP (27→22.9us).

2. **[Twelve Attempts (Amandeep)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)**: Problem 1. 12 approaches, final ~26.7us (3.1x off SOL). Key lesson: "Run Nsight Compute to confirm memory-bound behavior."

3. **[Simon's posts](https://veitner.bearblog.dev/nvfp4-gemv/)**: CuTe DSL tutorial approach with block parallelization, shared memory reduction, atomic ops.

## 1.7 The Reward Hack

Submission to `nvfp4_group_gemm` reporting 11.191us (~2us ahead of next):
- Correctness phase: harness clones data → real kernel runs correctly
- Timing phase: harness reuses same objects → first call fires 120-group super-batch (all 15 problems fused), calls 2-15 skip
- Led to improvements in FlashInfer-Bench evaluation methodology

## 1.8 B200 Specs (Competition Context)
- sm_100a, 142 SMs
- 8 TB/s memory bandwidth
- Native FP4 (E2M1) tensor core instructions
- TMA, TMEM (128×512 × 32-bit per SM)
- tcgen05.mma operates directly on shared memory

---

# 2. FlashInfer AI Kernel Generation Contest (MLSys 2026)

## 2.1 Overview

One of three MLSys 2026 competitions. Targets B200 GPUs. Human, AI, or hybrid submissions.

**URLs**:
- Contest: [mlsys26.flashinfer.ai](https://mlsys26.flashinfer.ai/)
- Starter kit: [flashinfer-ai/flashinfer-bench-starter-kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
- Agent baseline: [flashinfer-ai/mlsys26-agent-baseline](https://github.com/flashinfer-ai/mlsys26-agent-baseline)
- Dataset: [flashinfer-ai/mlsys26-contest on HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest) (1.88 GB)

## 2.2 Timeline

| Date | Milestone |
|---|---|
| Jan 22, 2026 | Launch, registration opens |
| Feb 5, 2026 | Full dataset released |
| Feb 9, 2026 | Baselines released |
| Feb 15, 2026 | Registration deadline (teams up to 5) |
| Apr 24, 2026 | Kernel submission deadline |
| May 1, 2026 | Technical writeup due (4 pages max) |
| May 11, 2026 | Winners notified |
| May 17-22, 2026 | Award ceremony, Bellevue, WA |

## 2.3 Submission Format

- Fork starter kit per track
- Implement in `solution/triton/kernel.py` or `solution/cuda/kernel.cu` + `binding.py`
- `python scripts/pack_solution.py` → `solution.json`
- Tag commits for biweekly evaluations
- DPS (Destination Passing Style) by default
- CUDA binding via TVM FFI

## 2.4 Track A: Fused MoE (FP8)

**Benchmark**: `fp8_moe_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

**What**: Fused MoE with FP8 block-scale — routing, dispatch, dual GEMM (gate-up + down), SwiGLU, combine.

**Parameters**: topk=8, num_groups=8, topk_group=4, num_experts=32, hidden=7168, intermediate=2048

**Challenges**:
1. No pre-tuned FP8 MoE config for B200
2. FP8 numerical overflow requires careful block scaling (block size 128)
3. Performance varies drastically by batch size
4. Multiple kernel launches (7 in vLLM, 5 in SGLang fused)
5. Variable expert load balancing
6. TMA 128-byte alignment requirements

**Baseline performance**:
| Framework | Batch 4096 TFLOPS | Batch 1 Latency |
|---|---|---|
| SGLang | 1262 | 206.9us |
| FlashInfer CuTeDSL | 1225 | 481.9us |
| vLLM | 1117 | 369.5us |

**API**: `flashinfer.fused_moe.trtllm_fp8_block_scale_moe()`

## 2.5 Track B: DeepSeek V3.2 Sparse Attention

**Benchmarks**: Indexer `fp8_h64_d128_topk2048_ps64`, Attention `h16_ckv512_kpe64_topk2048_ps64`

**Two-stage**:
1. **Lightning Indexer**: FP8 scorer, selects top-2048 tokens per query
2. **Sparse Attention**: MLA over selected 2048 tokens only

**Architecture**: MLA cache = 656 bytes/token (512 FP8 + 16 scales + 128 RoPE). Block size 64.

**Challenges**:
1. Continuous batching with varying sequence lengths
2. Kernel padding overhead (TP=8 → 16 heads padded to 64)
3. Variable sparsity patterns
4. Two-kernel coordination
5. "Nobody has fully unlocked performance potential" (vLLM blog)

**References**: DeepGEMM for indexer, FlashMLA for sparse attention, TileLang for research.

## 2.6 Track C: Gated Delta Net (Qwen3-Next)

**Benchmarks**: Decode `qk4_v8_d128_k_last`, Prefill `qk4_v8_d128_k_last`

**What**: Linear attention with delta rule mechanism (ICLR 2025, NVlabs). Used in Qwen3-Next-80B with 3:1 hybrid ratio (3 GDN : 1 full attention).

**Parameters**: qk_dim=4, v_dim=8, d=128

**Unique challenges**:
1. O(n) linear complexity — different optimization targets than O(n²) attention
2. Recurrent state management with learned decay
3. Gating mechanism adds branching complexity
4. Dual-mode: chunk-based prefill, streaming decode
5. Variable-length support needed for production

**Status** (FlashInfer issue #1690):
- Prefill: Done on Hopper, in progress for Blackwell
- Decode: Done for both Hopper and Blackwell

**Implementations**: NVlabs/GatedDeltaNet (reference), FLA optimized (recommended, faster, varlen support).

## 2.7 Agent Baseline

[mlsys26-agent-baseline](https://github.com/flashinfer-ai/mlsys26-agent-baseline):
- Iterative Agent: propose → refine via `str_replace`
- Evolve Agent: multiple proposals → elite pool → evolution
- Supports OpenAI + Claude models

## 2.8 FlashInfer-Bench Leaderboard (Current)

| Model | Avg Speedup | Resolved % |
|---|---|---|
| Gemini 2.5 Pro | 0.628x | 73.1% |
| GPT-5 (2025-08-07) | 0.467x | 92.3% |
| Claude Opus 4.1 | 0.456x | 73.1% |
| GPT-O3 | 0.450x | 92.3% |

All below 1.0x vs FlashInfer baselines — AI-generated kernels still lag behind human-optimized.

## 2.9 K-Search

Automated kernel generation framework using co-evolving world model to guide LLM optimization. 2.10x average improvement over OpenEvolve, up to 14.3x on complex MoE kernels.

---

## Key Sources

- [GPU MODE Hackathon (Luma)](https://luma.com/9n27uem4)
- [NVIDIA Forums Announcement](https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092)
- [Yue's Blog](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Twelve Attempts (Amandeep)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Simon's Blog](https://veitner.bearblog.dev/nvfp4-gemv/)
- [Reward Hack Writeup](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [TFLOPS Gap Blog](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
- [NVFP4 Format Details](https://haroldbenoit.com/notes/ml/engineering/precision/nvfp4-format)
- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels)
- [MLSys 2026 Contest](https://mlsys26.flashinfer.ai/)
- [FlashInfer Starter Kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
- [FlashInfer Agent Baseline](https://github.com/flashinfer-ai/mlsys26-agent-baseline)
- [FlashInfer-Bench Paper](https://arxiv.org/abs/2601.00227)
- [FlashInfer-Bench Leaderboard](https://bench.flashinfer.ai/)
- [mlsys26-contest Dataset](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)
- [K-Search Paper](https://arxiv.org/abs/2602.19128)
