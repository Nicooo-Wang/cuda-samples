# Research Results: DeepSeek + Qwen Kernel Optimizations

> Raw research data collected 2026-04-16. To be processed into wiki pages after plan finalization.

---

# DeepSeek Kernel Optimizations

## 1. DeepGEMM: FP8 GEMM Kernels with Fine-Grained Scaling

**Source**: [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
**Problem**: Achieving high-performance FP8 matrix multiplication with fine-grained per-tile/per-block scaling factors for training and MoE inference.

**Key techniques**:
- **Fine-grained quantization**: Tile-wise 1x128 scaling for activations, block-wise 128x128 scaling for weights. This prevents outliers from destroying quantization precision.
- **FP8 accumulation with CUDA core promotion**: On Hopper (SM90), every 4 consecutive WGMMA operations accumulate in Tensor Core limited-precision registers, then promote the partial sum to a separate FP32 accumulator on CUDA Cores. The Nc=128 interval (4 WGMMAs) is the minimum that significantly improves precision without substantial overhead.
- **SM90 kernel**: Uses NT (non-transposed A, transposed B) memory layout only. Scaling factors in FP32 format. Leverages WGMMA with JIT compilation via a lightweight module.
- **SM100 (Blackwell) kernel**: Supports all memory layouts (NT, TN, NN, TT). Scaling factors use packed UE8M0 format (4 values per integer). Uses tcgen05.mma with tensor memory for accumulation.
- **MoE grouped GEMMs**: Groups only the M-axis (variable tokens per expert) while N and K remain fixed -- tailored for MoE expert computations. Supports contiguous layout (prefill), masked layout (decode with CUDA graphs), and K-axis grouped (weight gradients).
- **Performance**: Up to 1550 TFLOPS on H800. Only ~300 lines of core CUDA kernel code.

**Hopper/Blackwell features leveraged**: WGMMA (SM90), tcgen05.mma (SM100), TMA, tensor memory, JIT compilation via NVRTC.

## 2. FlashMLA: Multi-head Latent Attention Kernels

**Source**: [github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
**Problem**: Efficient MLA decoding for DeepSeek-V3/V3.2 models with massive KV cache compression (70KB/token vs 327-516KB for competitors).

**Key techniques**:
- **Dense MLA decoding** (SM90): BF16, paged KV cache (block size 64). Up to 3000 GB/s memory bandwidth, 660 TFLOPS compute-bound on H800.
- **Sparse MLA decoding** (SM90/SM100): FP8 KV cache with token-level sparsity via indices tensor. Each token occupies 656 bytes: 512 bytes FP8 data + 16 bytes FP32 scales + 128 bytes BF16 RoPE embeddings. Achieves 410 TFLOPS on H800, 350 TFLOPS on B200.
- **Dense prefill** (SM100 only): MHA forward/backward. 1460 TFLOPS forward, 1000 TFLOPS backward on B200.
- **Sparse prefill** (SM90/SM100): Up to 640 TFLOPS on H800, 1450 TFLOPS on B200.

**Blackwell features**: SM100 kernels use tcgen05.mma and tensor memory. The B200 achieves substantially higher TFLOPS for prefill operations compared to H800.

## 3. Native Sparse Attention (NSA)

**Source**: [arxiv.org/abs/2502.11089](https://arxiv.org/abs/2502.11089) (ACL 2025), [github.com/lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch)
**Problem**: Reducing attention compute for long sequences (64K+) without sacrificing quality, while being natively trainable (end-to-end differentiable).

**Key techniques**:
- **Three parallel attention paths**: (1) Token compression via learnable MLP creating coarse-grained representations; (2) Token selection using blockwise importance scores from compression attention to pick top-n fine-grained blocks; (3) Sliding window (w=512) for local context.
- **Hardware-aligned design**: Blockwise memory access exploits spatial continuity for contiguous GPU memory loads. Group-centric loading shares sparse KV blocks across all query heads in a GQA group, minimizing redundant KV transfers.
- **Triton kernel implementation**: Group-centric data loading, shared KV fetching, grid-based loop scheduling. Achieves near-optimal arithmetic intensity.
- **Performance**: 9x forward speedup, 6x backward speedup at 64K sequences vs FlashAttention-2. 11.6x decoding speedup at 64K context.

**Deployed in**: DeepSeek-V3.2-Exp with FlashMLA sparse kernels. SGLang and vLLM both provide day-0 support.

## 4. DeepEP: Expert Parallelism Communication Library

**Source**: [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
**Problem**: Efficient all-to-all GPU communication for MoE dispatch/combine across multi-node clusters.

**Key techniques**:
- **Two dispatch modes**: Normal (high throughput, prefill/training) with NVLink-to-RDMA forwarding; Low-latency (minimal delay, decode) with pure RDMA and hook-based overlap that consumes zero SMs.
- **Undocumented PTX instruction**: `ld.global.nc.L1::no_allocate.L2::256B` -- uses non-coherent read-only path with L1 cache bypass to avoid thrashing between communication and compute streams.
- **NVLink/InfiniBand overlap**: Tokens routed to max 4 nodes. First hop via InfiniBand, then forwarded via NVLink. IB and NVLink transfers fully overlap.
- **FP8 dispatch / BF16 combine**: Communication volume halved for dispatch while maintaining precision on the combine path.
- **Performance**: Intra-node 153-158 GB/s (near NVLink max of 160 GB/s). Low-latency: 77us for 8 experts, 192us for 256 experts.

## 5. DualPipe: Bidirectional Pipeline Parallelism

**Source**: [github.com/deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)
**Problem**: Hiding communication latency in MoE training.

**Key techniques**:
- Chunks divided into 4 components: attention, all-to-all dispatch, MLP, all-to-all combine.
- Bidirectional scheduling feeds micro-batches from both pipeline ends simultaneously.
- While one micro-batch does MLA/MoE compute, another simultaneously handles dispatch communication.
- Neither bubbles nor activation memory increase with micro-batch count.

## 6. EPLB: Expert Parallel Load Balancer

**Source**: [github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
**Problem**: Uneven expert utilization in MoE models causing GPU load imbalance.

**Key techniques**:
- Three-tier balancing: node-level grouping, intra-node expert replication, GPU allocation.
- Redundant expert strategy: duplicates heavy-loaded experts across GPUs.
- Achieves 1.49x prefill speedup and 2.54x decode speedup when enabled.

## 7. DeepSeek-V3 FP8 Training Framework

**Source**: [arxiv.org/abs/2412.19437](https://arxiv.org/html/2412.19437v1)
**Problem**: Training a 671B MoE model cost-effectively on H800 GPUs.

**Key techniques**:
- First validated FP8 training at extreme scale (2788M GPU-hours on 2048 H800s).
- Fine-grained quantization: tile-wise 1x128 activations, block-wise 128x128 weights.
- Hopper hardware limitation: Tensor Cores constrain accumulation to FP22 not true FP32. The Nc=128 promotion technique mitigates this.
- Node-limited routing: 256 experts grouped into 8 groups (32/node), tokens route to max 4 nodes.
- MLA KV cache compression to 70KB/token (4.66-7.28x vs competitors).

---

# Qwen Kernel Optimizations

## 1. Qwen3-Next: Hybrid GatedDeltaNet + MoE Architecture

**Source**: [NVIDIA blog](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
**Problem**: Efficient long-context inference with extreme parameter sparsity.

**Architecture**: 80B parameters, only 3B active per token. 48 layers in pattern: 12 x (3 x [GatedDeltaNet -> MoE] -> [Full Attention -> MoE]). 512 routed experts with 1:50 activation ratio.

**Key techniques**:
- **Gated DeltaNet** (75% of layers): Linear attention with O(n) complexity. Uses delta rule for error-correcting memory updates, exponential gating for adaptive decay.
- **Full attention** (25% of layers): Standard GQA attention for global context and strong retrieval.
- **Ultra-sparse MoE**: 512 experts, only ~19 active per token.

## 2. Qwen3.5: Full Gated DeltaNet Integration

**Architecture**: 60 layers = 15 x (3 x [GatedDeltaNet -> MoE] -> 1 x [Full Attention -> MoE]). Supports 262K token context windows.

**Key techniques**:
- Attention output gating eliminates Attention Sink and Massive Activation problems.
- O(1)-per-token inference for linear attention layers during decoding.
- 10x+ throughput improvement over Qwen3-32B at 32K+ context lengths.

## 3. Gated Delta Networks (GatedDeltaNet)

**Source**: [github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) (ICLR 2025)
**Problem**: Linear attention mechanism matching standard attention quality at O(n) complexity.

**Key techniques**:
- **Delta rule mechanism**: Targeted state updates that know what to keep and forget.
- **Exponential gating**: Adaptive memory decay preventing state saturation.
- **Chunk-based parallelism**: Sequences divided into chunks processed in parallel.
- **Two kernel implementations**: NVlabs reference Triton kernels; FLA optimized kernels (recommended, significantly faster).

**GPU kernel challenges on Blackwell**: Triton-based kernels have CPU launch overhead impacting decode batches. vLLM enables full CUDA graph mode to mitigate.

## 4. Tiled Flash Linear Attention (TFLA)

**Source**: [arxiv.org/abs/2503.14376](https://arxiv.org/html/2503.14376v1)
**Problem**: Flash Linear Attention chunk sizes limited, causing many intermediate state materializations.

**Key techniques**:
- Two levels of sequence parallelism: standard chunkwise + tiling within chunks.
- Enables arbitrarily large chunk sizes.
- Matmuls emitted as inline PTX assembly for Hopper (WGMMA) and Blackwell (tcgen05).

---

# Blackwell/Hopper Kernel Programming Resources

## 1. FlashAttention-4

**Source**: [arxiv.org/abs/2603.05451](https://arxiv.org/abs/2603.05451), [Tri Dao's blog](https://tridao.me/blog/2026/flash4/)
**Problem**: Blackwell's asymmetric scaling -- tensor core throughput doubles but SFU count and SMEM bandwidth unchanged.

**Key techniques**:
- **Ping-pong scheduling**: Two 128-token query tiles per CTA with dedicated softmax warpgroups handling TMEM.
- **Software-emulated exponential**: Distributes 2^x across FMA units via Cody-Waite range reduction + Horner polynomial. Multiplies exp throughput without additional SFU hardware.
- **Conditional softmax rescaling**: Only rescales when max jump is large, reducing non-matmul ops.
- **2-CTA backward**: Spans paired CTAs in a cluster, sharing TMEM. Halves shared memory traffic.
- **Performance**: Up to 1605 TFLOPS on B200 BF16 (71% utilization). 1.1-1.3x over cuDNN, 2.1-2.7x over Triton.
- **Implementation**: Written in CuTe-DSL (Python), 20-30x faster compilation than C++ templates.

## 2. tcgen05 for Dummies (gau-nernst)

**Source**: [gau-nernst.github.io/tcgen05/](https://gau-nernst.github.io/tcgen05/)
**Key findings**:
- Achieved 98% of CuBLAS speed: basic (255 TFLOPS) -> 128B swizzling (695) -> pipelining (940) -> persistent kernel (1476 vs 1507 CuBLAS).
- tcgen05.mma operates directly on shared memory -- no ldmatrix needed.
- "Tensor Cores programming on Blackwell is easier than previous generations" due to hardware abstractions.

## 3. Colfax Research CUTLASS Tutorials

**Source**: [Tensor Memory GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/), [Sub-byte GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
**Key patterns**:
- UMMA replaces WGMMA: register-free operation, single-thread launch, built-in block scaling for FP4/FP6/FP8.
- CUTLASS two-level abstraction: MMA_Atom (PTX wrapper) and MMA_Traits (CuTe layouts).

## 4. Modular Blog Series: Matrix Multiplication on Blackwell

**Source**: [Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction), [Part 3](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
**Key techniques**:
- TMA multicasting, 2-SM MMA, 5-stage circular buffer pipelining.
- Reached 85% of SOTA with clear progression.

## 5. CUTLASS SM100 Attention and MLA Kernels

**Source**: [CUTLASS Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
**Key additions**:
- SM100 Attention kernels with fused reduction for MLA.
- FlashMLA-like weight-absorbed decoding kernel.
- MLA supports splitting K across multiple SMs.
- 16-warp kernels with distinct warp specialization roles.

## 6. Blackwell Microbenchmarking

**Source**: [arxiv.org/abs/2512.02189](https://arxiv.org/html/2512.02189v1), [arxiv.org/abs/2507.10789](https://arxiv.org/html/2507.10789v2)
**Key findings**:
- TMEM achieves 420 clock cycles for end-to-end cache-miss access -- 58% reduction vs Hopper's 1000 cycles.
- B200 tensor cores: 1.56x higher mixed-precision throughput, 42% better energy efficiency than H200.
- TMEM best for multi-stage tensor pipelines with large working sets; SMEM better for single-shot small-matrix ops.

## 7. JAX Pallas Blackwell Matmul

**Source**: [JAX Pallas docs](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)
Full standalone optimized kernel with performance comparisons to CUTLASS and cuBLAS.

## 8. TileLang

**Source**: [github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang)
**Key features**:
- Pythonic DSL on TVM. Auto TMA/WGMMA on H100.
- Flash MLA Decoding on par with FlashMLA on H100.
- Fused attention in under 80 lines of Python.
- Supports H100, A100, V100, RTX, AMD MI250/MI300X, Apple Metal, Ascend NPUs.

## 9. Tilus (NVIDIA)

**Source**: [github.com/NVIDIA/tilus](https://github.com/NVIDIA/tilus), ASPLOS 2025
**Key features**:
- Thread-block-level programming with hierarchical memory space.
- Supports arbitrary bit widths 1 to 8 bits.
- Targets SM100, SM103, SM110, SM120, SM121 (full Blackwell family).
- Outperforms Triton (1.75x), Ladder (2.61x), QuantLLM (1.29x), Marlin (1.03x).

## 10. Hopper-to-Blackwell Migration Patterns

| Aspect | Hopper (SM90) | Blackwell (SM100) |
|---|---|---|
| MMA instruction | wgmma.mma_async (warpgroup scope) | tcgen05.mma (single-thread, CTA scope) |
| MMA output | Registers | Tensor Memory (TMEM, 256KB/SM) |
| Max BF16 MMA shape | m64n256k16 | m128n256k16 (1-CTA), m256n256k16 (2-CTA) |
| Matrix loading | ldmatrix to registers | Direct from SMEM (no ldmatrix) |
| Synchronization | Warpgroup (4 warps) | Single thread launch, fully async |
| New data types | FP8 (E4M3, E5M2) | FP4, FP6, FP8 with block scaling |
| Scaling support | External (CUDA core promotion) | Native UE8M0 block scaling in MMA |
| Register pressure | High (accumulator + operands) | Low (TMEM holds accumulators) |
