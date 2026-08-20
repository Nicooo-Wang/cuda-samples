NVCC      := nvcc
# WGMMA / TMA / cluster 需要 sm_90a (不是 sm_90)
ARCH      := sm_90a
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include

SRCS := cute_gemm_v0.cu cute_gemm_v1.cu cute_gemm_v2.cu cute_gemm_v3.cu cute_gemm_v4.cu cute_gemm_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: naive GEMM —— 先搭骨架 =========="
	@./cute_gemm_v0
	@echo ""
	@echo "========== v1: smem 中转 (Ampere 单缓冲) =========="
	@./cute_gemm_v1
	@echo ""
	@echo "========== v2: 多 stage cp.async (Ampere 上限) =========="
	@./cute_gemm_v2
	@echo ""
	@echo "========== v3: TMA + WGMMA + mbarrier (Hopper 正解) =========="
	@./cute_gemm_v3
	@echo ""
	@echo "========== v4: Warp Specialization =========="
	@./cute_gemm_v4
	@echo ""
	@echo "========== capstone: Block Cluster =========="
	@./cute_gemm_capstone

# 练习：题目见 README 的"练习"一节，答案填在 exercises/ex.cu 的 TODO 处
ex:
	@$(MAKE) -C exercises run

clean:
	rm -f $(BINS) *.o
	@$(MAKE) -C exercises clean

.SECONDARY: $(BINS)

.PHONY: all run ex clean

# 顺手刷新仓库根的 compile_commands.json（VS Code 跳转用），细节见该文件注释
include ../../tools/compile_commands.mk
