NVCC      := nvcc
# WGMMA / TMA 需要 sm_90a (不是 sm_90)
ARCH      := sm_90a
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include

SRCS := cute_mma_v0.cu cute_mma_v1.cu cute_mma_v2.cu cute_mma_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: MMA Atom 的解剖 (Ampere warp 级) =========="
	@./cute_mma_v0
	@echo ""
	@echo "========== v1: WGMMA —— warpgroup 级异步 MMA =========="
	@./cute_mma_v1
	@echo ""
	@echo "========== v2: TMA —— 喂给 MMA 的数据通路 =========="
	@./cute_mma_v2
	@echo ""
	@echo "========== capstone: 单 CTA 的 TMA + WGMMA =========="
	@./cute_mma_capstone

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
