NVCC      := nvcc
# swizzle 本身 sm_80 就有, 但 GMMA::Layout_K_SW*_Atom 属于 sm_90 -> 统一用 sm_90a
ARCH      := sm_90a
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include

SRCS := cute_tiled_v0.cu cute_tiled_v1.cu cute_tiled_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: bank conflict 与 Swizzle =========="
	@./cute_tiled_v0
	@echo ""
	@echo "========== v1: 协作式 Copy Atom (ldmatrix) =========="
	@./cute_tiled_v1
	@echo ""
	@echo "========== capstone: 矩阵转置三版对比 =========="
	@./cute_tiled_capstone

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
