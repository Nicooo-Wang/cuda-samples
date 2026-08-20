NVCC      := nvcc
ARCH      := sm_90a
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             --expt-relaxed-constexpr -I$(CUTLASS_DIR)/include

SRCS := cute_tiled_v0.cu cute_tiled_v1.cu cute_tiled_v2.cu cute_tiled_v3.cu \
        cute_tiled_v4.cu cute_tiled_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: 手写搬运基准 =========="
	@./cute_tiled_v0
	@echo ""
	@echo "========== v1: TMA load + mbarrier =========="
	@./cute_tiled_v1
	@echo ""
	@echo "========== v2: TMA store + 边界 =========="
	@./cute_tiled_v2
	@echo ""
	@echo "========== v3: swizzle 与 smem 摆法 =========="
	@./cute_tiled_v3
	@echo ""
	@echo "========== v4: Multi-stage =========="
	@./cute_tiled_v4
	@echo ""
	@echo "========== capstone: 转置 =========="
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
