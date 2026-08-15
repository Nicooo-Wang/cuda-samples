NVCC      := nvcc
ARCH      := sm_90
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             -I$(CUTLASS_DIR)/include

SRCS := cute_layout_v0.cu cute_layout_v1.cu cute_layout_v2.cu cute_layout_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: Layout 基础 =========="
	@./cute_layout_v0
	@echo ""
	@echo "========== v1: Layout 代数运算 =========="
	@./cute_layout_v1
	@echo ""
	@echo "========== v2: Layout 操作数据 =========="
	@./cute_layout_v2
	@echo ""
	@echo "========== capstone: GPU 转置 =========="
	@./cute_layout_capstone

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
