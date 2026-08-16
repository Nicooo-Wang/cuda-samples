NVCC      := nvcc
ARCH      := sm_90
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             -I$(CUTLASS_DIR)/include

SRCS := cute_copy_v0.cu cute_copy_v1.cu cute_copy_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: Copy_Atom 是什么 =========="
	@./cute_copy_v0
	@echo ""
	@echo "========== v1: make_tiled_copy =========="
	@./cute_copy_v1
	@echo ""
	@echo "========== capstone: 高带宽 memcpy =========="
	@./cute_copy_capstone

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
