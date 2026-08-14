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

clean:
	rm -f $(BINS) *.o

.SECONDARY: $(BINS)

.PHONY: all run clean
