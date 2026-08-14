NVCC      := nvcc
ARCH      := sm_90
CUTLASS_DIR := ../../third-party/cutlass
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda \
             -I$(CUTLASS_DIR)/include

SRCS := cute_tensor_v0.cu cute_tensor_v1.cu cute_tensor_v2.cu cute_tensor_capstone.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "========== v0: CPU Tensor 基础 =========="
	@./cute_tensor_v0
	@echo ""
	@echo "========== v1: GPU Tensor (gmem/smem) =========="
	@./cute_tensor_v1
	@echo ""
	@echo "========== v2: local_partition =========="
	@./cute_tensor_v2
	@echo ""
	@echo "========== capstone: GEMV =========="
	@./cute_tensor_capstone

clean:
	rm -f $(BINS) *.o

.SECONDARY: $(BINS)

.PHONY: all run clean
