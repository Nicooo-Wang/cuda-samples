# Makefile

NVCC       = nvcc
ARCH_FLAGS = -arch=sm_80
CUTLASS_DIR = /mnt/c/Users/hamdy/cuda_course/cutlass
COMMONFLAGS = -O3

all: 01_naive 02_wmma 03_cutlass 04_cublass

01_naive: 01_naive.cu
	$(NVCC) $(ARCH_FLAGS) $(COMMONFLAGS) 01_naive.cu -o 01_naive

02_wmma: 02_wmma.cu
	$(NVCC) $(ARCH_FLAGS) $(COMMONFLAGS) 02_wmma.cu -o 02_wmma

04_cublass: 04_cublas.cu
	$(NVCC) $(ARCH_FLAGS) $(COMMONFLAGS) -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=0 04_cublas.cu  -lcublas -o 04_cublass

03_cutlass: 03_cutlass.cu
	$(NVCC) $(ARCH_FLAGS) $(COMMONFLAGS) -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=0 -I$(CUTLASS_DIR)/include -I/mnt/c/Users/hamdy/cuda_course/cutlass/tools/util/include   03_cutlass.cu -o 03_cutlass

clean:
	rm -f 01_naive 02_wmma 04_cublass 03_cutlass *.ncu-rep

.PHONY: all clean

