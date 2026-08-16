NVCC        := nvcc
ARCH        := sm_90
CUTLASS_DIR := ../../../third-party/cutlass
NVCCFLAGS   := -O2 -std=c++17 -arch=$(ARCH) --extended-lambda -I$(CUTLASS_DIR)/include

ex: ex.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: ex
	./ex

clean:
	rm -f ex

.PHONY: run clean

# 顺手刷新仓库根的 compile_commands.json（VS Code 跳转用），细节见该文件注释
include ../../../tools/compile_commands.mk
