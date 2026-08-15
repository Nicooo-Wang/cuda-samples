NVCC      := nvcc
ARCH      := sm_90
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: softmax

softmax: softmax.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: softmax
	./softmax

# 抓一份完整报告，产物是 softmax.ncu-rep，用 ncu-ui 或 ncu -i 打开
profile: softmax
	ncu --set full -f -o softmax ./softmax

clean:
	rm -f softmax *.ncu-rep

.PHONY: all run profile clean

# 顺手刷新仓库根的 compile_commands.json（VS Code 跳转用），细节见该文件注释
include ../../tools/compile_commands.mk
