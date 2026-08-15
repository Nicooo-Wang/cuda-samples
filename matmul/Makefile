NVCC      := nvcc
ARCH      := sm_90
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda -Xcompiler -fopenmp
LDLIBS    := -lcublas -lgomp

# 优化阶梯，从上到下逐步逼近 cuBLAS
SRCS := v0_wmma_naive.cu \
        v1_smem.cu \
        v2_warp_tile.cu \
        v3_smem_padding.cu \
        v4_tile_tuning.cu \
        v5_cp_async.cu \
        v6_cublas.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

run: all
	@for b in $(BINS); do ./$$b; echo; done

# 逐个 binary 抓一份完整报告，产物是 <name>.ncu-rep，用 ncu-ui 或 ncu -i 打开。
# 各版本都只启动一次 kernel，所以 ncu 看到的就是程序自己打印的那一次。
profile: all
	@for b in $(BINS); do echo "==> profiling $$b"; ncu --set full -f -o $$b ./$$b; done

clean:
	rm -f $(BINS) *.ncu-rep

# %: %.cu 是 match-anything 规则，产物会被当成中间文件用完自动删掉，
# 加 .SECONDARY 让二进制留在磁盘上（方便 ncu 直接 profile）
.SECONDARY: $(BINS)

.PHONY: all run profile clean

# 顺手刷新仓库根的 compile_commands.json（VS Code 跳转用），细节见该文件注释
include ../tools/compile_commands.mk
