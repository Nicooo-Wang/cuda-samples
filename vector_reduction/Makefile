NVCC      := nvcc
ARCH      := sm_90
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SRCS := v0_interleaved_divergent.cu v1_contiguous.cu v2_two_per_thread.cu \
        v3_warp_unroll.cu v4_warp_shuffle.cu v5_grid_stride.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@for b in $(BINS); do ./$$b; done

# 逐个 binary 抓一份完整报告，产物是 <name>.ncu-rep，用 ncu-ui 或 ncu -i 打开
profile: all
	@for b in $(BINS); do echo "==> profiling $$b"; ncu --set full -f -o $$b ./$$b; done

clean:
	rm -f $(BINS)

cleanall: clean
	rm -f *.ncu-rep

# %: %.cu 是 match-anything 规则，产物会被当成中间文件用完自动删掉，
# 加 .SECONDARY 让二进制留在磁盘上（方便 ncu 直接 profile）
.SECONDARY: $(BINS)

.PHONY: all run profile clean cleanall
