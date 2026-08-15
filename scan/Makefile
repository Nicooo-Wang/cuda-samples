NVCC     := nvcc
ARCH     := sm_90
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo --extended-lambda

SRCS := v0_naive_hillis_steele.cu v1_work_efficient.cu v2_bank_conflict_free.cu \
        v3_arbitrary_length.cu v4_warp_shuffle.cu v5_single_pass_lookback.cu \
        v6_striped_load.cu
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@for b in $(BINS); do ./$$b; echo; done

# 逐个 binary 抓一份完整报告，产物是 <name>.ncu-rep，用 ncu-ui 或 ncu -i 打开
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
