NVCC      := nvcc
ARCH      := sm_90
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

# 各版本对应的源文件（每个 .cu 自包含，可单独编译运行）
V1 := v1_baseline
V2 := v2_online
V3 := v3_vectorized
V4 := v4_warp_per_row
V5 := v5_singlepass
V6 := v6_tma

# 默认只构建已实现的版本；随进度把新版本加到 all 里
all: $(V1) $(V2) $(V3) $(V4) $(V5) $(V6)

# 通用编译规则
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# V6 用到 TMA，需要 sm_90a（仅在 Hopper 上能跑）
$(V6): $(V6).cu
	$(NVCC) -O3 -std=c++17 -arch=sm_90a -lineinfo $< -o $@

run: $(V1)
	./$(V1)

# 抓一份完整报告，产物是 <ver>.ncu-rep，用 ncu-ui 或 ncu -i 打开
profile: $(V1)
	ncu --set full -f -o $(V1) ./$(V1)

clean:
	rm -f $(V1) $(V2) $(V3) $(V4) $(V5) $(V6) *.ncu-rep

.PHONY: all run profile clean
