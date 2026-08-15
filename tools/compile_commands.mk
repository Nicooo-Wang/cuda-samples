# 让 `make` 顺手刷新仓库根的 compile_commands.json（给 VS Code / clangd 跳转用）。
#
# 用法：在教程 Makefile 的**最后一行**加上
#     include ../tools/compile_commands.mk
# （在 softmax/2d、cute/xxx/exercises 这类两层深的目录里写 ../../tools/compile_commands.mk）
#
# 为什么放最后：这里只有变量赋值和一次 $(shell ...)，没有规则，
# 所以不会抢走"第一个目标就是默认目标"这条规矩。放最后纯粹是为了读起来清楚。

# ---- 递归刹车 ----
# gen_compile_commands.py 会去每个教程目录跑 `make -B -n` 把编译命令抓出来，
# 那些子 make 同样会解析到本文件。不刹一脚的话：
#   make -> 脚本 -> 11 个 make -> 11 个脚本 -> 121 个 make -> ...
# 脚本给子 make 的环境里塞了 CUDA_COURSE_CC_GEN=1，看见它就整段跳过。
#
# 另外 `make -n`（dry run）也跳过：那种场合用户只想看命令，不该有副作用。
# MAKELEVEL 也顺手一挡：`make ex` 会 $(MAKE) -C exercises，父层解析时已经刷过一次了，
# 子 make 再刷一遍纯属浪费。（这一条挡不住脚本里那些子 make——它们是 level 0，
# 所以上面的环境变量守卫不能省。）
ifndef CUDA_COURSE_CC_GEN
ifeq (0,$(MAKELEVEL))
ifeq (,$(findstring n,$(firstword -$(MAKEFLAGS))))

# 从本文件自己的路径反推仓库根，这样不管在哪层目录 include 都能找对。
# MAKEFILE_LIST 的最后一项就是"正在解析的这个文件"，即 <repo>/tools/compile_commands.mk
CC_GEN_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
CC_GEN_PY   := $(CC_GEN_ROOT)/tools/gen_compile_commands.py

# $(shell ...) 在 Makefile **解析期**执行，也就是任何编译动作之前，
# 所以不用挂到某个具体目标上，`make`/`make run`/`make clean` 都会刷新一次。
# 全量重跑约 70ms，随手挂着没有感知。
#
# -q：内容没变时一个字都不打（避免每次 make 都刷一行噪音，也避免 mtime 变化
#     让 cpptools 重扫数据库）。
# 2>/dev/null && || true：环境里没 python3 也不许影响编译。
CC_GEN_OUT := $(shell python3 $(CC_GEN_PY) -q 2>/dev/null || true)
ifneq (,$(CC_GEN_OUT))
$(info $(CC_GEN_OUT))
endif

endif
endif
endif
