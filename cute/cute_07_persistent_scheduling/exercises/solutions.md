# cute_07 练习解答

题面在 `ex.cu`，README 里的对应节号写在每道题开头的注释里。

---

## 1 —— tile 编号 → 坐标的两种映射（§2.1）

8×8 网格，id=5：

**row-major**：

```cpp
by_rm = 5 / 8 = 0;
bx_rm = 5 % 8 = 5;   // -> (0,5)
```

**swizzled**（SWIZ=2，按 2 个 tile 一组沿 M 排）：

```cpp
int group = 5 / 2 = 2;   // 第 2 组
int within = 5 % 2 = 1;  // 组内第 1 个
int group_row = 2 / 8 = 0;  // 组在第 0 行
int group_col = 2 % 8 = 2;  // 组在第 2 列
by_sw = 0 * 2 + 1 = 1;
bx_sw = 2;                 // -> (1,2)
```

> 对比：row-major 的 id=5 是 (0,5)，swizzled 是 (1,2)。swizzled 让相邻 id
> 在 2D 空间上聚成小块（块大小 = SWIZ），这就是空间局部性的来源。

---

## 2 —— persistent 主循环的边界条件（§1.1）

grid=4，tiles=10：

```
CTA 0: 0 4 8     (3 个)
CTA 1: 1 5 9     (3 个)
CTA 2: 2 6       (2 个)
CTA 3: 3 7       (2 个)
```

```cpp
int avg = 10 / 4 = 2;        // 整除
int max_tiles = 2 + (10 % 4 != 0 ? 1 : 0) = 3;   // 前 2 个 CTA 多吃一个
```

> 负载不均衡 = 3/2.5 = 1.2x。tile 越少、CTA 越多，不均衡越明显。
> 这是 persistent 的固有代价，CUTLASS 用"tile 按需领取 + 细粒度调度"缓解。

---

## 3 —— barrier phase 重置（§1.2）

每个 tile 开头补三行：

```cpp
if (warp == 0 && one) ProducerBar::init(&bar[0], 1);
cutlass::arch::fence_barrier_init();
__syncthreads();
```

**为什么必须重置**：barrier 的 phase 在上一个 tile 结束时已经翻转（arrive 过、
wait 过）。第二个 tile 里 `wait(&bar[0], 0)` 等的是"phase 0 完成"，但 barrier
现在停在 phase 1 —— 于是 wait 立即返回（或永远等不到，取决于实现），
TMA 的数据还没到就往下走 → 数据错（或死锁）。

> 不填的话跑出来是 `验证失败的元素个数 = 2` —— 第二个 tile 的数据是旧的。
> 填对后 0。这就是"phase 对不上"最直观的现场。
