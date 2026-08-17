# cute_04 练习参考解答

先自己做，卡住了再看。每题都标了对应的 README 小节。

---

## 练习 1 — 数 bank ★☆☆

**答案：32**

```cpp
constexpr int EX1_CONFLICT = 32;
```

第 5 列的偏移是 `s(l, 5) = l * 32 + 5`（行 stride = 32）：

```
l=0: 偏移   5    bank = 5 % 32 = 5
l=1: 偏移  37    bank = 37 % 32 = 5
l=2: 偏移  69    bank = 69 % 32 = 5
...
l=31: 偏移 997   bank = 997 % 32 = 5
```

每行偏移加 32，而 `32 % 32 = 0`，所以 bank 号一动不动。32 个 lane 全撞 bank 5，
**32-way conflict**。

这不是第 5 列特有的。第 c 列的 lane l 偏移是 `l*32 + c`，bank 是
`(l*32 + c) % 32 = c % 32`——与 l 无关，任何列都一样全撞。

> README §1

---

## 练习 2 — 手算 swizzle 映射 ★★☆

**答案：197**

```cpp
constexpr int EX2_SWZ_OFF = 197;
```

按 §3.2 的四步，`Swizzle<5,0,5>` 作用在 `(r,c) = (6,3)`：

```
第 1 步  plain_off = 6*32 + 3 = 195 = 0b0011000011
                                        \_r=6_/\_c=3_/

第 2 步  取出高 5 位 (S=5, 即右移 5 位) = 0b00110 = 6      ← 就是 r

第 3 步  M=0, 不左移; 和低 5 位异或:
           c XOR r = 0b00011 XOR 0b00110 = 0b00101 = 5

第 4 步  拼回去 = r*32 + 5 = 192 + 5 = 197
```

一句话记法：`Swizzle<5,0,5>` 就是 **`c_new = c XOR r`**，行号不变。

检查一下合理性：`swz(6,3) = 197`，而 `plain(6,3) = 195`。两者的 bank 分别是
`197 % 32 = 5` 和 `195 % 32 = 3`——同一列的不同行被推到了不同 bank，这正是
swizzle 要达到的效果。

> README §3.2

---

## 练习 3 — 选 M 保住向量化 ★★★

**答案：`EX3_VEC_MASK = 4`（只有 `Sw<3,2,3>` 可用）**

```cpp
constexpr int EX3_VEC_MASK = 4;   // 0b100: bit2 = Sw<3,2,3>
```

`M` 保护最低 M 位不参与异或，所以 **2^M 个相邻元素保持连续**：

| swizzle | M | 2^M | 行内最短连续段 | 128-bit 需要 4 个 | 列读冲突 |
|---|---|---|---|---|---|
| `Sw<5,0,5>` | 0 | 1 | 1 个 float | ✗ | 1-way |
| `Sw<4,1,4>` | 1 | 2 | 2 个 float | ✗ | 2-way |
| `Sw<3,2,3>` | 2 | 4 | 4 个 float | ✓ | 4-way |

看第 1 行的偏移就明白了（v0 会打印）：

```
Sw<5,0,5>  r=1:  33 32 35 34 37 36 ...   每个元素单独被打乱, 一个连续对都没有
Sw<4,1,4>  r=1:  34 35 32 33 38 39 ...   2 个一组
Sw<3,2,3>  r=1:  36 37 38 39 32 33 ...   4 个一组  ← 刚好够一条 128-bit
```

**这题的重点是那个反直觉的结论**：冲突最少的 `Sw<5,0,5>`（1-way）恰恰是唯一
不能向量化的。用它配 128-bit atom 会编译期失败：

```
static assertion failed:
  "Copy_Traits: dst failed to vectorize into registers.
   Layout is incompatible with this CopyOp."
```

所以选参数的顺序是：**先按向量宽度定 M，再让 B/S 去消冲突**。GMMA 四个官方原子
M 全部 = 4（2⁴ = 16 个 half = 32 字节），就是这条规则的体现。

> README §3.4

---

## 练习 4 — 选对 GMMA swizzle 原子 ★★☆

**答案：`EX4_MASK = 0b1110 = 14`**

```cpp
constexpr int EX4_MASK = 14;
```

BK = 32，规则是 **BK 必须能被原子的 K 长度整除**：

| 原子 | K 长度 | `32 % K` | 可用？ |
|---|---|---|---|
| SW128 | 64 | `32 % 64 = 32 ≠ 0` | ✗ |
| SW64 | 32 | `32 % 32 = 0` | ✓ |
| SW32 | 16 | `32 % 16 = 0` | ✓ |
| INTER | 8 | `32 % 8 = 0` | ✓ |

只有 SW128 不行（BK 比它的 K 维还小），bit0 = 0，其余 = 1 → `0b1110 = 14`。

违反了会编译期报 `"tile_to_shape: block shape does not divide the target shape"`。

**实用规则：选能用的里面 K 最长的**——对齐最大、访存最宽。这里选 SW64。

> README §5.4

---

## 练习 5 — 谁挑 layout ★★☆

**答案：TMA 能搬对（true），WGMMA 编译不过（false）**

```cpp
constexpr bool EX5_TMA_OK = true;
constexpr bool EX5_WGMMA_OK = false;
```

这题就是要打掉一个常见误解：**"TMA 要求 swizzle"是错的。**

v2 §5.2 实测五种 smem layout 走真实 TMA load（128×64 half）：

```
  SW128 / SW64 / SW32 / INTER / plain row-major
  -> 五种全部 TMA = OK, 落数 = 正确
```

TMA 只是照 descriptor 搬，plain row-major 也搬得又快又对。差别只体现在
**consumer 侧读它时的冲突**：plain 是 32-way，INTER 是 4-way。

真正拒绝 plain 的是 WGMMA，而且是**编译期**拒绝（v2 §5.3）：

```
static assertion failed:
  "Not a canonical GMMA_K Layout: Expected stride failure."
```

想亲眼看到，把 `ex.cu` 里的 `EX5_TRY_PLAIN_WGMMA` 改成 1 再编译一次。

原因：WGMMA 不经过寄存器，而是把 smem 地址和摆法编码成一个 **descriptor**，
硬件按 descriptor 直读 smem。descriptor 里只有几个比特存 swizzle 模式，
能表达的摆法就那么四种（就是练习 4 的四个原子），plain 和 padded 都不在其中。

> README §5.2 §5.3

---

## 练习 6 — 修一个 smem layout bug ★★★

**答案：`Swizzle<3,2,3>`**

```cpp
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
auto slay  = composition(Swizzle<3, 2, 3>{}, plain);
```

这题有两个检查，故意用来排除"看起来最优"的答案：

| 候选 | 列读冲突 ≤ 4-way | 行内连续 ≥ 4 | 结果 |
|---|---|---|---|
| `plain`（原样） | ✗ 32-way | ✓ 32 | 不合格 |
| `Swizzle<5,0,5>` | ✓ 1-way | ✗ 只有 1 | **不合格** |
| `Swizzle<4,1,4>` | ✓ 2-way | ✗ 只有 2 | 不合格 |
| **`Swizzle<3,2,3>`** | ✓ 4-way | ✓ 4 | **合格** |
| `padding stride 33` | ✓ 1-way | ✓ 32 | 能过检查，但见下 |

填对后的输出：

```
转置结果 = 正确, 列读最坏 = 4-way, 行内最短连续 = 4 个 float
```

**为什么不选 `Swizzle<5,0,5>`**：它把冲突消得最干净，但 M=0 意味着行内一个连续对
都不剩。这道题的第二个检查就是在模拟真实约束——一旦你想用 128-bit atom 搬这块
smem，`Sw<5,0,5>` 直接编译不过。

**为什么 padding 能过检查却不推荐**：数组是按 `32*33` 开的，padding 装得下，
两项检查也都能过。但在 SM90 上它过不了 WGMMA 的编译期检查（练习 5），
而且行首不再 128B 对齐。这道题的检查只覆盖 bank 和连续性两项，
所以 padding 能"通过"——这恰好说明**自动检查不等于工程上正确**。

> README §3 §4，SM90 约束见 §5.3

---

## 关于 capstone 的测量（供对照）

8192×8192 float 转置，sm_90a，本机 HBM ~4.9 TB/s。这组数字随机器和邻居负载
波动，看相对关系而不是绝对值：

| 版本 | 带宽 (GB/s) | 相对 plain | SM90 可用性 |
|---|---|---|---|
| v1 naive（不过 smem） | ~547 | — | — |
| v2 plain smem（32-way） | ~1238 | 1.00x | — |
| v3 padded（stride 33） | ~2903 | **2.34x** | ✗ WGMMA 拒绝 |
| v4 `Swizzle<5,0,5>` | ~2718 | **2.20x** | 冲突最少但不能向量化 |
| v5 `Swizzle<3,2,3>` | ~2697 | **2.18x** | ✓ 和 GMMA 官方原子同路线 |

三点值得注意：

1. **消掉 32-way conflict 带来 2.2x 以上提升**，和理论预测一致。
2. **v3/v4/v5 三者性能接近**（2.34x / 2.20x / 2.18x）：冲突已经不是瓶颈了，
   剩下的差别来自偏移计算的复杂度。所以选哪个不该看这几个百分点，
   而要看**能不能用**——这就是第 3 点。
3. **padding 最快但在 SM90 上不能用。** 这是本章最重要的工程结论：
   性能排序和可用性排序不是一回事。
