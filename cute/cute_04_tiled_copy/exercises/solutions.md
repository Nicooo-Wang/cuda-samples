# cute_04 练习参考解答

先自己做，卡住了再看。每题都标注了对应的 README 小节。

---

## 练习 1 — 数 bank ★☆☆

**答案：32**

```cpp
constexpr int EX1_CONFLICT = 32;
```

第 5 列的偏移是 `s(l, 5)` = `l * 32 + 5`（row-major，行 stride = 32）。

```
l=0: 偏移 5,   bank = (5/1) % 32 = 5
l=1: 偏移 37,  bank = (37) % 32 = 5
l=2: 偏移 69,  bank = (69) % 32 = 5
...
```

每行偏移加 32，bank 加 0（因为 32 mod 32 = 0）。32 个 lane 全撞 bank 5，**32-way conflict**。

这不是第 5 列特有的问题。任何列的情形都一样：第 c 列的 lane l 偏移是 `l*32 + c`，bank 是 `(l*32 + c) % 32 = c % 32`，全部相同。

> README §1

---

## 练习 2 — 为 half 选 padding ★★☆

**答案：2**

```cpp
constexpr int EX2_PAD_HALVES = 2;
```

推导过程：

half 是 2 字节，一个 bank 装 4 字节 = **2 个 half**。所以 half 下的"bank 编号"是 `(half_index / 2) % 32`。

plain 情形：tile 是 `(32, 64)` half，行 stride = 64。第 0 列的 lane l 偏移（以 half 为单位）是 `l * 64 + 0`，对应字节偏移 `l * 128`，bank = `(l * 128 / 4) % 32 = (l * 32) % 32 = 0`。32-way conflict，和 float 的 plain 一模一样。

消掉冲突需要让相邻行的同列偏移落在不同 bank。"错开一个 bank"需要让行 stride 增加 **2 个 half**（每个 bank 装 2 个 half）：

```
stride = 64 + 2 = 66
lane l 的偏移 = l * 66, 字节偏移 = l * 132
bank = (l * 132 / 4) % 32 = (l * 33) % 32

l=0: bank 0
l=1: bank 1
l=2: bank 2
...
```

`(l*33) % 32 = l % 32`，32 个 lane 落在 32 个不同 bank，无冲突。

加 1 个 half（stride 65）时：`(l*65) % 32 = l % 32`……等等，65 是奇数，`65 % 32 = 1`，所以 `l=0` bank=0，`l=1` bank=1，…实际上也没有冲突。

运行 ex2 时，EX2_PAD_HALVES 有两个合法值：**1 或 2**（stride 65 或 66），题目要求"最小的 padding"，所以答案是 **2**……要等等：实际上 stride = 65 时 bank 计算：

```
l=0: (0*65*2/4)%32 = 0
l=1: (1*65*2/4)%32 = (130/4)%32 ... 130/4 = 32.5 -> 取整 32 -> 32%32=0  还是撞!
```

注意 half 是 2 字节，不是按 half 个数而是按字节算 bank：`(half_index * 2) / 4 % 32 = (half_index / 2) % 32`。stride 65（奇数）时，lane l 的字节偏移是 `l * 65 * 2 = l * 130`，bank = `(l * 130 / 4) % 32`。`130 / 4 = 32.5`，截断成 32，所以每个 lane 的 bank 差是 `32 % 32 = 0`。32-way conflict！

stride 66 时，`l * 66 * 2 = l * 132`，`132 / 4 = 33`，bank 差是 `33 % 32 = 1`，各 lane bank 不同，无冲突。所以最小 padding 是 **2 个 half（即 stride 66）**。

> README §2

---

## 练习 3 — Swizzle 的三个不变量 ★★☆

```cpp
constexpr bool EX3_A = false;   // cosize 不变
constexpr bool EX3_B = true;    // 行方向仍连续
constexpr bool EX3_C = true;    // 仍是双射
```

逐一分析：

**A：cosize 变大了？假。** Swizzle 只改偏移映射，不改 shape，也不在末尾追加元素。plain 的 cosize = 1024，swizzled 的 cosize = 1024，一个字节都没多用。这是 swizzle 相对 padding 最大的优势。

**B：行方向仍然连续？真。** `Swizzle<5,0,5>` 的比特异或作用在行之间（row bits XOR with row bits），同一行内的列偏移不受影响。`swz(0, 0..7)` = 0, 1, 2, 3, 4, 5, 6, 7，和 plain 完全一样。行方向连续是合并访存的前提，swizzle 保留了它。

**C：仍是双射？真。** XOR 是可逆运算，swizzle 是 32×32 偏移集合上的置换（排列）。每个逻辑坐标映射到唯一的偏移，没有两个坐标共享同一个偏移，也没有偏移被跳过。实际运行时 v0 的代码扫过所有 1024 个坐标验证没有重复：结果是"双射：是"。

> README §3

---

## 练习 4 — 选对 GMMA swizzle 原子 ★★☆

**答案：`EX4_MASK = 0b1110 = 14`**

```cpp
constexpr int EX4_MASK = 14;
```

BK = 32，四个原子的 K 长度：

| 原子 | K 长度 | `BK % K_len` | 可用？ |
|---|---|---|---|
| SW128 | 64 | `32 % 64 = 32 ≠ 0` | 不可用 |
| SW64 | 32 | `32 % 32 = 0` | 可用 |
| SW32 | 16 | `32 % 16 = 0` | 可用 |
| INTER | 8 | `32 % 8 = 0` | 可用 |

只有 SW128 不行（BK 比它的 K 维还小），bit0 = 0，其余 = 1，mask = `0b1110 = 14`。

实用规则：**选能用的里面 K 维最长的**（最大对齐 = 最少的 bank 冲突 = 最好的 WGMMA 性能）。这里选 SW64。

> README §4

---

## 练习 5 — ldmatrix 搬多少 ★★☆

**答案：256**

```cpp
constexpr int EX5_TOTAL = 256;
```

```
总量 = ThrID × NumValSrc = 32 × 8 = 256 个 half
```

这 256 个 half = 512 字节 = 4 个 8×8 half 矩阵（每个 64 个 half = 128 字节）。名字里的 `U32x4` 就是 4 个 32-bit（每个 32-bit 装 2 个 half）× 4 = 一次操作发 4 条"32-bit × 32线程"的指令。

> README §5

---

## 练习 6 — 修一个 smem layout bug ★★★

**改法一：padding（stride 33）**

```cpp
auto slay = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
```

**改法二：swizzle（推荐）**

```cpp
auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
auto slay  = composition(Swizzle<5, 0, 5>{}, plain);
```

两种改法都能通过"转置结果正确"和"列读无 bank conflict"两项检查。

**两种改法各自的代价：**

padding：数组已按 `32 * 33` 开好了（`raw[32 * 33]`），所以装得下。stride 33 让每行的起始地址偏移 33 个 float，不再是 128B 对齐，宽向量指令（`LDG.E.128`）用不了。这里是标量读写所以不影响正确性，但这道题的目的是让你感受一下"padding 能通，但有隐性代价"。

swizzle：同样装得下（cosize = 1024，而 `raw` 有 `32*33 = 1056` 个 float），行方向仍连续，无对齐问题，Hopper 上也适用。

**选哪个？** 如果你明天还要把这个 kernel 改成 Hopper 版本，选 swizzle。如果只是 Ampere 的临时实验，padding 也行。

> README §3 §8

---

## 关于 capstone 的测量（供对照）

实测 8192×8192 float 转置（sm_90，本机 HBM ~4.9 TB/s）：

| 版本 | 带宽 (GB/s) | 相对 plain |
|---|---|---|
| v1 naive（不过 smem） | 558.1 | — |
| v2 plain smem（32-way conflict） | 1518.9 | 1.00x |
| v3 padded（stride 33） | 3015.7 | **1.99x** |
| v4 Swizzle<5,0,5> | 2915.0 | **1.92x** |

消掉 32-way conflict 带来约 2× 提升，与理论预测完全吻合。v3（padding）和 v4（swizzle）性能接近，差别来自 v3 的偏移计算略简单。但 v3 在 Hopper 上不能用，所以 v4 是真正的选择。
