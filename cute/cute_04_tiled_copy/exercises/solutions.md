# cute_04 练习参考解答

先自己做，卡住了再看。每题都标了对应的 README 小节。

---

## 练习 1 — 数 bank ★☆☆（§4.1）

**答案：32**

```cpp
constexpr int EX1_CONFLICT = 32;
```

第 5 列的偏移是 `s(l, 5) = l * 32 + 5`（行 stride = 32）：

```
l=0:  偏移   5    bank = 5 % 32 = 5
l=1:  偏移  37    bank = 37 % 32 = 5
l=2:  偏移  69    bank = 69 % 32 = 5
...
l=31: 偏移 997    bank = 997 % 32 = 5
```

每行偏移 +32，而 `32 % 32 = 0`，bank 号纹丝不动。32 个 lane 全撞 bank 5 →
**32-way conflict**。第 c 列也一样：`(l*32 + c) % 32 = c % 32`，与 l 无关。

---

## 练习 2 — 手算 swizzle 映射 ★★☆（§4.3）

**答案：197**

```cpp
constexpr int EX2_SWZ_OFF = 197;
```

`Swizzle<5,0,5>`，`(r,c) = (6,3)`：

```
第 1 步  plain off = 6*32 + 3 = 195 = 0b011000011
                              +-r=6-++-c=3-+
第 2 步  取出高 5 位 (r)      = 0b00110 = 6
第 3 步  M=0, 不左移          = 6
第 4 步  c XOR r             = 0b00011 XOR 0b00110 = 0b00101 = 5
         拼回去               = 6*32 + 5 = 197
```

口诀：`c_new = c XOR r`。

---

## 练习 3 — 选 M 保住向量化 ★★★（§4.4）

**答案：只有 `Sw<3,2,3>` 能用。**

```cpp
constexpr bool EX3_505_OK = false;
constexpr bool EX3_323_OK = true;
```

| swizzle | 列读 | 行内连续 | 128-bit |
|---|---|---|---|
| `Sw<5,0,5>` | 1-way | 1 | ✗ |
| `Sw<4,1,4>` | 2-way | 2 | ✗ |
| `Sw<3,2,3>` | 4-way | 4 | ✓ |

冲突最少的（1-way）恰恰不能向量化 —— M 保护的是 2^M 个连续元素，128-bit 需要
4 个连续 float，所以 M 至少 2。**先保向量化，再谈消冲突。**

---

## 练习 4 — 选对 TMA 模式 ★★☆（§4.6）

**答案：四个全可用，选 SW128。**

```cpp
constexpr bool EX4_SW128_OK = true;   // 32 % 32 == 0
constexpr bool EX4_SW64_OK  = true;   // 32 % 16 == 0
constexpr bool EX4_SW32_OK  = true;   // 32 % 8  == 0
constexpr bool EX4_INTER_OK = true;   // 32 % 4  == 0
constexpr bool EX4_PICK_SW128 = true;
```

实用规则：选能用的里面一行字节数最大的（对齐越大访存越宽）。CN=32 float =
128 字节 → SW128。

---

## 练习 5 — TMA 语义 ★★☆（§2 §4.6）

**答案：(a) 对 (b) 错**

```cpp
constexpr bool EX5_A = true;   // 手写 = 逻辑层, TMA = 物理层
constexpr bool EX5_B = false;  // plain 交给 TMA 也搬得对
```

(a) 手写 swizzle 改的是 `s(r,c)` 的偏移（逻辑层）；TMA 的 swizzle 改的是 smem
**物理字节序**（硬件写 smem 时应用 XOR），逻辑坐标读取永远正确。
(b) v3 §4.6 实测：五种 layout（plain/SW128/SW64/SW32/INTER）TMA 全搬对。
"TMA 要求 swizzle"是误解 —— 它不要求。

---

## 练习 6 — 修一个转置 bug ★★★（§6）

**答案：**

```cpp
auto slay = tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{},
                          make_shape(Int<32>{}, Int<32>{}));
```

改一行之后：列读 32-way → 16-way，结果仍正确。要点：SW128 的 M=4 保住 16 个
float 连续（§4.4 规则），且 32 % 32 == 0 满足整除（§4.6）。

---

## 练习 7 — 手写一段 TMA 搬运 ★★★（§2）

五个 TODO 对应 §2.2 的五个新东西：

```cpp
// TODO 1: 这一次搬多少字节
constexpr int tx_bytes = 32 * 32 * sizeof(float);

// TODO 2: smem 128B 对齐 + mbarrier
__shared__ __align__(128) float smem[32 * 32];
__shared__ uint64_t bar;

// TODO 3: 坐标 tensor + 本 CTA 的 tile
auto gc = tma.get_tma_tensor(make_shape(Int<256>{}, Int<128>{}));
auto gt = local_tile(gc, Shape<Int<32>, Int<32>>{}, make_coord(blockIdx.x, blockIdx.y));

// TODO 4: barrier 初始化 + thread 0 发 TMA
if (threadIdx.x == 0) initialize_barrier(bar, 1);
__syncthreads();
if (threadIdx.x == 0) {
    set_barrier_transaction_bytes(bar, tx_bytes);
    auto per = tma.get_slice(0);
    copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
}

// TODO 5: 等硬件搬完 (只用一轮 -> phase 0)
__syncthreads();
wait_barrier(bar, 0);
```

---

## 练习 8 — TMA Double Buffer ★★★（§5.2）

三个 TODO：

```cpp
// TODO 1: phase 公式 —— buffer s 第 k 轮被用
int phase = (k / STAGES) & 1;

// TODO 2: 通知 producer 用完
arrive_barrier(empty[s]);

// TODO 3: 先等 empty 再补货 (这里也要 phase)
wait_barrier(empty[s], phase);
```

写错 phase 的症状：**死锁**（k=1 时等 full[1] 的 phase 0，但 prologue 已把它
翻到 1 —— 等一个永远不会到来的翻转）。prologue 填满 STAGES 个 buffer 后，
full/empty 都从 phase 0 开始，不要"预推进"。
