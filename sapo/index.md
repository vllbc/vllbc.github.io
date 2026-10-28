# SAPO


# SAPO: Soft Adaptive Policy Optimization

## 背景

在 LLM 的 RL 训练中，**稳定的策略优化仍然具有挑战性**。Token-level 的重要性采样比往往表现出高方差——这一问题在 MoE（Mixture-of-Experts）模型中尤为严重，会导致不稳定的更新。

现有的基于组的策略优化方法（如 GSPO 和 GRPO）通过**硬裁剪（hard clipping）**来缓解高方差问题，但这使得稳定性和有效学习之间难以兼顾：

- **GSPO**：在序列级别进行硬裁剪，一旦某个序列中有少数 token 严重偏离策略，整个序列的梯度都会被抑制
- **GRPO**：在 token 级别进行硬裁剪，虽然 token 级别更细粒度，但裁剪边界过于"脆弱"

## 什么是 SAPO

SAPO（Soft Adaptive Policy Optimization）提出用**平滑的、温度控制的门控机制**替代硬裁剪，自适应地衰减 off-policy 更新的同时保留有用的学习信号。

> SAPO replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals.

## 核心思想

### 硬裁剪的问题

以 GSPO 为例，其硬裁剪逻辑是：

$$
r_t = 	ext{clip}\left(rac{\pi_	heta(a_t|s_t)}{\pi_{	heta_{	ext{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon
ight)
$$

当一个序列中某个 token 严重 off-policy 时（$r_t$ 超出裁剪区间），GSPO 会抑制**整个序列**的所有梯度。这固然稳定，但也**浪费了那些 near-on-policy token 的有效学习信号**。

### SAPO 的解决方案

SAPO 的核心是用一个**软门控函数**替代硬裁剪：

$$
g(r_t; 	au) = rac{r_t}{	au + r_t}
$$

或者更一般地，使用 sigmoid 类型的平滑过渡：

$$
g(r_t; 	au) = 	ext{sigmoid}\left(rac{\log r_t}{	au}
ight)
$$

其中 $	au$ 是温度参数，控制平滑程度。

- 当 $	au 	o 0$ 时，$g(r_t) 	o 1$（无裁剪）
- 当 $	au 	o \infty$ 时，$g(r_t) 	o 0$（完全抑制）
- 中间状态提供一个**连续的信任域**，而不是 GSPO 那种"断裂"的硬裁剪带

### SAPO vs GSPO vs GRPO

| 特性 | SAPO | GSPO | GRPO |
|------|------|------|------|
| 裁剪方式 | 平滑软门控 | 序列级硬裁剪 | Token级硬裁剪 |
| 序列连贯性 | 保持 | 保持 | Token独立 |
| Token自适应 | 是 | 否（全序列抑制） | 否（独立裁剪） |
| 信任域 | 连续 | 断裂带 | 断裂带 |
| 训练稳定性 | 高 | 高 | 中 |
| 样本效率 | 高（保留有效信号） | 中 | 中 |

## 关键特性

### 1. 序列连贯性（Sequence Coherent）

与 GRPO 不同，SAPO 保持了序列级别的连贯性。GSPO 虽然也是序列级别，但使用的是硬裁剪。SAPO 通过软门控，在保持序列级语义的同时，对严重 off-policy 的 token 进行选择性降权，而不是全序列抑制。

### 2. Token 自适应（Token-Adaptive）

当一个序列包含少数严重 off-policy 的 token 时：

- **GSPO**：抑制整个序列的梯度
- **SAPO**：只对偏离的 token 进行降权，保留 near-on-policy token 的学习信号

### 3. 温度控制（Temperature-Controlled）

温度参数 $	au$ 控制裁剪的"软硬"程度：

- 较小的 $	au$：更接近无裁剪，保留更多学习信号，但可能不稳定
- 较大的 $	au$：更 aggressive 地抑制 off-policy，更稳定但可能欠学习

这提供了一个**连续可调**的稳定性和效率的权衡空间。

## 与 CISPO 的关系

值得注意的是，CISPO 和 SAPO 都指向了同一个问题：**GRPO/PPO 的硬裁剪会损害探索性 token 的学习**。但它们的解决思路不同：

- **CISPO**（MiniMax）：对 IS 权重裁剪并断开梯度，用 token mask 替代梯度裁剪，同时用熵增强探索
- **SAPO**（Qwen）：用平滑的软门控替代硬裁剪，保持序列连贯性的同时 token 自适应地衰减

两者都是对 GRPO 硬裁剪问题的有力改进。

## 实验结果

在数学推理基准测试中，SAPO 表现出：

- **更好的训练稳定性**
- **更高的 Pass@1 性能**（在相同训练预算下）

此外，SAPO 被用于训练 Qwen3-VL 模型系列，在不同任务和不同模型规模上都取得了一致的性能提升。

## 总结

SAPO 的核心贡献在于：

1. **用软门控替代硬裁剪**，提供一个连续而非断裂的信任域
2. **Token 自适应降权**，只抑制严重 off-policy 的 token，保留 near-on-policy token 的有效学习信号
3. **温度控制的平滑过渡**，提供连续可调的稳定性和效率权衡
4. **保持序列连贯性**，避免 GSPO 全序列抑制的样本效率损失

SAPO 为 RL 训练提供了一个更可靠、更可扩展、更有效的优化策略。

## 参考

- [Qwen3 Technical Report]（RL 优化部分）
- [LLM 强化的"炼金术"：主流开源模型的 RL 优化策略赏析 | 长琴](https://yam.gift/2026/01/14/NLP/LLM-Training/2026-01-14-Open-LLM-RL-ShowCase/)

