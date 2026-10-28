# CISPO


# CISPO: Clipped IS-weight Policy Optimization

## 背景

在 GRPO 的训练过程中，观察到一个显著问题：模型会出现性能下降，且不能有效促进 CoT 推理行为。通过一系列实验后发现，**问题主要出在 PPO/GRPO 损失函数中的裁剪（Clipping）操作**。

具体来说，与反思行为相关的 token 往往是推理路径中的"分叉点"，在 Base 模型中出现频率低且被赋予较低的概率。这些 token 往往具备较高的重要性采样比 $r_t$，因此在第一次 on-policy 更新后就会被裁剪掉，无法参与后续的 off-policy 梯度更新。然而，**这些低概率的 token 恰恰对于稳定熵以及促进可扩展的强化学习至关重要**。

尽管 DAPO 尝试通过提高裁剪上界来缓解这一问题，但效果有限——每个生成批次包含 16 轮 off-policy 更新，简单的上界调整无法从根本上解决问题。

## CISPO 核心思想

CISPO（Clipped IS-weight Policy Optimization）的设计目标非常明确：

> 明确避免丢弃 token，即使是那些对应较大更新的 token，同时通过内在机制将熵控制在合理范围内，以确保探索过程的稳定性。

## 从 REINFORCE 出发

CISPO 从标准 REINFORCE 出发，其损失函数为：

$$
J_{\text{REINFORCE}}(\theta) = \mathbb{E}_{(q,a)\sim D, o_i \sim \pi_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\text{sg}(r_{i,t}(\theta))\hat{A}_{i,t}\log\pi_\theta(o_{i,t}|q,o_{i,<t})\right]
$$

其中 $\text{sg}$ 表示停止梯度操作。

CISPO 对重要性采样权重进行裁剪：

$$
\hat{r}_{i,t}(\theta) = \text{clip}\left(r_{i,t}(\theta), 1-\epsilon^{\text{IS}}_{\text{low}}, 1+\epsilon^{\text{IS}}_{\text{high}}\right)
$$

代入得到 CISPO 的目标函数：

$$
J_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a)\sim D, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\text{sg}(\hat{r}_{i,t}(\theta))\hat{A}_{i,t}\log\pi_\theta(o_{i,t}|q,o_{i,<t})\right]
$$

## 关键创新：stop-gradient

这里的核心是 **sg 操作**——让重要性采样比停止梯度。这么做的好处是：

- IS 权重只作为一种**固定的重新加权因子**，衡量采样分布和目标分布（新旧策略）之间的差异
- **不会像 PPO 那样因 clip 导致梯度被截断**，进而导致 token 更新被强行剪掉
- 即使某些 token 的 $r_t$ 很大，也只是被重新加权，而不是完全丢弃

### 与 PPO clip 的本质区别

PPO 的裁剪是直接对梯度进行裁剪，一旦超出裁剪区间，该位置的梯度直接变为零。而 CISPO 的裁剪是对 IS 权重进行裁剪并断开梯度，IS 权重本身不再推动参数更新，但它作为固定系数参与损失计算，不存在"梯度被截断"的问题。

## 通用公式与 Token Mask

文中还提出了一个通用公式，引入 token-level 的 mask：

$$
J_{\text{unify}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_i|o_i|}\sum_i\sum_t\text{sg}(\hat{r}_{i,t})\hat{A}_{i,t}\log\pi_\theta(o_{i,t})M_{i,t}\right]
$$

其中：

$$
M_{i,t} = \begin{cases}
0 & \text{if }\hat{A}_{i,t} > 0 \text{ and } r_{i,t}(\theta) > 1 + \epsilon_{\text{high}} \\
0 & \text{if }\hat{A}_{i,t} < 0 \text{ and } r_{i,t}(\theta) < 1 - \epsilon_{\text{low}} \\
1 & \text{otherwise}
\end{cases}
$$

简单来说，就是让过分放大奖励和过分惩罚的 token 损失为 0，完全不参与梯度更新——这在一定程度上等价于 PPO 带梯度的 clip。

## 熵增强

CISPO 还采用了**基于熵的项来增强优势函数**。研究发现：

> 高熵区域与三类探索性推理行为存在正相关性：
> 1. 决定或连接逻辑步骤的关键 Token
> 2. 自我验证和修正等反思性行为
> 3. Base 模型未充分探索的罕见行为

因此，**熵可以作为识别语言模型中探索性推理行为的有用信号**。

具体做法是使用裁剪并断开梯度的熵项来增强优势函数。与传统的最大熵方法（通过促进不确定性来鼓励探索）不同，CISPO 通过促进更长、更深的推理链来鼓励探索。

- 裁剪操作确保熵项既不会主导优势项，也不会改变其符号
- 梯度分离则保持了原始的优化方向

## 其他技术细节

CISPO 还借鉴了 DAPO 的两项技术：

1. **动态采样（Dynamic Sampling）**：过滤掉优势为零的样本
2. **长度惩罚（Length Penalty）**：缓解长度偏差问题

同时，CISPO **没有使用 KL 惩罚项**。

## 总结

| 特性 | CISPO | GRPO | PPO |
|------|-------|------|-----|
| Clip 位置 | IS 权重（断开梯度） | IS 权重 | 梯度直接裁剪 |
| 探索性 Token | 保留 | 可能被裁剪 | 可能被裁剪 |
| KL 惩罚 | 无 | 可选 | 有 |
| 熵增强 | 有 | 无 | 无 |
| 序列级/Token级 | Token级 | Token级 | Token级 |

## 参考

- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- [GRPO优化在继续——CISPO和熵 | 长琴](https://yam.gift/2025/06/19/NLP/LLM-Training/2025-06-19-CISPO-and-Entropy/)

