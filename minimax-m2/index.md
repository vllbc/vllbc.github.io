# MiniMax M2


# MiniMax M2

> 原文第 13 节。MiniMax（MiniMax）在 M1 之后发布的旗舰。benchmark 表现目前被认为是开源 SOTA。

## 关键背景

- M1（2025-06）用了 lightning attention（线性 attention 变体），456B 总参 / 46B 激活
- M2（2025-12 之后）**回退到 full attention**——团队公开说 linear attention 在生产 LLM 里 tricky，普通 prompt 看起来 OK，但 reasoning / multi-turn / agent 掉点严重

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154925.png)

*Figure 37: MiniMax-M2 benchmark performance compared to other popular open-weight and proprietary LLMs.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154926.png)

*Figure 38: A timeline of the main LLMs covered in this article, next to some of the attention-hybrid models.*

## 总体架构

> MiniMax-M2 与 Qwen3 惊人相似——只有层数 / 大小等数字差异，组件几乎一样。

## 13.1 Per-Layer QK-Norm

QK-Norm 一般是 per-head 共用（MHA / GQA 都一样）。MiniMax-M2 的"per_layer"实现：

```python
self.q_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_heads, eps=...)
self.k_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_kv_heads, eps=...)
```

- hidden_size = num_heads * head_dim
- RMSNorm 的 scale 向量对 **每个 head 的每个 head_dim 都有独立参数**

→ "per_layer" 意思是：每个 transformer block 各有自己的 QK-Norm（普通 QK-Norm 也这样），并且**不是 head 间共享，而是每个 head 独立一套**。

**注**：config 里有 sliding window 设置，但默认关闭（与 Mistral Small 3.1 类似）。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154927.png)

*Figure 39: Comparison between Qwen3 and MiniMax-M2.*

## 13.2 MoE Sparsity

- **没有 shared expert**（与 Qwen3 一致，与 Qwen3-Next 不一样）
- **sparsity 极高**：4.37% 激活 vs Qwen3 235B-A22B 的 9.36%
- 同样大小下，激活参数只有 10B（Qwen3 是 22B）

## 13.3 [[Partial RoPE]]

类似 M1，M2 在 attention 里用 **partial RoPE**：

- 每个 head 的 rotary_dim 通道才做旋转
- 剩下 head_dim - rotary_dim 通道不变

```
Full RoPE:     [r r r r r r r r]
Partial RoPE:  [r r r r — — — —]
```

> M1 README 原文："Rotary Position Embedding (RoPE) applied to half of the attention head dimension with a base frequency of 10,000,000"

### 目的

> "implementing RoPE on half of the softmax attention dimensions enables length extrapolation without performance degradation."

作者猜测：长序列（特别是超过训练数据最长长度的）下，"完全不旋转" 比 "训练时没见过的剧烈旋转" 更安全。

## 相关笔记

- [[QK-Norm]]（普通版 vs Per-Layer 版）
- [[Partial RoPE]]
- [[MoE（Mixture of Experts）]]
- [[Qwen3 系列]] —— 同期对照
- [[Kimi 系列]] Kimi Linear —— MiniMax 是先 linear attention 后回退，Kimi 是坚持 linear

## 来源

> 原文 section 13
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

