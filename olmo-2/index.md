# OLMo 2 & Olmo 3


# OLMo 2 & Olmo 3 Thinking

> 原文第 2、15 节。OLMo 系列是 Allen Institute for AI 发布的"全透明"开源模型——训练数据 / 训练代码 / 训练报告全公开。在 benchmark 上不算最顶尖，但是 LLM 复现的"模板"。

## OLMo 2 核心：Post-Norm + QK-Norm

OLMo 2 整体沿用 GPT 骨架（仍是 MHA，不用 GQA / MLA），但有两点与主流不同的设计：

### 2.1 [[Post-Norm vs Pre-Norm|Post-Norm 风格]]

OLMo 2 的 RMSNorm 放在 attention / FFN **之后**（Post-Norm），而不是主流的 Pre-Norm 位置。Norm 仍然在残差连接内部。

- 原始 Transformer（2017）：Post-LN（norm 在残差外）
- GPT-2 之后大多数 LLM：Pre-Norm（norm 在 attention / FFN 前）
- OLMo 2：Post-Norm（norm 在 attention / FFN 后，仍在残差内）

这种放置是 OLMo 2 团队在实验中发现的稳定训练的手段。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154855.png)

*Figure 8: A comparison of Post-Norm, Pre-Norm, and OLMo 2's flavor of Post-Norm.*

### 2.2 [[QK-Norm]]

QK-Norm = 另一个 RMSNorm 层，放在 attention 模块内，对 q、k 应用 RoPE **之前**做归一化。

- 起源：2023 年的 Scaling Vision Transformers 论文
- 效果：与 Post-Norm 一起，显著稳定训练
- 后续广泛采用：Gemma 2/3、MiniMax M2 等

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, ...):
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        ...

    def forward(self, x, mask, cos, sin):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        if self.q_norm: queries = self.q_norm(queries)
        if self.k_norm: keys = self.k_norm(keys)
        # apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        ...
```

### 2.3 小结

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154856.png)

*Figure 9: A plot showing the training stability for Pre-Norm vs OLMo 2's flavor of Post-Norm.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154857.png)

*Figure 10: An architecture comparison between Llama 3 and OLMo 2.*

> 注意：Figure 9 同时混入了 QK-Norm 的影响，所以 Post-Norm 单独贡献多少并不容易看出。

3 个月后发布的 32B OLMo 2 改用了 GQA。

## Olmo 3 Thinking（2025-11-20 发布）

Allen AI 在 2025 年 11 月发布了 Olmo 3 7B 和 32B（注意官方拼写改了，从 OLMo → Olmo），并加了 reasoning 变体。

### 关键设计

- **继承 Post-Norm**（沿用 OLMo 2 的经验，验证稳定性）
- **7B 仍用 MHA**（与 OLMo 2 一致），但加了**滑动窗口注意力**来缩 KV cache（类似 Gemma 3）
- **32B 改用 GQA**（与 Llama 3 / Qwen3 一致）
- **YaRN** 用于 context extension 到 64k（Qwen3 是可选 32k → 131k）
- 32B 架构缩放与 Qwen3 32B 比例相近（intermediate 从 5x 提到 5.4x）

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154932.png)

*Figure 44: Olmo 3 7B and Qwen3 8B side by side.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154933.png)

*Figure 45: Olmo 3 32B and Qwen3 32B side by side.*

### 训练与代码

Olmo 3 提供了 from-scratch 实现 notebook，是学习 LLM 架构的好资源。

## 相关笔记

- [[QK-Norm]]
- [[Post-Norm vs Pre-Norm]]
- [[滑动窗口注意力（Sliding Window Attention）]]
- [[Qwen3 系列]]

## 来源

> 原文 section 2 & 15
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

