# QK-Norm


# QK-Norm

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

QK-Norm = 在 attention 模块内，对 **q、k** 各自加一层 RMSNorm，**在 RoPE 之前** 应用。起源是 2023 年的 *Scaling Vision Transformers* 论文。

## 代码示例

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups,
                 head_dim=None, qk_norm=False, dtype=None):
        # ...
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # ...
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        # apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        # ...
```

## 作用

- 与 [[Post-Norm vs Pre-Norm|Post-Norm]] 一起 → 稳定训练
- QK 的尺度被显式控制 → 防止 attention logits 爆炸

## 谁在用

- OLMo 2（首次出现在主流 LLM）
- Gemma 2 / Gemma 3
- MiniMax M2（**Per-Layer QK-Norm**——见下）
- Arcee Trinity Large
- GLM-4.5

## Per-Layer QK-Norm（M2 变体）

普通 QK-Norm 是一组 `q_norm` / `k_norm` 在每层共享。M2 的实现更"细"：

```python
self.q_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_heads, eps=...)
self.k_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_kv_heads, eps=...)
```

- hidden_size = num_heads × head_dim
- scale 向量对**每个 head 的每个 head_dim 都有独立参数**——不是 head 间共享

## 相关笔记

- [[Post-Norm vs Pre-Norm]]
- [[OLMo 2]]
- [[Gemma 系列]]
- [[MiniMax M2]]
- [[Arcee Trinity Large]]

## 来源

> *The Big LLM Architecture Comparison*, section 2.2
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

