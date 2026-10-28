# Partial RoPE


# Partial RoPE

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

Partial RoPE = 只对每个 head 的部分通道应用 RoPE 旋转，剩下通道不变。

```
Full RoPE:     [r r r r r r r r]
Partial RoPE:  [r r r r — — — —]
```

- 每个 head 的 rotary_dim 通道做旋转
- 剩余 (head_dim - rotary_dim) 通道保持不变

## 起源与目的

> M1 README 原文："Rotary Position Embedding (RoPE) applied to half of the attention head dimension with a base frequency of 10,000,000"

> M1 论文："implementing RoPE on half of the softmax attention dimensions enables length extrapolation without performance degradation."

作者猜测的 rationale：长序列（特别是比训练数据最长还长的）下，"完全不旋转" 比 "训练时没见过的剧烈旋转" 更安全。

## 谁在用

- **MiniMax M1**（首创）
- **MiniMax M2**
- **Gemma 4**：p-RoPE，只对 25% 的频率对加位置信息（思想类似）

## 与 [[NoPE（No Positional Embedding）|NoPE]] 的对比

- NoPE：完全不加位置信息
- Partial RoPE：只对一部分通道加
- 两者都旨在"减少位置信息强度"，让长上下文更稳

## 相关笔记

- [[NoPE（No Positional Embedding）]]
- [[MiniMax M2]]
- [[Gemma 系列]] Gemma 4
- [[Qwen3 系列]] Qwen3-Next —— 也用 partial RoPE

## 来源

> *The Big LLM Architecture Comparison*, section 13.3
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

