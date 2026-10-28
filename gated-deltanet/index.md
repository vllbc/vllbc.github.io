# Gated DeltaNet


# Gated DeltaNet

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。一个**线性 attention 变体**，是 Qwen3-Next 的核心组件。

## 一句话

Gated DeltaNet 是 **DeltaNet + Mamba 风格 gating** 的组合：

- DeltaNet：用 q / k / v / α / β 维护一个**fast-weight memory**，按 delta rule 更新（线性 O(n) 缩放）
- Mamba gating：额外用门控机制控制 memory 写入/读取

→ 本质是"线性 attention 的现代变种"，常被拿来做 3:1 混合中的"线性那 3 层"。

## DeltaNet 是什么

DeltaNet 起源：用一个 fast-weight memory（类似 RNN 的 hidden state）来记录历史信息。Q 读取，α / β 控制更新强度。

## 为什么不直接用 linear attention

- 早期 linear attention（2020 Transformers are RNNs）总掉精度，所以没在 SOTA 模型中落地
- DeltaNet 通过 delta rule update，比"加性"linear attention 更精细
- Gated DeltaNet 加上 Mamba 的 gating，更稳定

## 与 Mamba-2 的对比

> "Mamba keeps a state with a learned state-space filter (essentially a dynamic convolution over time). DeltaNet keeps a tiny fast-weight memory updated with α and β and reads it with q, with small convolutions only used only to help form q, k, v, α, β."

两者都属于"linear-time, cache-free"族，本质都是 state-space 风格的 update，但具体数学结构不同。

## 谁在用

- **Qwen3-Next**：3 层 Gated DeltaNet + 1 层 Gated Attention 3:1
- **Kimi Linear**：进一步改进为 [[KDA（Kimi Delta Attention）|KDA]]（channel-wise gate）
- **Nemotron 3**：用 Mamba-2（DeltaNet 的近亲）

## 图示

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154924.png)

*Figure 36: The Gated DeltaNet + Gated Attention hybrid mechanism. Note that these are arranged in a 3:1 ratio.*

## 相关文章

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06458)

## 相关笔记

- [[KDA（Kimi Delta Attention）]]
- [[Qwen3 系列]] Qwen3-Next
- [[Kimi 系列]] Kimi Linear
- [[Nemotron 3]]

## 来源

> *The Big LLM Architecture Comparison*, section 12.2, 14
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

