# Arcee AI Trinity Large


# Arcee AI Trinity Large

> 原文第 20 节。Arcee AI 在 2026-01-27 发布的 400B MoE（13B 激活）。技术报告相当详细。

## 关键设计

### 3:1 SWA（介于 Gemma 3 与 Olmo 3 之间）

- local:global = 3:1（不像 Gemma 3 / Xiaomi 是 5:1，也不像纯 global）
- 滑动窗口大小 = 4096（与 Olmo 3 类似）

### [[NoPE（No Positional Embedding）|NoPE]] in global layers

- global attention 层用 NoPE（参考 [[SmolLM3]] 的做法）
- 局部层照常用 RoPE

### Gated Attention

- 不像 Qwen3-Next 是完整 GatedDeltaNet，而是借鉴了 Qwen3-Next 的 attention 门控
- 在 scaled dot-product **后、output linear projection 前** 加 elementwise gating
- 效果：减少 attention sinks，改善长序列泛化 + 训练稳定性

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154946.png)

*Figure 55: Illustration of the gating mechanism that Trinity Large uses in the attention mechanism.*

### Depth-Scaled Sandwich Norm

之前没见过这种设计。每个 block 有 **4 个 RMSNorm**（不像 Gemma 3 的 2 个）：

- 第二个 RMSNorm 的 gain 被初始化为约 1/√L（L = 总层数）
- 训练早期残差更新小，随训练学到合适 scale 长大
- 看起来像 Gemma 3 的 sandwich norm，但加了"深度缩放"

### [[MoE（Mixture of Experts）|MoE]]

- DeepSeek-like：大量小专家
- 但比 DeepSeek 略粗（coarser）—— 利于推理吞吐（Mistral 3 Large 也这么干）

### 其他训练改进

- 新的 MoE load-balancing
- MuOpt optimizer（与 Kimi K2 用 Muon 是不同方案）

## 模型族

- Trinity Large 400B-A13B（旗舰）
- Trinity Mini 26B-A3B
- Trinity Nano 6B-A1B

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154944.png)

*Figure 53: Overview of the Trinity Large architecture.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154945.png)

*Figure 54: Arcee AI Trinity Large next to GLM 4.5 of a relatively similar size (400B vs 355B).*

## 相关笔记

- [[滑动窗口注意力（Sliding Window Attention）]]
- [[NoPE（No Positional Embedding）]]
- [[MoE（Mixture of Experts）]]
- [[QK-Norm]] —— Trinity 也用了 QK-Norm
- [[SmolLM3]] —— NoPE
- [[GLM 系列]] GLM-4.5 —— 大小相近
- [[Gemma 系列]] —— 同样 sandwich norm

## 来源

> 原文 section 20
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

