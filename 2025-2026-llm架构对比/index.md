# 2025-2026 LLM 架构对比


# 2025-2026 大型 LLM 架构对比

> 整理自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*（最后更新于 2026 年 4 月 2 日）。原文以架构为主线，扫描了 23+ 个 2025-2026 年发布的主流开源 LLM。本文按模型族拆成多份笔记，每篇都包含关键技术点、对应原图与索引。

## 总览

文章开头的一张图囊括了 2025-2026 年开源 LLM 的主要架构变化趋势：从 GPT-2 到 DeepSeek V3 / Llama 4，骨架仍惊人相似——结构性的"大变化"其实不多，更多是细节打磨。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154848.png)

*Figure 1: A subset of the architectures covered in this article.*

作者关注的核心问题是：哪些架构上的小修小补真的有用？例如：

- 位置编码：absolute → RoPE
- 多头注意力：MHA → GQA（甚至 MLA）
- 激活函数：GELU → SwiGLU
- MoE 在 2025 年大规模回潮

## 23 个模型 / 系列笔记

按原文顺序，每篇都是一个独立笔记：

| # | 模型 / 系列 | 笔记 | 关键创新点 |
|---|---|---|---|
| 1 | DeepSeek V3 / R1 | [[DeepSeek V3 系列]] | MLA、MoE + shared expert |
| 2 | OLMo 2 | [[OLMo 2]] | Post-Norm、QK-Norm |
| 3 | Gemma 3 / 3n | [[Gemma 系列]] | 滑动窗口注意力、Pre+Post-Norm、PLE |
| 4 | Mistral Small 3.1 | [[Mistral 系列]] | 放弃滑动窗口、推理优化 |
| 5 | Llama 4 | [[Llama 4]] | 交替 MoE + dense |
| 6 | Qwen3 (Dense / MoE) | [[Qwen3 系列]] | dense 与 MoE 并存 |
| 7 | SmolLM3 | [[SmolLM3]] | NoPE（每 4 层） |
| 8 | Kimi K2 / K2 Thinking | [[Kimi 系列]] | 1T 参数、Muon optimizer |
| 9 | GPT-OSS | [[GPT-OSS]] | 滑动窗口 + 注意力 bias + attention sinks |
| 10 | Grok 2.5 | [[Grok 2.5]] | 少量大专家、类共享专家 |
| 11 | GLM-4.5 | [[GLM 系列]] | 3 层 dense + MoE、共享专家 |
| 12 | Qwen3-Next / Coder-Next | [[Qwen3 系列]] | Gated DeltaNet + Gated Attention、MTP |
| 13 | MiniMax M2 | [[MiniMax M2]] | per-layer QK-Norm、Partial RoPE |
| 14 | Kimi Linear | [[Kimi 系列]] | KDA、channel-wise gating |
| 15 | Olmo 3 Thinking | [[OLMo 2#Olmo 3 Thinking]] | Post-Norm、YaRN、滑动窗口 |
| 16 | DeepSeek V3.2 | [[DeepSeek V3 系列#DeepSeek V3.2]] | Sparse Attention |
| 17 | Mistral 3 | [[Mistral 系列]] | 复用 DeepSeek V3 架构 |
| 18 | Nemotron 3 Nano / Super | [[Nemotron 3]] | Mamba-Transformer 混合、latent MoE |
| 19 | Xiaomi MiMo-V2-Flash | [[Xiaomi MiMo]] | 激进滑动窗口 (size=128) |
| 20 | Arcee AI Trinity Large | [[Arcee Trinity Large]] | 3:1 SWA、NoPE、深度缩放 sandwich norm |
| 21 | GLM-5 | [[GLM 系列]] | 744B、MLA + DSA |
| 22 | Feb 2026 速览 | [[Feb 2026 LLM 新发布]] | 10+ 个模型 |
| 23 | Gemma 4 | [[Gemma 系列#Gemma 4]] | V=K 共享、p-RoPE |

## 关键技术概念笔记

这些是跨模型反复出现的机制，独立成笔记方便查阅：

- [[MLA（Multi-Head Latent Attention）]] — DeepSeek 系的核心 KV 压缩方案
- [[MoE（Mixture of Experts）]] — 2025 年最热趋势
- [[QK-Norm]] — 稳定训练的常见手段（OLMo、Gemma 3、MiniMax M2 等）
- [[Post-Norm vs Pre-Norm]] — Normalization 位置选择
- [[滑动窗口注意力（Sliding Window Attention）]] — Gemma 3、GPT-OSS、Mistral 等
- [[NoPE（No Positional Embedding）]] — SmolLM3、Arcee Trinity 等
- [[Gated DeltaNet]] — 线性注意力变体，Qwen3-Next / Kimi Linear 用
- [[KDA（Kimi Delta Attention）]] — channel-wise gating 的 KDA
- [[MTP（Multi-Token Prediction）]] — 训练信号增强
- [[Partial RoPE]] — MiniMax 系的做法
- [[Attention Sinks]] — 长上下文稳定手段
- [[Latent Experts]] — Nemotron 3 Super 的 MoE 改造

## 线性注意力的兴起（贯穿全文）

2025 下半年有一个清晰的趋势：**用线性注意力替代 O(n²) 的标准注意力**，又保持一定的全局注意力来弥补精度损失。

- 2025/06：MiniMax M1（lightning attention）
- 2025/08：Qwen3-Next（Gated DeltaNet + Gated Attention 3:1）
- 2025/09：DeepSeek V3.2（DeepSeek Sparse Attention）
- 2025/10：Kimi Linear（KDA + MLA 3:1）
- 2025/12：Nemotron 3（Mamba-2 + MoE + 少量 attention）

但作者也指出：**MiniMax M2 反而放弃了 M1 的线性注意力**，回到 full attention——线性注意力在推理 / 多轮对话 / agentic 场景下表现欠佳。这让"线性注意力是否能落地"暂时成疑，而 Kimi Linear 算是给出了有竞争力的回应。

## 一些观察

1. **MoE 已经不再是少数派**：DeepSeek V3、Llama 4、Qwen3 MoE、Mistral 3、Kimi K2、MiniMax M2、Nemotron 3、Xiaomi MiMo、Arcee Trinity、GLM-5、Gemma 4 MoE……几乎所有旗舰都给了 MoE 版本。
2. **架构层面的"小改进"很密集**：QK-Norm、滑动窗口、NoPE、Partial RoPE、attention sinks、latent expert、shared expert、post-norm……这些都不是颠覆性创新，但叠加后能稳定拿到质量 / 效率收益。
3. **共享专家仍有争议**：Qwen3 放弃了共享专家（认为对推理优化不利），DeepSeek V3 保留，Qwen3-Next 又加回来——团队之间没有共识。
4. **滑动窗口的"激进"程度差异大**：Gemma 3 是 1024，Xiaomi MiMo 是 128，Arcee Trinity 是 4096。
5. **稀疏度差异巨大**：从 4.37%（MiniMax M2）到 9.36%（Qwen3 235B）到更高（Grok 2.5 / Nemotron 3）。

## 原文引用

> Last updated: Apr 2, 2026 (added Gemma 4 in section 23)

> "It has been seven years since the original GPT architecture was developed. ... beneath these minor refinements, have we truly seen groundbreaking changes, or are we simply polishing the same architectural foundations?"

来源：[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

