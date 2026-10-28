# MLA（Multi-Head Latent Attention）


# MLA（Multi-Head Latent Attention）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

MLA 是 DeepSeek 提出的 attention 变体：把 K / V 压缩到低维 latent space 再存进 KV cache，推理时再展开。比 GQA 在相同 cache 大小下表现更好。

## 背景：GQA 是什么

[[MHA]] 每个 head 都有独立的 K / V；[[GQA]] 把多个 head 共享一组 K / V（group 内共享）：

- 降低 KV cache 内存
- 推理时少算 K / V → 快
- 性能几乎不掉（消融证明）

## MLA 的不同思路

不共享 K / V head，而是把 K / V 张量**压缩到低维**再存：

- 训练时 Q 也压缩（推理时不压缩，避免额外计算）
- 推理时压缩的 K / V 要展开才能用 → 多一次 matmul
- 优点：内存更省，且 DeepSeek-V2 论文消融显示**建模性能优于 MHA**

## 适用模型

- DeepSeek V2（首创）
- DeepSeek V3 / R1 / V3.1
- Kimi K2（K2 是 V3 架构的放大版）
- Mistral 3 Large（与 V3 几乎一样）
- GLM-5（2026 年新采用）

## 对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154849.png)

*Figure 2: A comparison between MHA and GQA. Here, the group size is 2, where a key and value pair is shared among 2 queries.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154850.png)

*Figure 3: Comparison between MLA (used in DeepSeek V3 and R1) and regular MHA.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154851.png)

*Figure 4: Annotated tables from the DeepSeek-V2 paper showing that MLA outperforms GQA.*

## Kimi Linear 用 MLA + Gate

[Kimi Linear](Kimi%20系列.md) 在 global attention 层把 Gated GQA 换成 **MLA + gate**——是 MLA 唯一一次被用在 Gated DeltaNet 混合架构中。

## 延伸阅读

- 本仓库已有更深入的 MLA 笔记：[[MLA]]
- [DeepSeek-V2 论文](https://arxiv.org/abs/2405.04434)
- [Build a Large Language Model (From Scratch)](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167) — 有 GQA + KV-cache 实现

## 来源

> *The Big LLM Architecture Comparison*, section 1, 8, 14, 16, 17, 21
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

