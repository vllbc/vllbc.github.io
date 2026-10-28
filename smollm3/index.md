# SmolLM3


# SmolLM3

> 原文第 7 节。HuggingFace 的 3B 模型，介于 Qwen3 1.7B 和 4B 之间。

## 总体定位

- 3B 参数 size：兼顾能力与本地可运行性
- 训练透明度：与 OLMo 类似，公开了大量训练细节
- 性能：在与 Qwen3 1.7B / 4B、Llama 3 3B、Gemma 3 4B 的对比中处于前列

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154907.png)

*Figure 20: Win rate of SmolLM3 vs Qwen3 1.7B/4B, Llama 3 3B, Gemma 3 4B.*

## 关键设计：每 4 层用 [[NoPE（No Positional Embedding）|NoPE]]

SmolLM3 整体架构"看起来"很标准。**唯一有意思的点是 NoPE（无位置编码）**——只在每 4 层中"用" NoPE（即省略 RoPE）。

### NoPE 是什么

传统的 LLM 必须注入位置信息，因为 self-attention 把 token 视为无序集合。两种主流方案：

1. **绝对位置编码**（早期 GPT）：加一层 embedding，把位置信息加到 token embedding 上
2. **RoPE**（当前主流）：旋转 Q / K 向量，编码相对位置

**NoPE = 什么位置信息都不加**。起源于 2023 年 *The Impact of Positional Encoding on Length Generalization in Transformers* 论文。

### 为什么 NoPE 也能工作？

- 因果 mask 天然编码了方向：位置 t 只能看到 ≤ t 的 token
- 模型通过 gradient descent 可以学到"位置"信息，如果对 loss 有利的话
- NoPE 论文的额外发现：**长度泛化能力更好**

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154908.png)

*Figure 21: A side-by-side architecture comparison between Qwen3 4B and SmolLM3 3B.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154909.png)

*Figure 22: Absolute positional embeddings (from the "Build a Large Language Model (From Scratch)" book).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154910.png)

*Figure 23: Annotated figure from the NoPE paper showing better length generalization with NoPE.*

> 论文实验用的是 ~100M 参数的小模型，小 context；SmolLM3 团队显然认为不能完全照搬，所以**只在每 4 层用 NoPE**，其余仍用 RoPE。

## 相关笔记

- [[NoPE（No Positional Embedding）]]
- [[Arcee Trinity Large]] —— 在 global 层用 NoPE
- [[Kimi 系列]] Kimi Linear —— 在 MLA 层用 NoPE

## 来源

> 原文 section 7
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

