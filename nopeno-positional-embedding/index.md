# NoPE（No Positional Embedding）


# NoPE（No Positional Embedding）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

NoPE = 完全不加位置编码。没有绝对位置 embedding，也没有 RoPE。起源是 2023 年的 *The Impact of Positional Encoding on Length Generalization in Transformers* 论文。

## 怎么 work

- 因果 mask 已经天然编码方向：位置 t 只能 attend 到 ≤ t
- 模型通过 gradient descent 可以学到位置信息，如果对 loss 有利
- NoPE 论文额外发现：**长度泛化能力更好**——context 变长时性能下降更少

## 局限

- 原论文实验用 ~100M 参数小模型 + 小 context
- 大模型 + 长 context 下效果如何，不清楚
- 所以主流 LLM 都"谨慎"使用 NoPE——**只在部分层**用，或**只在 global 层**用

## 谁在用

| 模型 | 用法 |
|---|---|
| SmolLM3 | 每 4 层用 NoPE，其余仍用 RoPE |
| Arcee Trinity Large | global attention 层用 NoPE |
| Kimi Linear | 在 MLA（global）层用 NoPE，让 MLA 跑成纯 MQA |

## 对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154909.png)

*Figure 22: Absolute positional embeddings (from the "Build a Large Language Model (From Scratch)" book).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154910.png)

*Figure 23: Annotated figure from the NoPE paper showing better length generalization with NoPE.*

## 相关笔记

- [[Partial RoPE]] —— 另一种"减少 RoPE 强度"的方案
- [[SmolLM3]]
- [[Arcee Trinity Large]]
- [[Kimi 系列]] Kimi Linear

## 来源

> *The Big LLM Architecture Comparison*, section 7.1
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

