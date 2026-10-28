# Attention Sinks


# Attention Sinks

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

在序列开头放一个"永远被 attend 到"的特殊 token，让长上下文 attention 稳定。这个特殊 token 会自然学到"整序列通用信息"。

## 起源

*Efficient Streaming Language Models with Attention Sinks* 论文。

## 经典做法 vs gpt-oss 做法

| 做法 | 描述 |
|---|---|
| 经典 | 在 tokenized input 开头放一个特殊 token |
| gpt-oss | 不污染 input，而是用**每个 head 学到的 bias logits**，append 到 attention scores |

→ 同样的稳定效果，但不修改 token 序列。

## Arcee Trinity Large 的"减少 attention sinks"

Arcee Trinity 在 attention 中加 elementwise gating，能"减少 attention sinks"——意思是设计上对 attention sinks 现象做了抑制。

## 图示

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154919.png)

*Figure 31: The use of attention sinks in gpt-oss.*

## 相关笔记

- [[GPT-OSS]]
- [[Arcee Trinity Large]]

## 来源

> *The Big LLM Architecture Comparison*, section 9.3
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

