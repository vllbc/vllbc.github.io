# GPT-OSS（OpenAI 的开源模型）


# GPT-OSS

> 原文第 9 节。OpenAI 自 GPT-2（2019）以来首次发布开源模型，2025-08。两个 size：gpt-oss-20b 和 gpt-oss-120b。

## 总体架构

整体骨架"很眼熟"——所有组件都在其他模型中见过。3.6B 激活参数，MoE + 滑动窗口 attention + GQA。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154914.png)

*Figure 26: Architecture overview of the two gpt-oss models.*

## 9.1 宽度 vs 深度

| 维度 | gpt-oss | Qwen3 30B-A3B |
|---|---|---|
| 层数 | 24（浅） | 48（深） |
| Embedding dim | 2880（宽） | 2048 |
| FFN expert dim | 2880 | 768 |
| 注意力头数 | 2× | 1× |

作者在 Gemma 2 论文里找到的消融：9B 参数下，宽略胜于深（52.0 vs 50.8）。

- **深模型**：灵活，但训练不稳定（梯度消失 / 爆炸）
- **宽模型**：tokens/sec 更高，memory cost 更高

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154915.png)

*Figure 27: Architecture comparison between gpt-oss and Qwen3.*

## 9.2 少量大专家 vs 大量小专家

gpt-oss 用 32 个专家（每 token 激活 4），比 Qwen3 的 128 个少很多。**这与"细粒度专家"趋势相反**——论文证明同样总参数下细粒度更好，但 gpt-oss 选了少量大专家。

没有用 shared expert（与 Qwen3 一致）。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154916.png)

*Figure 28: From the DeepSeekMoE paper, illustrating the small-experts trend.*

## 9.3 Attention Bias + Attention Sinks

两个"复古"细节：

### Attention bias

gpt-oss 在 attention 层**重新启用了 bias units**——上次见到是 GPT-2 时代。

- 一般认为 redundant
- 2023 年论文（arXiv:2302.08626）证明对 k_proj 至少数学上多余
- 消融显示对最终 loss 影响微乎其微

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154917.png)

*Figure 29: gpt-oss models use bias units in the attention layers.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154918.png)

*Figure 30: Table from https://arxiv.org/pdf/2302.08626 showing average test loss with and without bias.*

### [[Attention Sinks]]

经典 attention sinks：开头放一个"总是被 attend 到"的特殊 token，存一些"整序列通用信息"，让长上下文 attention 稳定。

gpt-oss 的实现更巧妙：**不是真实 token**，而是**每个 head 学到的 bias logits，append 到 attention scores**。

- 同样的稳定效果
- 不污染 tokenized input

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154919.png)

*Figure 31: The use of attention sinks in gpt-oss.*

## 其他

- **滑动窗口 attention**：每 2 层一层 SWA（Gemma 3 是 5:1）
- 用了[[滑动窗口注意力（Sliding Window Attention）|sliding window]]
- 与 GPT-2 的对比作者另写了一篇长文：[From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss)

## 相关笔记

- [[MoE（Mixture of Experts）]]
- [[滑动窗口注意力（Sliding Window Attention）]]
- [[Attention Sinks]]
- [[Qwen3 系列]] —— 与 gpt-oss 大小相近的 MoE
- [[DeepSeek V3 系列]] DeepSeek-V3
- [[GLM 系列]] GLM-4.5 —— 也保留了 attention bias

## 来源

> 原文 section 9
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

