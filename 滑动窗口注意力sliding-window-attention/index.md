# 滑动窗口注意力（Sliding Window Attention）


# 滑动窗口注意力（Sliding Window Attention, SWA）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

把 attention 从"每个 token 都能 attend 所有 token"变成"每个 token 只能 attend 邻近 window 内的 token"——KV cache 内存大幅下降。

## 起源

2020 年 LongFormer 论文首次提出。Gemma 2 是首个在主流开源 LLM 中使用 SWA 的。

## 与 Global Attention 配合

实际使用是"局部 + 全局"混合：每隔若干层保留一层 full attention，让全局信息仍能流动。

## 谁在用 + 配置

| 模型 | local:global 比例 | 窗口大小 |
|---|---|---|
| Gemma 2 | 1:1 | 4096 |
| Gemma 3 | 5:1 | 1024 |
| Gemma 4 | 5:1 | 1024（global 层 V = K） |
| GPT-OSS | 1:1 | — |
| Olmo 3 7B | 5:1（参考 Gemma 3） | — |
| Arcee Trinity Large | 3:1 | 4096 |
| Xiaomi MiMo | 5:1 | **128**（极激进） |
| Mistral Small 3.1 | 不使用 | — |
| MiniMax M2 | 默认关闭 | — |

## 效果

- KV cache 内存大幅下降
- 对 perplexity 几乎无影响（Gemma 3 论文消融）
- 推断：SWA 减少 memory 但**不减少 latency**（所以 Mistral Small 3.1 放弃）

## 对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154858.png)

*Figure 11: Annotated figure from Gemma 3 paper showing the KV cache memory savings via sliding window attention.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154859.png)

*Figure 12: A comparison between regular attention (left) and sliding window attention (right).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154900.png)

*Figure 13: Sliding window attention has little to no impact on perplexity (Gemma 3 paper).*

## 相关笔记

- [[Gemma 系列]]
- [[GPT-OSS]]
- [[OLMo 2]] Olmo 3
- [[Arcee Trinity Large]]
- [[Xiaomi MiMo]]
- [[Mistral 系列]]

## 来源

> *The Big LLM Architecture Comparison*, section 3.1
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

