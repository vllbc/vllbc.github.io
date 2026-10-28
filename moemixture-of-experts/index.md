# MoE（Mixture of Experts）


# MoE（Mixture of Experts）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。本文梳理 2025-2026 年 LLM 中 MoE 的各种设计选择。

## 一句话

MoE 把 transformer block 里的单个 FeedForward 模块替换成**多个 FFN（专家）** + 一个**路由器**。每 token 路由器只选少量专家激活。

## 为什么 2025-2026 全面回潮

- 推理成本主要看**激活参数**而非总参数 → 巨大总参数 + 稀疏激活 = 强 capacity + 低 inference FLOPs
- DeepSeek V3 出圈，证明了这条路在大模型上有效
- 之后几乎所有旗舰开源模型都给了 MoE 版本

## 基本对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154852.png)

*Figure 5: An illustration of the MoE module in DeepSeek V3/R1 (right) compared to a standard FeedForward block (left).*

## 设计维度

| 维度 | 选项 | 典型例子 |
|---|---|---|
| 专家数 | 少量大 / 大量小 | Grok 2.5 8 个 vs DeepSeek 256 个 |
| 是否 shared expert | 有 / 无 | DeepSeek V3 有，Qwen3 无，Qwen3-Next 又加回 |
| 放置 | 每层 / 隔层 | DeepSeek V3 每层（除前 3），Llama 4 隔层 |
| Expert placement | latent space / 原 dim | Nemotron 3 Super 用 latent experts |
| Load balancing | 各种 | Arcee Trinity 提出新策略 |

## Shared Expert 的争议

> Qwen3 砍了 shared expert：Junyang Lin 公开说"没发现明显收益，且担心对推理优化的影响"。
>
> DeepSeek V3 / R1 保留：源于 DeepSpeedMoE 论文，证明能提升建模性能（因为通用模式被 shared 专家吸收，其他专家腾出 capacity 学更专门的东西）。
>
> Qwen3-Next 又加了回来：作者之前在视频里把"高专家数 + 共享专家"列为未来方向。
>
> Grok 2.5：用一个 SwiGLU 模块当"类 shared expert"（intermediate dim 翻倍，不是经典方案）。
>
> Mistral 3 Large、Nemotron 3 Nano、MiniMax M2、Xiaomi MiMo：shared expert 情况各异。

## 少而大 vs 多而小

| 风格 | 代表 | 论点 |
|---|---|---|
| 少而大专家 | Grok 2.5（8 个）、Llama 4 Maverick（2 个大） | 老派做法；推理调度简单 |
| 多而小专家 | DeepSeek V3（256 个）、Qwen3（128 个） | DeepSeekMoE 论文证明在总参数相同时多小专家更好 |
| Arcee Trinity：coarser 多专家 | 介于两者之间 | 利于推理吞吐 |

## 综述图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154853.png)

*Figure 6: From "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" — the shared expert idea.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154916.png)

*Figure 28: From the DeepSeekMoE paper, illustrating the small-experts trend.*

## 相关笔记

- 本仓库已有基础 MoE 笔记：[[MoE]]
- [[DeepSeek V3 系列]]
- [[Llama 4]]
- [[Qwen3 系列]]
- [[Grok 2.5]]
- [[Nemotron 3]]

## 来源

> *The Big LLM Architecture Comparison*, 全文 23 节
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

