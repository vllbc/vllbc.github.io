# KDA（Kimi Delta Attention）


# KDA（Kimi Delta Attention）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。Kimi Linear 对 [[Gated DeltaNet]] 的精细改进。

## 一句话

KDA = 把 Gated DeltaNet 中的**标量 gate** 改成 **channel-wise gate**（每个 feature 通道独立）：

- Gated DeltaNet：每 head 一个标量 gate 控制 memory decay
- KDA：每 head × 每 feature 通道 一个 gate，更精细地控制 memory 读写

→ 论文说能给 memory 更细的控制，改进长上下文 reasoning。

## Kimi Linear 的整体结构

- 3 层 KDA + 1 层 MLA + gate 3:1 混合
- MLA（global）层用 [[NoPE（No Positional Embedding）|NoPE]]，让 MLA 跑成纯 MQA
- RoPE 不在 MLA 层用，作者说"位置偏置由 KDA block 处理"

## 与 Qwen3-Next 的对比

| 维度 | Qwen3-Next | Kimi Linear |
|---|---|---|
| 线性层 | Gated DeltaNet（scalar gate） | KDA（channel-wise gate） |
| 全局层 | Gated GQA | MLA + gate |
| RoPE | 标准 | 在 MLA 层用 NoPE |

## 性能

> Kimi Linear is as fast as GatedDeltaNet, and much faster than an architecture with multi-head latent attention (like DeepSeek V3/R1), while having a higher benchmark performance.

→ 在 long-context / reasoning benchmark 上**linear but no accuracy drop**。

## 图示

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154929.png)

*Figure 41: An overview of the linear attention hybrid architectures.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154930.png)

*Figure 42: Qwen3-Next and Kimi Linear side by side.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154931.png)

*Figure 43: Annotated figure from the Kimi Linear paper showing that Kimi Linear is as fast as GatedDeltaNet.*

## 相关笔记

- [[Gated DeltaNet]]
- [[MLA（Multi-Head Latent Attention）]]
- [[NoPE（No Positional Embedding）]]
- [[Kimi 系列]] Kimi Linear
- [[Qwen3 系列]] Qwen3-Next

## 来源

> *The Big LLM Architecture Comparison*, section 14.5
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

