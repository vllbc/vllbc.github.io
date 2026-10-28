# Latent Experts


# Latent Experts

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。

## 一句话

Latent Experts = MoE 中的专家在 **latent space（低维）** 工作，而非原始 token embedding 空间：

- 输入 4096 维 down-project 到 1024
- 专家在 1024 维工作
- 输出 up-project 回 4096

→ 节省计算量，专家可以更"小"。

## 谁在用

- **Nemotron 3 Super**（2026-03 发布）

## 对比传统 MoE

| 维度 | 传统 MoE | Latent MoE |
|---|---|---|
| 专家 dim | 等于 FFN 隐藏层 | 等于 latent dim（更小） |
| 计算 | 直接 in/out | 多了 down/up project |
| 专家容量 | 大 | 小但高效 |

## 图示

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154941.png)

*Figure 51.3: Nemotron 3 Super 120B-A12B with latent MoE layers, multi-token prediction, and the Mamba-2 hybrid attention approach.*

## 相关笔记

- [[MoE（Mixture of Experts）]]
- [[Nemotron 3]]

## 来源

> *The Big LLM Architecture Comparison*, section 18.2
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

