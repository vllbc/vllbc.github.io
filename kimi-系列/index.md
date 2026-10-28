# Kimi 系列（K2 / K2 Thinking / Kimi Linear）


# Kimi 系列（K2 / K2 Thinking / Kimi Linear）

> 原文第 8、14 节。Moonshot AI 的旗舰开源系列。K2 是 1T 参数的"超大杯"，Kimi Linear 是 2025 年 10 月推出的线性注意力版本。

## 8 Kimi K2 / K2 Thinking

K2（2025 年）是个 1T 参数的"庞然大物"：

- 训练用了 **Muon optimizer**（不是 AdamW）——首次在 1T 量级使用
- 架构基本是 **DeepSeek V3 放大版**（更多专家、更少 MLA 头）
- 2025-11-06 推出 K2 Thinking：架构不变，context 从 128k 扩到 256k

### 训练 loss 曲线

> Kimi 团队特别强调训练 loss 曲线"异常平滑"——但作者指出这不稀奇（看 OLMo 2 也很平滑），真正值得注意的是 loss 衰减得多好。这得益于 Muon optimizer。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154911.png)

*Figure 24: Annotated figures from the Kimi K2 announcement blog article and the OLMo 2 paper.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154912.png)

*Figure 25.1: An architecture comparison between DeepSeek V3 and Kimi K2.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154913.jpeg)

*Figure 25.2: DeepSeek R1 versus Kimi K2 Thinking architecture (top) and Kimi K2 Thinking benchmarks (bottom).*

> 截至本文撰写时，K2 可能是当世最大开源 LLM（Llama 4 Behemoth 没发、闭源不算、Google Switch Transformer 1.6T 是不同代 encoder-decoder）。

## 14 Kimi Linear（2025-10）

Kimi Linear 是在 MiniMax M2 刚回归 full attention 之后推出的"线性注意力"反击。

### 14.1 / 14.2 传统 attention 与线性 attention

传统 attention：QK^T 是 n×n 矩阵 → O(n²) 时间空间。

线性 attention 用 kernel trick 把计算重排：

$$\text{Attention}(Q, K, V) = \phi(Q)(\phi(K)^\top V)$$

→ O(n) 而非 O(n²)。问题：早期方案（2020 Transformers are RNNs）都因为精度下降而未在 SOTA 模型中落地。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154928.png)

*Figure 40: Illustration of the quadratic cost in attention due to sequence length n.*

### 14.3 线性注意力的"复兴"

| 时间 | 模型 | 线性 attention 方案 |
|---|---|---|
| 2025-06 | MiniMax M1 | lightning attention |
| 2025-08 | Qwen3-Next | [[Gated DeltaNet]] + Gated Attention 3:1 |
| 2025-09 | DeepSeek V3.2 | DeepSeek Sparse Attention |
| 2025-12 | MiniMax M2 | **回退**到 full attention |
| 2025-10 | Kimi Linear | KDA + MLA 3:1 |

> MiniMax 团队公开说：linear attention 在生产 LLM 里很棘手——普通 prompt 看起来还行，但 reasoning / multi-turn / agentic 任务掉点严重。
>
> 这本来可能是 linear attention 走不通的"判决"——但 Kimi Linear 用 KDA 拿回了一些精度。

### 14.4 Kimi Linear vs. Qwen3-Next

两者结构上很像：

- 都用 3:1 混合（3 层线性 + 1 层 full attention）
- 共享 [[Gated DeltaNet]] 思路

区别：

| 维度 | Qwen3-Next | Kimi Linear |
|---|---|---|
| full attention 层 | Gated GQA | [[MLA（Multi-Head Latent Attention）|MLA]] + gate |
| 线性层 | Gated DeltaNet（scalar gate） | [[KDA（Kimi Delta Attention）|KDA]]（channel-wise gate） |
| 位置编码 | RoPE | 在 MLA 层用 NoPE（让 MLA 跑成纯 MQA） |

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154930.png)

*Figure 42: Qwen3-Next and Kimi Linear side by side.*

### 14.5 KDA（Kimi Delta Attention）

KDA 是对 Gated DeltaNet 的精细改进：

- Gated DeltaNet：每个 head **一个标量 gate** 控制 memory decay
- KDA：**每个 feature 通道一个 gate**（channel-wise）

→ 论文说能给"memory 更细的控制"，改进长上下文 reasoning。

> MLA 在 DeepSeek-V2 论文的消融里，在调好超参后与 full attention 持平——这让 Kimi Linear 在长上下文 / reasoning benchmark 上"linear 但不掉点"成为可能。

48B 参数，比 K2（1T）小 20×，但走另一条路。

## 相关笔记

- [[MLA（Multi-Head Latent Attention）]]
- [[MoE（Mixture of Experts）]]
- [[Gated DeltaNet]]
- [[KDA（Kimi Delta Attention）]]
- [[NoPE（No Positional Embedding）]]
- [[Qwen3 系列]] Qwen3-Next
- [[DeepSeek V3 系列]] DeepSeek V3.2
- [[MiniMax M2]]

## 来源

> 原文 section 8 & 14
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

