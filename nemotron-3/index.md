# Nemotron 3（Nano / Super）


# Nemotron 3（Nano / Super）

> 原文第 18 节。NVIDIA 在 2025-12 发布的 Nemotron 3 系列，开源权重 + 技术报告 + 数据集 + 训练代码（与 Olmo 3 透明度相当）。

## 18.1 Nemotron 3 Nano（30B-A3B）

### 架构总览

Nemotron 3 Nano 是 **Mamba-Transformer 混合 + MoE**：

- 52 层（13 macro blocks × Mamba-2 + MoE sub-blocks + 少量 GQA 层）
- MoE：128 专家，每 token 激活 1 shared + 6 routed
- 几乎不用 self-attention，只在少量层用

比 Qwen3-Next / Kimi Linear 更激进——attention 层比例极少。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154939.png)

*Figure 51.1: Outline of the Nemotron 3 Nano model, which is a Transformer-Mamba hybrid.*

### Mamba-2 是什么

概念上类似 [[Gated DeltaNet]] / [[KDA（Kimi Delta Attention）|KDA]]：

- 维护一个 running hidden state
- 用 learned gates 把新输入混进 state
- 线性 O(n) 缩放（不像 attention O(n²)）

> "Beyond Standard LLMs" 是作者的另一篇专门讲 SSM 的文章。

### 性能

> "really good performance compared to pure transformer architectures of similar size, while achieving much higher tokens-per-second throughput"

## 18.2 Nemotron 3 Super（120B，2026-03-11 发布）

Super 版在 Nano 基础上做了两个主要改动：

### 改动 1：[[MTP（Multi-Token Prediction）|MTP]]

MTP 训练时同时预测多个未来 token：

- 训练信号更丰富 → 质量 + 推理效率都受益
- 与一般 MTP 不同：**Nemotron 3 Super 在推理时也用 MTP**
- 共享权重的 MTP head 充当"内部 draft model"，做 native speculative decoding
- 不需要外部 draft model

作者建议改叫"shared-weight MTP for speculative decoding"，而非"MTP-3"。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154940.png)

*Figure 51.2: Multi-Token Prediction versus regular next token prediction.*

### 改动 2：Latent Experts

专家在 latent space 操作：

- 输入 4096 维先 down-project 到 1024
- 专家在 1024 维上工作
- 输出 up-project 回 4096

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154941.png)

*Figure 51.3: Nemotron 3 Super 120B-A12B with latent MoE layers, multi-token prediction, and the Mamba-2 hybrid attention approach.*

### 性能

- 2× faster than Qwen3.5 122B-A10B
- NVFP4 版本 2.2× faster than GPT-OSS 120B
- benchmark 与 Qwen3.5 122B-A10B、GPT-OSS 120B 同档

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154942.png)

*Figure 51.4: Nemotron 3 Super benchmark comparison from the Hugging Face Hub page.*

## 相关笔记

- [[MoE（Mixture of Experts）]]
- [[MTP（Multi-Token Prediction）]]
- [[Gated DeltaNet]] —— Mamba-2 的近亲
- [[Qwen3 系列]] Qwen3-Next —— 同样走 linear attention 路线

## 来源

> 原文 section 18
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

