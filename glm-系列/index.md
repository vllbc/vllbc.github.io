# GLM 系列（GLM-4.5 / GLM-5）


# GLM 系列（GLM-4.5 / GLM-5）

> 原文第 11、21 节。智谱（z.AI / Zhipu）的旗舰开源模型系列。

## 11 GLM-4.5

GLM-4.5 是 2025 年的主推模型：

- 旗舰 355B MoE，平均 12 个 benchmark 超过 Claude 4 Opus
- Air 版本 106B
- Instruct / reasoning hybrid，类似 Qwen3，但更针对 function calling / agent

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154921.jpeg)

*Figure 33: GLM-4.5 benchmark.*

### 关键设计

- **开头 3 层 dense + 之后 MoE**（学自 DeepSeek V3）
- dense 开头的目的：稳定早期收敛；MoE 路由决策如果一开始就介入，会干扰低层语法 / 语义特征提取
- **保留 shared expert**（与 Qwen3 不一样）
- **保留 attention bias**（与 GPT-2 / GPT-OSS 一致）

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154922.png)

*Figure 34: GLM-4.5 next to a similarly-sized Qwen3 model.*

## 21 GLM-5（2026-02-11）

春节前一周发布，是 2026 年 GLM 系列的开年旗舰：

- 总参数 744B（GLM-4.5 是 355B，翻倍）
- 激活参数 40B（GLM-4.5 是 32B）
- 介于 DeepSeek V3.2 和 Kimi K2 之间

### 关键升级

- **采用 DeepSeek 的 [[MLA（Multi-Head Latent Attention）|MLA]]** —— 这是 GLM-4.5 没有的
- **采用 DeepSeek Sparse Attention**（与 V3.2 同源）
- 专家从 160 增到 256
- embedding / expert dim 从 5120 → 6144
- intermediate 从 1536 → 2048
- 层数从 92 → 78（变浅，可能是为了降推理延迟——层深度无法并行）

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154947.png)

*Figure 56: Architecture of GLM-5 and GLM-4.5 side by side.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154948.png)

*Figure 57: GLM architectures next to benchmarks. GLM-4.7 architecture is similar to GLM-4.5.*

## 相关笔记

- [[DeepSeek V3 系列]] —— GLM 学了 V3 的 MLA + sparse attention
- [[MoE（Mixture of Experts）]]
- [[MLA（Multi-Head Latent Attention）]]
- [[Qwen3 系列]] —— 同期对照

## 来源

> 原文 section 11 & 21
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

