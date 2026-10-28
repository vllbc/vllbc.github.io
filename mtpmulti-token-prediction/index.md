# MTP（Multi-Token Prediction）


# MTP（Multi-Token Prediction）

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。一个**训练阶段**的多 token 预测技巧。

## 一句话

传统 LLM 每个位置只预测下一个 token（next-token prediction）。MTP 在每个位置加几个小的 head（线性层），同时输出 t+1..t+k 的 logits，对所有偏移的 cross-entropy 求和（论文推荐 k=4）。

## 训练 vs 推理

| 阶段 | 行为 |
|---|---|
| 训练 | 主 loss + MTP extra loss → 训练信号更丰富，训练速度↑ |
| 推理（普通 MTP） | 仍按 next-token 逐步生成 |
| 推理（speculative MTP） | MTP head 作为 draft model 一次提议 k 个 token，主模型验证 → 推理加速 |

## 与 Speculative Decoding 的关系

MTP head 天然适合做 speculative decoding 的"内部 draft model"——不需要单独训一个 draft 模型。

## 谁在用

- **DeepSeek V3 & V3.2**：训练用
- **GLM-4.5 / GLM-5**：训练用
- **Qwen3-Next**：训练用，且专门针对多步推理优化
- **MiniMax M2**：训练用
- **Xiaomi MiMo**：训练用
- **Nemotron 3 Super**：训练 + 推理都用（共享权重 MTP head 作为 native speculative decoder）

## 对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154940.png)

*Figure 51.2: Multi-Token Prediction versus regular next token prediction.*

## 相关笔记

- [[DeepSeek V3 系列]]
- [[GLM 系列]]
- [[Qwen3 系列]]
- [[MiniMax M2]]
- [[Xiaomi MiMo]]
- [[Nemotron 3]]

## 来源

> *The Big LLM Architecture Comparison*, section 12.3, 18.2
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

