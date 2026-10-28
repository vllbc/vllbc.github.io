# Gemma 系列（Gemma 3 / 3n / 4）


# Gemma 系列（Gemma 3 / 3n / 4）

> 原文第 3、23 节。Google 的 Gemma 系列一直是"被低估"的高质量开源模型——27B 这个 size 区间在 Mac Mini 上都能跑。

## Gemma 3（2025 年初）

### 3.1 [[滑动窗口注意力（Sliding Window Attention）]]

Gemma 2 已经用滑动窗口，Gemma 3 进一步把 local:global 比例从 1:1 改成 **5:1**（每 5 层 SWA + 1 层 full attention），窗口大小从 4096 缩到 **1024**。

效果：KV cache 内存大幅下降，对 perplexity 几乎没影响。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154858.png)

*Figure 11: Annotated figure from Gemma 3 paper showing the KV cache memory savings via sliding window attention.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154859.png)

*Figure 12: A comparison between regular attention (left) and sliding window attention (right).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154900.png)

*Figure 13: Sliding window attention has little to no impact on perplexity (Gemma 3 paper).*

### 3.2 [[Post-Norm vs Pre-Norm|Pre-Norm + Post-Norm 混合]]

Gemma 3 在 attention / FFN **前后都加 RMSNorm**（Gemma 2 也是这样），算"两头兼顾"。

- 原始 Transformer：Post-LN
- GPT-2 / 主流：Pre-Norm
- OLMo 2：Post-Norm（在残差内）
- Gemma 3：Pre + Post 都有

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154901.png)

*Figure 14: An architecture comparison between OLMo 2 and Gemma 3; note the additional normalization layers in Gemma 3.*

### 3.3 小结

- 27B size 在 Mac Mini 上能跑，是个甜蜜点
- 滑动窗口是核心效率手段
- Pre+Post Norm 是它的"招牌"
- 与 DeepSeek V3 不同，Gemma 3 是 dense 模型（不是 MoE）

## Gemma 3n（手机优化版）

针对小设备（手机）优化。两大新设计：

1. **Per-Layer Embedding (PLE)**：只把"模型核心参数"放 GPU，token / layer 特定的 embedding（文本、音频、视觉）按需从 CPU / SSD 流式加载
2. **MatFormer (Matryoshka Transformer)**：单个 LLM 可以"切片"成多个独立可用的子模型，推理时只跑需要的部分

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154902.png)

*Figure 15: Annotated figure from Google's Gemma 3n blog illustrating the PLE memory savings.*

## Gemma 4（2026 年 4 月发布）

架构上**变化很小**——基本与 Gemma 3 一样：

- 5:1 SWA + GQA
- Pre+Post Norm
- 仍是 dense 27B（升级到 31B）

两个容易被忽略的小变化：

1. **V = K**：全局 attention 层把 value 复用为 key，KV cache 再省一份
2. **p-RoPE**：只对 25% 的频率对加位置信息，减少长上下文的位置噪声

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154949.png)

*Figure 58: Gemma 3 (27B) and Gemma 4 (31B) side by side.*

虽然架构改动小，但 benchmark 跳跃很大：在 AI Arena 上 Gemma 4 31B 接近更大的 Qwen3.5-397B-A17B。

### Gemma 4 MoE

也发布了 MoE 版：26B-A4B，benchmark 与 dense 31B 几乎打平。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154951.png)

*Figure 60: Qwen3 Coder Flash compared to Gemma 4 MoE.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154952.png)

*Figure 61: Gemma 4 MoE (26B-A4B) is only slightly worse than Gemma 4 (31) dense.*

## 相关笔记

- [[滑动窗口注意力（Sliding Window Attention）]]
- [[Post-Norm vs Pre-Norm]]
- [[QK-Norm]]
- [[Partial RoPE]]
- [[Mistral 系列]] —— Mistral Small 3.1 与 Gemma 3 的对比
- [[GPT-OSS]] —— 也用滑动窗口
- [[Xiaomi MiMo]] —— 激进的 SWA

## 来源

> 原文 section 3 & 23
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

