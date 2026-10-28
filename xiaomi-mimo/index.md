# Xiaomi MiMo-V2-Flash


# Xiaomi MiMo-V2-Flash

> 原文第 19 节。小米 2025-12 发布的 309B MoE（15B 激活），benchmark 接近 DeepSeek V3.2，参数只有一半，推理更快。

## 关键设计

### 极激进滑动窗口

- **5:1 SWA + global attention**（与 Gemma 3 同款比例）
- **滑动窗口大小 = 128** —— 比 Gemma 3 的 1024 还小 8×
- 作者说这是已知的"最大 SWA 模型"

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154943.png)

*Figure 52: Xiaomi MiMo-V2-Flash compared to DeepSeek V3.2.*

### 训练时用 [[MTP（Multi-Token Prediction）|MTP]]

与 DeepSeek V3、V3.2、GLM-4.5、MiniMax M2 类似，训练时启用 MTP 提供额外 signal。

## 相关笔记

- [[滑动窗口注意力（Sliding Window Attention）]]
- [[MoE（Mixture of Experts）]]
- [[MTP（Multi-Token Prediction）]]
- [[Gemma 系列]] Gemma 3 —— 同样 5:1 SWA
- [[DeepSeek V3 系列]] —— benchmark 对照

## 来源

> 原文 section 19
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

