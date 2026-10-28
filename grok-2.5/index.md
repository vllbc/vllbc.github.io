# Grok 2.5


# Grok 2.5

> 原文第 10 节。xAI 在 2025 年中发布了 Grok 2.5 的权重（去年 xAI 的旗舰生产模型）。这给我们提供了一个少有的"真实生产系统"视角。

## 整体架构

270B dense（或 MoE，看 source），整体架构"标准"。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154920.png)

*Figure 32: Grok 2.5 next to a Qwen3 model of comparable size.*

## 关键点

- **少量大专家**（8 个）—— 与近年"小而多专家"趋势相反
- **有一个 SwiGLU 模块作为类共享专家**：不是经典 shared-expert 设计（它的 intermediate dim 翻倍），但思路一致
- 作者特别指出：Qwen3 砍了 shared expert，看 Qwen4 / 后续模型是否回归会很有意思

## 与其他模型的关系

- 跟 [[GPT-OSS]] 一样：都选择"少量大专家"
- 与 [[Qwen3 系列]] 对照：Qwen3 不用 shared expert，Grok 2.5 用
- 真实生产模型，权重可分析——难得

## 来源

> 原文 section 10
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

