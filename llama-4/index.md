# Llama 4


# Llama 4 Maverick

> 原文第 5 节。Llama 4 用 MoE，结构上与 DeepSeek V3 很像，但选择不同。

## 关键设计

Llama 4 Maverick 整体骨架与 DeepSeek V3 类似，但有这些显著区别：

| 维度 | DeepSeek V3 | Llama 4 Maverick |
|---|---|---|
| 总参数 | 671B | 400B |
| 激活参数 | 37B | 17B |
| 注意力 | MLA | GQA |
| MoE 模式 | 9 专家（每专家 hidden 2048） + 1 共享 | 2 专家（每专家 hidden 8192） |
| MoE 放置 | 每层（除前 3 层） | 隔层（dense / MoE 交替） |

也就是说 Llama 4 Maverick 走"**少量大专家**"路线，DeepSeek V3 走"**大量小专家 + 共享**"路线。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154904.png)

*Figure 17: An architecture comparison between DeepSeek V3 (671B) and Llama 4 Maverick (400B).*

## 关键洞察

- 同样 MoE 趋势下，两种设计哲学：DeepSeek 偏"细粒度 + shared"，Llama 4 偏"少而大"
- Llama 4 原生多模态支持（但本文只关注文本部分）
- 2025 年 MoE 已经"不再是少数派"

## 相关笔记

- [[DeepSeek V3 系列]] —— Maverick 的设计参照
- [[MoE（Mixture of Experts）]] —— 两种 MoE 哲学的对比
- [[Kimi 系列]] Kimi K2 —— 又一个 DeepSeek V3 放大版

## 来源

> 原文 section 5
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

