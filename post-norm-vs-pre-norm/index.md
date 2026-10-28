# Post-Norm vs Pre-Norm


# Post-Norm vs Pre-Norm

> 来自 Sebastian Raschka 的 *The Big LLM Architecture Comparison*。汇总 2025 年 LLM 中 normalization 放置的各种设计。

## 一句话

Normalization 放在 attention / FFN 之前还是之后，对训练稳定性影响很大。2025 年的 LLM 至少分 4 派。

## 四种主要放置

1. **Post-LN**（原始 Transformer 2017）：norm 在残差**外**
2. **Pre-Norm**（GPT-2 起主流）：norm 在 attention / FFN **前**
3. **Post-Norm 在残差内**（OLMo 2 风格）：norm 在 attention / FFN **后**，但仍在残差内
4. **Pre + Post 都有**（Gemma 系列风格）：norm 在 attention / FFN **前后都有**

## 对比图

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154855.png)

*Figure 8: A comparison of Post-Norm, Pre-Norm, and OLMo 2's flavor of Post-Norm.*

## 各家选择

| 模型 | 放置 |
|---|---|
| 原始 Transformer | Post-LN（norm 在残差外） |
| GPT-2 / 大多数 LLM | Pre-Norm |
| OLMo 2 | Post-Norm（在残差内） |
| Gemma 2 / 3 / 4 | Pre + Post 都有 |
| Arcee Trinity Large | Pre + Post + Depth-Scaled Sandwich（4 个 norm） |

## 训练稳定性证据

- 2020 年 Xiong et al. 证明 Pre-LN 在初始化时梯度更 well-behaved
- Pre-LN 不需要 learning rate warm-up
- OLMo 2 团队的消融显示 Post-Norm + [[QK-Norm]] 组合能稳定 loss

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154856.png)

*Figure 9: A plot showing the training stability for Pre-Norm vs OLMo 2's flavor of Post-Norm.*

## Depth-Scaled Sandwich Norm（Arcee Trinity）

Arcee Trinity 4 个 RMSNorm，并且第 2 个 RMSNorm 的 gain 初始化为约 1/√L（L = 总层数）：

- 训练早期残差更新小
- 随训练学到合适 scale 长大
- "看起来像 Gemma 3 sandwich norm，但加了深度缩放"

## 相关笔记

- [[QK-Norm]] —— 通常与 Post-Norm 一起出现
- [[OLMo 2]]
- [[Gemma 系列]]
- [[Arcee Trinity Large]]

## 来源

> *The Big LLM Architecture Comparison*, section 2.1, 3.2, 15, 20
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

