# DeepSeek V3 系列


# DeepSeek V3 系列（V3 / R1 / V3.2）

> 原文第 1、16 节。DeepSeek R1（2025 年 1 月）发布后引起极大关注，它是构建在 DeepSeek V3（2024 年 12 月）架构之上的 reasoning 模型。V3.2（2025 年 12 月）则是另一个重要升级，引入了 sparse attention。

## 关键架构：V3 / R1

V3 是 671B 总参数 / 37B 激活参数的 MoE 模型。两大核心技术：

### 1.1 [[MLA（Multi-Head Latent Attention）]]

V3 用 MLA 替代了 GQA。MLA 的核心思想是把 K / V 压缩到低维 latent space 再存进 KV cache，推理时再展开——比 GQA 在相同 cache 大小下表现更好（依据 DeepSeek-V2 论文的消融）。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154850.png)

*Figure 3: Comparison between MLA (used in DeepSeek V3 and R1) and regular MHA.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154851.png)

*Figure 4: Annotated tables from the DeepSeek-V2 paper showing that MLA outperforms GQA.*

> Q 在训练时也会被压缩，但推理时不压缩（避免额外计算）。

### 1.2 [[MoE（Mixture of Experts）]]

V3 用 256 个专家 + 1 个共享专家，每 token 路由 8 个专家 + 始终激活 1 个共享专家：

- 总参数 671B，激活 37B
- 共享专家：始终激活，处理常见 / 重复模式，让其他专家腾出 capacity 学更专门的模式

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154852.png)

*Figure 5: An illustration of the MoE module in DeepSeek V3/R1 (right) compared to a standard FeedForward block (left).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154853.png)

*Figure 6: From "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" — the shared expert idea.*

### 1.3 小结

- V3 是个巨大但推理高效的模型：activation 只有 37B / 671B
- MLA 替代 GQA 是关键设计选择
- 与 Llama 3 405B 相比，V3 更大但推理便宜得多

## DeepSeek V3.2

2025 年 12 月发布，引入 **DeepSeek Sparse Attention (DSA)**，在不显著损失建模性能的前提下，把长上下文推理成本压低。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154935.png)

*Figure 47: A timeline of the DeepSeek model releases since DeepSeek V3.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154936.png)

*Figure 48: The DeepSeek model architecture with multi-head latent and sparse attention.*

> 作者把 V3.2 的细节放到了单独一篇长文里：[A Technical Tour of the DeepSeek Models from V3 to V3.2](https://magazine.sebastianraschka.com/p/a-technical-tour-of-the-deepseek-models-from-v3-to-v3-2)

## 与其他模型的关系

- [[Llama 4]] Maverick 整体结构与 V3 相似，但用 GQA 不用 MLA；MoE 模式是"少量大专家"。
- [[Kimi 系列#8 Kimi K2 / K2 Thinking|Kimi K2]] 是 V3 架构放大到 1T 参数的版本。
- [[Mistral 系列#Mistral 3（2025-12-02）|Mistral 3]] Large 几乎完全是 V3 架构的复刻。
- [[GLM 系列]] GLM-5 学习了 MLA + DSA。

## 来源

> 原文：*The Big LLM Architecture Comparison*, section 1 & 16
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

