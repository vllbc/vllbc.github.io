# Mistral 系列（Small 3.1 / Mistral 3）


# Mistral 系列（Mistral Small 3.1 / Mistral 3）

> 原文第 4、17 节。Mistral 团队的两代代表性模型：一个 24B dense、一个 675B MoE。

## Mistral Small 3.1（2025-03）

24B dense，超过 Gemma 3 27B（除数学 benchmark 外），且推理更快。

速度优势的来源：

- **自定义 tokenizer**
- **缩 KV cache**
- **层数更少**

架构上属于"标准 LLM"——GQA、无 SWA。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154903.png)

*Figure 16: An architecture comparison between Gemma 3 27B and Mistral 3.1 Small 24B.*

> 早期 Mistral 模型用过 SWA，但 Small 3.1 似乎放弃了——`sliding_window: null` 在 model hub config 里。可能是为了用更优化的 FlashAttention（sliding window 减少内存但不减少 latency，而 Small 3.1 关注的是 latency）。

## Mistral 3（2025-12-02）

发布与 DeepSeek V3.2 前后只差一天。包含：

- 3 个小 dense 模型（Ministral 3 系列）：3B / 8B / 14B
- 旗舰：Mistral 3 Large 675B MoE（41B 激活） + 2.5B 视觉编码器

Mistral 自 Mixtral（2023）以来第一次做 MoE。

### 关键点

- 训练 in-house（不从 DeepSeek V3 继续训练，因为 tokenizer 不同）
- 与 NVIDIA 合作优化 Blackwell 芯片的 tokens/sec
- 架构上**几乎与 DeepSeek V3 / V3.1 完全一样**，只把专家大小 ×2、数量 ÷2
- 与 DeepSeek V3.2-Thinking 在 benchmark 上比差距明显
- 卖点：低成本、低延迟、多模态

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154937.png)

*Figure 49: Mistral 3 Large benchmarks from the Mistral 3 announcement, with the DeepSeek V3.2 results overlayed on top.*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154938.png)

*Figure 50: DeepSeek V3 and Mistral 3 Large side by side.*

## 相关笔记

- [[DeepSeek V3 系列]] —— Mistral 3 Large 与 V3 同构
- [[Kimi 系列]] Kimi K2 —— 同样放大 V3 架构
- [[滑动窗口注意力（Sliding Window Attention）]] —— Small 3.1 不用 SWA

## 来源

> 原文 section 4 & 17
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

