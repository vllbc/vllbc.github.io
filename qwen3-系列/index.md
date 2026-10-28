# Qwen3 系列（Qwen3 / Qwen3-Next / Coder-Next）


# Qwen3 系列

> 原文第 6、12 节。Qwen3 是 2025 年开源模型里"六边形战士"，3 个分支（dense、MoE、Next）覆盖了从 0.6B 到 80B 几乎所有 size 区间。

## 6.1 Qwen3 Dense

Qwen3 dense 有 0.6B / 1.7B / 4B / 8B / 14B / 32B。0.6B 可能是目前最小的"当代"开源 LLM，本地训练 / 推理都极其方便。

### 与 Llama 3 1B 的对比

| 维度 | Qwen3 0.6B | Llama 3 1B |
|---|---|---|
| 设计哲学 | 深而窄 | 浅而宽 |
| 层数 | 较多 | 较少 |
| Hidden / Heads | 较小 | 较大 |
| Tokens/sec | 较慢（深度并行度低） | 较快 |

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154905.png)

*Figure 18: An architecture comparison between Qwen3 0.6B and Llama 3 1B; notice that Qwen3 is a deeper architecture with more layers, whereas Llama 3 is a wider architecture with more attention heads.*

## 6.2 Qwen3 MoE

Qwen3 MoE 有 30B-A3B 和 235B-A22B 两种。

- **dense 适合**：fine-tuning、部署、跨硬件优化
- **MoE 适合**：固定推理预算下的高 capacity
- 同时提供两种 = 满足不同 use case

### 与 DeepSeek V3 的对比（Qwen3 235B-A22B）

Qwen3 235B-A22B 与 DeepSeek V3 惊人相似，**唯一显著区别是 Qwen3 移除了 shared expert**。

> Junyang Lin（Qwen3 开发成员）回复：当时没发现 shared expert 带来足够收益，且担心对推理优化的影响——没有"标准答案"。

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154906.png)

*Figure 19: An architecture comparison between DeepSeek V3 and Qwen3 235B-A22B.*

## 12 Qwen3-Next（2025-09-11）

80B-A3B，Instruct + Thinking 两种变体。比 235B-A22B 小 3×，但专家数 4×，并**重新引入了 shared expert**（回应作者之前的"未来方向"预言）。

### 12.1 专家数 vs 专家大小

- 高专家数 + 共享专家 = 论文中证明对训练稳定性 + 最终质量有益
- 这次 Qwen3 把这条路走到底

### 12.2 [[Gated DeltaNet]] + Gated Attention 混合

Qwen3-Next 的最大亮点：用 **Gated DeltaNet（3 层）+ Gated Attention（1 层）** 混合 3:1 比例，替代纯 attention。

- 让 262k context 的 memory cost 大幅下降
- DeltaNet = 线性 attention 变体，RNN 风格的 fast-weight delta rule
- Gated Attention = 标准 GQA + 输出 gate + zero-centered RMSNorm for QK + partial RoPE
- 1 层 full attention 保留：弥补 DeltaNet 精度不足（content-based retrieval 不如 full attention）

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154923.png)

*Figure 35: The original Qwen3 (left) next to the Qwen3 Next (right).*

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260606154924.png)

*Figure 36: The Gated DeltaNet + Gated Attention hybrid mechanism. Note that these are arranged in a 3:1 ratio.*

### 12.3 [[MTP（Multi-Token Prediction）|MTP]]

每个位置预测 t+1..t+k 个 token（论文推荐 k=4）：

- 训练时：额外的 cross-entropy signal，加速训练
- 推理时：extra heads 可用于 speculative decoding
- Qwen3-Next 的 MTP 还针对多步推理做了优化，spec decode 接受率更高

> V3、V3.2、GLM-4.5、MiniMax M2 训练时也用 MTP（作者没在架构对比里展开讲）。

### 12.4 Qwen3-Coder-Next（2026-02）

80B、3B 激活。SWE-Bench Pro 接近 Claude-Sonnet-4.5。

架构与 Qwen3-Next 80B 完全一致——直接以 Qwen3-Next 为 base 继续训练。

## 相关笔记

- [[Gated DeltaNet]]
- [[MTP（Multi-Token Prediction）]]
- [[MoE（Mixture of Experts）]]
- [[Kimi 系列]] Kimi Linear —— 同样 3:1 混合
- [[MiniMax M2]] —— 同样用 MTP

## 来源

> 原文 section 6 & 12
> 链接：https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

