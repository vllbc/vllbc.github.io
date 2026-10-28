# LLM 八股导览


# LLM 八股导览

> [!info] TL;DR
> **本文是什么**：一张覆盖 LLM 时代核心概念的"知识地图"。不是教科书，是**导航**——告诉你"要懂什么"、"去哪篇笔记深入"、"看哪些外部参考"。
> **怎么用**：
> 1. 面试前 1 小时扫一遍 → 用 §0 章节地图和 §13 速查表检查覆盖
> 2. 准备某个模块 → 点对应章节 → 章节内的 `[[wikilink]]` 跳转到具体笔记
> 3. 想读 external material → 每节末尾的「延伸阅读」按优先级排好
> **范围**：Transformer → 主流架构 → 预训练 → 后训练 → 推理 → Agent → VLM → 工程实践。Foundation（ML/DL 基础）不在本文展开，见对应单独笔记。

## 笔记地图

```dataview
TABLE file.ctime AS 创建
FROM "LLM Architecture" OR "LLM Training" OR "LLM Inference" OR "Agent" OR "Engineering"
WHERE contains(tags, "LLM") OR contains(categories, "LLM")
SORT file.mtime DESC
LIMIT 30
```

---

## §0 章节地图

| 章节 | 主题 | 高频考点 | 推荐笔记入口 |
|---|---|---|---|
| §1 | Transformer 基础 | self-attention、RoPE、归一化 | [[LLM Architecture/model/Transformer\|Transformer]] |
| §2 | Attention 演进 | MHA/MQA/GQA/MLA、Flash/Sliding | [[MHA]] / [[MLA]] |
| §3 | 位置编码 | RoPE / ALiBi / NoPE / Partial RoPE | [[rope]] |
| §4 | 归一化与残差 | Pre/Post-Norm、QK-Norm | [[Post-Norm vs Pre-Norm]] / [[QK-Norm]] |
| §5 | Tokenization | BPE / WordPiece / ULM / SP | [[tokenization]] |
| §6 | MoE | shared expert、router、平衡 | [[MoE（Mixture of Experts）]] |
| §7 | 主流模型 | Llama / Qwen / DeepSeek / GLM / RWKV / Mamba | [[llama系列]] / [[DeepSeek V3 系列]] / [[rwkv]] |
| §8 | 预训练 | Scaling Law、数据工程、分布式 | [[UltraScale Playbook 笔记]] / [[tensor_parallel]] |
| §9 | 后训练 | PPO / GRPO / DPO / 推理强化 | [[grpo]] / [[ppo]] / [[OPD]] |
| §10 | 推理与压缩 | KV Cache / Paged / 量化 / 长度外推 | [[KV cache]] / [[paged attention]] |
| §11 | Agent | Memory / Planning / Tool / RL | [[agent概览]] / [[context_engineering]] |
| §12 | VLM | Patch、跨模态对齐 | [[patch详解]] |
| §13 | 工程实践 | verl / vLLM / 分布式 | [[vllm高效推理全链路]] / [[verl总体概览]] |
| §14 | 速查表 | 一句话定义清单 | §14 |

> [!tip] 推荐学习顺序
> §1 → §2 → §3 → §5 → §6 → §7 → §8 → §9 → §10 → §11 → §12 → §13
> 任何一节都允许跳读（wikilink 互相连通）。

---

## §1 Transformer 基础

> [!quote] 一句话
> Transformer = **Self-Attention + FFN + 残差 + LayerNorm**。Encoder-Decoder 分离；Decoder 用 masked self-attention 保证因果性。架构定义见 [[LLM Architecture/model/Transformer|Transformer]]。

### 核心组件
- **Self-Attention**：`Attention(Q, K, V) = softmax(QKᵀ/√d)V`，详见 [[MHA]]
- **Multi-Head**：把 Q/K/V 拆成 h 头，各头独立计算后 concat
- **Masked Attention**：Decoder 用上三角 mask 保证位置 t 看不到位置 >t
- **FFN**：`FFN(x) = W₂ · activation(W₁x + b₁) + b₂`，现代 LLM 用 SwiGLU
- **残差 + LayerNorm**：稳定深层梯度流

### 经典八股题（自检）
- [ ] 为什么除以 √d？→ 答：避免 softmax 饱和（梯度消失），详细推导见 [[Transformer]] §"为什么除以根号 d"
- [ ] 为什么 multi-head？→ 答：多子空间表达 + 不增加计算量
- [ ] Embedding × √d 的原因？→ 答：Xavier 初始化后 Embedding 方差 = 1/d，PE 是 sin/cos ∈ [-1,1]；两路相加需要恢复量级
- [ ] 权重共享在哪些地方？→ 答：(1) Encoder-Decoder Embedding 共享；(2) Decoder Embedding-FC 共享
- [ ] Pre-Norm vs Post-Norm？→ 详见 §4
- [ ] 为什么不用 BatchNorm 用 LayerNorm？→ 答：序列长度变化大时 BN 统计量不稳定；LN 无全局统计量

### 延伸阅读
- **必读**：Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — <https://arxiv.org/abs/1706.03762>
- **图解**：Jay Alammar, *The Illustrated Transformer* — <https://jalammar.github.io/illustrated-transformer/>
- **视频**：3Blue1Brown, *Transformers, explained visually* — <https://www.3blue1brown.com/topics/neural-networks>
- **从零实现**：Sebastian Raschka, *Build a Large Language Model (From Scratch)* — <https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167>
- **复现笔记**：[[Transformer]]（含 PyTorch 实现、padding mask、label smoothing、公式推导）

---

## §2 Attention 演进

> [!quote] 一句话
> 围绕 **KV cache 内存** 和 **建模质量** 两个目标，演化出 MHA → MQA → GQA → MLA 一条主线，外加 Flash / Sliding / Linear 等"工程 / 架构创新"分支。

### 2.1 KV 共享路线（MHA → MQA → GQA → MLA）

| 方法 | KV 头数 | KV cache 大小 | 质量 | 代表模型 |
|---|---|---|---|---|
| **MHA** | = num_q_heads | 最大 | 最好基线 | BERT、原始 Transformer |
| **MQA** | = 1 | 最小 | 略掉 | Falcon、Pythia |
| **GQA** | 分组共享 | 中 | ≈ MHA | Llama 2/3、Mistral |
| **MLA** | latent 压缩 | 接近 MQA | **> MHA**（DS 论文消融） | DeepSeek V2/V3/R1、Kimi K2、Mistral 3 Large、GLM-5 |

详细对比图见 [[MLA（Multi-Head Latent Attention）]]。

**八股考点**：
- **GQA vs MLA**：GQA 把 head 分组共享 K/V；MLA 把 K/V 投影到低维 latent 再 cache（且把 head 间差异信息吸收到 Q 中）
- **为什么 MLA 能比 MHA 还好**：把 head 差异信息转移到 Q 后，**Q 容量增加 → 建模能力上升**；同时 cache 只存共享 latent → 内存下降
- **Kimi Linear**：唯一把 MLA 用在 Gated DeltaNet 混合架构的工作，详见 [[Kimi 系列]] 和 [[MLA（Multi-Head Latent Attention）]]

### 2.2 高效 Attention

| 方法 | 核心思路 | 节省什么 | 代表 |
|---|---|---|---|
| **FlashAttention** | tiling + online softmax，重计算代替存中间矩阵 | HBM IO、显存 | Flash Attention 2/4 |
| **Online Softmax** | 单 pass 维护 running max + denom | 不存 N×N attention 矩阵 | FlashAttention 前置 |
| **PagedAttention** | OS 虚拟内存分页思想管理 KV cache | KV 碎片、显存 | vLLM |
| **Sliding Window** | 每 token 只 attend 邻近 W 个 token | KV cache O(N·W) | Gemma 2、LongFormer |
| **Linear Attention** | 核分解去掉 softmax，O(N) | 计算量 | 不主流 |
| **Gated DeltaNet / KDA** | 线性 RNN + Δ-rule + gate | O(1) decode | Kimi Linear、Qwen3-Next |
| **Mamba / RWKV** | SSM / RNN 结构替换 Attention | O(1) decode | Mamba、RWKV-7 |

笔记：[[flash attention]] / [[online attention]] / [[paged attention]] / [[滑动窗口注意力（Sliding Window Attention）]] / [[Gated DeltaNet]] / [[KDA（Kimi Delta Attention）]] / [[rwkv]]

### 2.3 注意力 Sinks

> [!note]+ Attention Sinks（一句话：初始 token 的 attention 分数被人为撑高，类似 sink）
> 现象：StreamingLLM 观察到，无论输入多长，**前几个 token 的 attention 值始终偏高**——作者把这几个 token 称作 "attention sinks"。去掉它们会让注意力分布坍塌。实操：保留这些 sink + 最近 window → 滑动窗口推理。
> 详见 [[Attention Sinks]]

### 延伸阅读
- **GQA**：Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 — <https://arxiv.org/abs/2305.13245>
- **MLA**：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024 — <https://arxiv.org/abs/2405.04434>
- **FlashAttention**：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022 — <https://arxiv.org/abs/2205.14135>
- **PagedAttention (vLLM)**：Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023 — <https://arxiv.org/abs/2309.06180>
- **Sliding Window**：Beltagy et al., *Longformer: The Long-Document Transformer*, 2020 — <https://arxiv.org/abs/2004.05150>
- **StreamingLLM / Attention Sinks**：Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2024 — <https://arxiv.org/abs/2309.17453>
- **综述大图**：Sebastian Raschka, *The Big LLM Architecture Comparison* — <https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>

---

## §3 位置编码

> [!quote] 一句话
> Transformer 本身对位置无感，需要显式注入。**RoPE 是 2024+ LLM 的事实标准**，其余方案是历史/特例。

| 方法 | 原理 | 特点 | 代表模型 |
|---|---|---|---|
| **绝对 Sin/Cos** | 三角函数位置嵌入 | 不可外推 | 原始 Transformer |
| **RoPE** | 复数域旋转 → 相对位置自然出现在 attention 内积 | 长度内插、远程衰减 | Llama / Qwen / DeepSeek 全系 |
| **ALiBi** | attention score 加线性 bias（与距离成反比） | 无参数、外推友好 | Bloom、MPT |
| **NoPE** | 去掉位置编码 | 推理时给个 causal mask 即可，COT/ICL 反而更好 | 部分实验 |
| **Partial RoPE** | 只对部分维度用 RoPE | 长度外推 + 性能平衡 | Qwen2/3 |
| **NTK-aware / YaRN** | 动态修改 RoPE base / 各频率分量插值 | 长度外推 | Code Llama、Llama 3 Long |

笔记：[[rope]] / [[ALiBi]] / [[NoPE]] / [[NoPE（No Positional Embedding）]] / [[Partial RoPE]]

### RoPE 关键性质
1. **远程衰减**：相对距离越远，attention score 越小（`|q·k|` 关于相对距离单调下降）
2. **长度外推支持**：可通过 frequency 缩放（PI / NTK / YaRN）扩展到训练长度之外
3. **远程相对位置**：`<f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m-n)` —— 内积只与相对位置有关（推导见 [[rope]]）

### 八股考点
- [ ] RoPE 为什么用复数？→ 答：复数乘法天然携带相位（位置），与相对距离挂钩
- [ ] RoPE + KV cache 怎么结合？→ 答：每个新 step 只算当前 token 的 q·rot(θ)，cache 里存已旋转的 k（见 [[KV cache]]）
- [ ] 长度外推三种方法：Position Interpolation（除 s）、NTK-aware（缩 base）、YaRN（按维度分组插值）

### 延伸阅读
- **RoPE**：Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 — <https://arxiv.org/abs/2104.09864>
- **ALiBi**：Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, 2022 — <https://arxiv.org/abs/2108.12409>
- **YaRN**：Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023 — <https://arxiv.org/abs/2309.00071>
- **中文综述（一文通透）**：July_v, *一文通透位置编码：从标准位置编码、旋转位置编码RoPE到ALiBi、LLaMA 2 Long* — <https://blog.csdn.net/v_JULY_v/article/details/134085503>

---

## §4 归一化与残差

### Pre-Norm vs Post-Norm

> [!note]+ 核心区别（折叠）
> - **Post-Norm**（原 Transformer）：`x + Sublayer(LayerNorm(x))` —— 但实际实现多写成 `LayerNorm(x + Sublayer(x))`，训练不稳定，深层必须 warmup
> - **Pre-Norm**（现代 LLM）：`x + Sublayer(LayerNorm(x))` —— 梯度流直接贯穿残差，更稳定，几乎不用 warmup
> - 现代 LLM（Llama/Qwen/DeepSeek）全部 Pre-Norm
> 详见 [[Post-Norm vs Pre-Norm]]

### 其它归一化

| 方法 | 思路 | 代表 |
|---|---|---|
| **LayerNorm** | 对每个 token 内部做均值方差归一 | 几乎所有 LLM |
| **RMSNorm** | 只用 RMS，不做 mean-center（计算更便宜） | Llama、LLaMA 2+ |
| **DeepNorm** | 修正 Post-Norm 让深层也能训练 | DeepSeek 系列 |
| **QK-Norm** | 对 Q、K 做 norm，控制 attention logit 大小 | Cohere Command-R、ChatGLM |
| **Sandwich Norm** | Sublayer 前后都做 LN | 部分 MoE |

笔记：[[Post-Norm vs Pre-Norm]] / [[QK-Norm]]

### 延伸阅读
- **RMSNorm**：Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 — <https://arxiv.org/abs/1910.07467>
- **DeepNorm**：Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*, 2022 — <https://arxiv.org/abs/2203.00555>
- **Peri-LN**（评论）：[[Research/Papers/Transformer/Peri-LN：Revisiting Normalization Layer in the Transformer Architecture|Peri-LN 笔记]]

---

## §5 Tokenization

> [!quote] 一句话
> 把字符串切成 token（整数 id），是 LLM 的入口。BPE / WordPiece / ULM 三大子词算法。

| 算法 | 选择合并标准 | 增量/减量 | 代表 |
|---|---|---|---|
| **BPE** | 频次最高的对 | 增量 | GPT 系列、Llama tokenizer |
| **WordPiece** | 互信息最大（语言模型增益） | 增量 | BERT |
| **Unigram LM** | 删掉后 loss 增加最少的 | 减量 | SentencePiece 默认、XLNet |
| **Byte-level BPE** | BPE over bytes | 增量 | GPT-2 起，避免 OOV |

笔记：[[tokenization]]（含三大算法实现代码）

### 现代 tokenizer 细节
- **SentencePiece**（Kudo, 2018）：语言无关的 tokenizer 训练框架，BPE / ULM 都可用，常用作 LLM 默认实现 — <https://arxiv.org/abs/1808.06226>
- **特殊 token**：`<bos>` / `<eos>` / `<pad>` / `<unk>` / chat template（`[INST]`, `<|im_start|>` 等）
- **多语言**：中文通常走字符级 BPE 或 SP 的 ULM
- **vocab 大小**：32k（Llama）、64k（DeepSeek）、128k+（部分中文模型）

### 八股考点
- [ ] 为什么 BPE 平衡字典大小和编码步骤数？→ 答：高频词留作整体，低频词拆 subword
- [ ] WordPiece 和 BPE 选合并对的方式？→ 答：WordPiece 看"合并后语言模型似然增益"
- [ ] 什么是 OOV？→ 答：词表外的 token，BPE/WordPiece 拆 subword；ULM 给概率；字节级方案天然无 OOV

### 延伸阅读
- **HuggingFace 课程**：*Tokenizers* — <https://huggingface.co/learn/nlp-course/chapter2>
- **SentencePiece**：Kudo, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*, 2018 — <https://arxiv.org/abs/1808.06226>
- **GPT-2 BPE**：Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019

---

## §6 MoE（Mixture of Experts）

> [!quote] 一句话
> 用**多个 FFN（专家）+ 路由器**替换单个 FFN；每 token 只激活少量专家 → 总参数大、激活参数小、推理便宜。2025-2026 旗舰模型全面回潮。

笔记：[[MoE（Mixture of Experts）]]（设计维度表 + shared expert 争议） / [[MoE]]

### 设计维度

| 维度 | 选项 | 典型选择 |
|---|---|---|
| **专家数** | 少而大 / 多而小 | DeepSeek 256 小 vs Grok 2.5 8 大 |
| **Shared Expert** | 有 / 无 | DS V3 有、Qwen3 无、Qwen3-Next 又加回 |
| **放置** | 每层 / 隔层 | DS V3 每层（除前 3）、Llama 4 隔层 |
| **Placement** | 原 dim / latent space | Nemotron 3 Super 用 latent experts |
| **Load Balancing** | aux loss / router bias / none | Arcee Trinity 新策略 |

### 八股考点
- [ ] 什么是 Shared Expert？→ 答：始终激活的"通用"专家，其他专家学更专门模式
- [ ] 为什么大模型几乎全 MoE？→ 答：激活参数决定推理 FLOPs，总参数决定容量；MoE 兼顾两者
- [ ] Router 路由不均怎么办？→ 答：aux loss / load balancing loss；DS V3 还用 dynamic bias
- [ ] 训练 vs 推理专家选择一致性问题？→ 答：训练采样到 top-k，推理保持一致；Switch Transformer 用 top-1

### 延伸阅读
- **原始 MoE**：Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017 — <https://arxiv.org/abs/1701.06538>
- **GShard**：Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, 2020 — <https://arxiv.org/abs/2006.16668>
- **Switch Transformer**：Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2022 — <https://arxiv.org/abs/2101.03961>
- **DeepSeekMoE**：DeepSeek-AI, *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, 2024 — <https://arxiv.org/abs/2401.06066>
- **Survey**：Fedus & Zoph, *A Review of Sparse Expert Models in Deep Learning*, 2022 — <https://arxiv.org/abs/2209.01667>

---

## §7 主流模型架构

> [!quote] 一句话
> 2025-2026 主流 = **Decoder-only + Pre-Norm + RoPE + SwiGLU + GQA/MLA + MoE**；少数 Linear/SSM 路线（Mamba / RWKV / Gated DeltaNet）作为补充。

### 7.1 Dense 系列
- **Llama**：Meta 开源基石，从 Llama 1 到 Llama 4；Llama 3 起 128K 上下文 + GQA；详见 [[llama系列]] / [[Llama 4]]
- **Qwen**：阿里，Qwen3 系列，dense + MoE 双版本，详见 [[Qwen3 系列]] / [[qwen]]
- **Gemma**：Google，2/3 代，SWA 早期代表，详见 [[Gemma 系列]] / [[Gemma]]
- **GLM**：智谱，从 GLM 130B 到 GLM-5（2026 新采用 MLA），详见 [[GLM 系列]]
- **Mistral**：法国 Mistral AI，详见 [[Mistral 系列]]
- **MiniMax M2 / K2 / GPT-OSS / OLMo 2 / SmolLM3 / Xiaomi MiMo / Nemotron 3 / Grok 2.5 / Arcee Trinity Large**：各家 2025-2026 新模型

### 7.2 MoE 系列
- **DeepSeek V3 系列**：MLA + MoE，R1 推理模型基础；详见 [[DeepSeek V3 系列]]
- **Qwen3 MoE**：128 专家（dense 同骨架）
- **Kimi K2**：V3 架构放大版；Kimi Linear = MLA + Gated DeltaNet 混合
- **Llama 4 Maverick**：少而大专家，2 大 FFN
- **Grok 2.5**：8 大专家 + 1 个 SwiGLU "类 shared expert"

### 7.3 Linear / SSM / Hybrid

| 模型 | 主结构 | 复杂度 | 笔记 |
|---|---|---|---|
| **Mamba** | SSM + 选择性扫描 | O(N) 训练 / O(1) decode | （未独立成笔记，见 RWKV 笔记的对比） |
| **RWKV** | RNN 化的 Transformer | O(1) decode | [[rwkv]] |
| **Gated DeltaNet** | Δ-rule + gate | O(1) decode | [[Gated DeltaNet]] |
| **KDA** | Kimi Delta Attention | O(1) decode | [[KDA（Kimi Delta Attention）]] |
| **Hybrid**（Jamba / Kimi Linear） | Attention + SSM 混合 | 折中 | [[Kimi 系列]] |

笔记：[[Transformer]] / [[llama系列]] / [[Qwen3 系列]] / [[DeepSeek V3 系列]] / [[Kimi 系列]] / [[GLM 系列]] / [[Gemma 系列]] / [[Mistral 系列]] / [[rwkv]] / [[Llama 4]] / [[Feb 2026 LLM 新发布]]

### 7.4 训练目标副产物
- **MTP（Multi-Token Prediction）**：DeepSeek V3 引入，每个位置同时预测未来 k 个 token，提升数据效率。详见 [[MTP（Multi-Token Prediction）]]
- **Latent Experts**：Nemotron 3 Super 把专家放到 latent space，详见 [[Latent Experts]]

### 延伸阅读
- **Llama 3**：Dubey et al., *The Llama 3 Herd of Models*, 2024 — <https://arxiv.org/abs/2407.21783>
- **DeepSeek V3**：DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 — <https://arxiv.org/abs/2412.19437>
- **Qwen3**：Qwen Team, *Qwen3 Technical Report*, 2025 — <https://arxiv.org/abs/2505.09388>
- **Mamba**：Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023 — <https://arxiv.org/abs/2312.00752>
- **RWKV-7**：Peng et al., *RWKV-7: Goose with Expressive Dynamic State* — <https://arxiv.org/abs/2503.14456>

---

## §8 预训练 (Pretraining)

> [!quote] 一句话
> Pre-train = **海量文本 + 分布式训练 + 长 context 训练**。LLM 99% 的能力来自 pretraining。

### 8.1 数据工程
- 数据清洗、dedup、quality filter、mixing ratio
- Tokenizer 训练
- 数据配比（CC / 书籍 / 代码 / 论文 / 数学）
- 长文 + 合成数据（Self-Instruct、Magpie、UltraChat）

### 8.2 Scaling Law
- Kaplan & Chinchilla：loss ≈ a · N^α + b · D^β + c
- **Chinchilla 法则**：compute-optimal 时模型参数 N 和数据量 D 平衡点（N ≈ 20D）
- 2025+ 的研究指出：inference scaling、reasoning 的兴起让最优 N 偏向更大

### 8.3 分布式训练

| 并行方式 | 切分维度 | 代表 | 笔记 |
|---|---|---|---|
| **DP（Data Parallel）** | 切 batch | PyTorch DDP | [[data_parallel]] |
| **FSDP / ZeRO** | 切 optimizer state + grad + param | FSDP2、DeepSpeed ZeRO | [[fsdp]] / [[zero]] |
| **TP（Tensor Parallel）** | 切 matmul 的 head / hidden | Megatron | [[tensor_parallel]] |
| **PP（Pipeline Parallel）** | 切层 | Megatron-Pipe | [[pipeline parallelism]] |
| **EP（Expert Parallel）** | 切 MoE 专家 | Megatron、DeepSpeed | （合在 [[zero]] 讨论） |
| **SP（Sequence Parallel）** | 切序列 dim | DeepSpeed Ulysses | [[ulysses_sequence_parallel]] |
| **CP（Context Parallel）** | 长序列切分 | Ring Attention | （待笔记化） |

> [!note]+ ZeRO 三阶段（折叠）
> - **ZeRO-1**：只切 optimizer state（4× 节省）
> - **ZeRO-2**：再切 gradient（8× 节省）
> - **ZeRO-3**：再切 parameter（线性节省）
> 详见 [[zero]]；FSDP 类似思想。

### 8.4 显存 / FLOPs 分析

> [!note]+ 经典结论（折叠）
> - 训练 1 token 的 FLOPs ≈ 6N（forward 2N + backward 4N）
> - 总训练 FLOPs ≈ 6ND（N 参数、D 数据 token 数）
> - Adam 优化器显存 ≈ 16N bytes（N params + 2× 梯度 + 2× optimizer state，fp32）
> - Activation 是 O(BB · S × hidden × L)，可用 activation checkpointing 降到 O(√L)
> - KV cache / Attention 计算见 [[显存占用计算]] / [[flops分析]] / [[transformer参数量分析]]

笔记：[[显存占用计算]] / [[flops分析]] / [[transformer参数量分析]] / [[Activation checkpointing]] / [[梯度累计]] / [[CPU offloading]] / [[remove_padding]]

### 8.5 长上下文
- **RoPE 扩展**：PI / NTK-aware / YaRN / LongRoPE
- **位置插值**：训练时把 max position 缩放
- **Continue training**：从 8K → 128K，~100B tokens 即可

笔记：[[长度扩展（外推）]]（**占位，待完善**）

### 8.6 Pretraining 实战笔记
- [[UltraScale Playbook 笔记]] — HuggingFace Ultra-Scale Playbook 笔记
- [[DeepSeek V4 Infra 4.4 - 训练显存管理|DeepSeek V4 Infra]] — DS V4 训练显存管理
- [[qwen]] — Qwen pretraining 流程
- [[从GPT-2到Kimi-K3]] — 模型规模演进史
- [[Engineering/coding/pai-megatron/快速入门|megatron 入门]] / [[hf2megatron]] / [[training_model]]

### 延伸阅读
- **Chinchilla**：Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022 — <https://arxiv.org/abs/2203.15556>
- **Scaling Law**：Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 — <https://arxiv.org/abs/2001.08361>
- **Megatron-LM**：Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, 2019 — <https://arxiv.org/abs/1909.08053>
- **FSDP**：Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*, 2023 — <https://arxiv.org/abs/2304.11277>
- **ZeRO**：Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2020 — <https://arxiv.org/abs/1910.02054>
- **Ultra-Scale Playbook**（HuggingFace）：<https://huggingface.co/spaces/nanotron/ultrascale-playbook>
- **Megatron 中文解读**：[[Engineering/coding/pai-megatron/ai阅读megatron|ai 阅读 megatron]] / [[cr阅读记录]]

---

## §9 后训练 / 对齐 (Post-training)

> [!quote] 一句话
> Post-training = **SFT + 对齐（RLHF/DPO/RLAIF）**。2024+ 多了**推理强化（RLVR/PRM/MCTS）**——为 reasoning 模型而设计。

### 9.1 流程全景

```text
Pretrained Model
   ↓ SFT (Supervised Fine-Tuning)
   ↓ RLHF (PPO / GRPO 系列)
   ↓ Reasoning RL (RLVR / PRM / Test-Time Scaling)
   ↓
Aligned Reasoning Model
```

笔记：[[SFT、RL 与 OPD：分布视角]]（理论框架）

### 9.2 RLHF 经典算法

| 算法 | 核心 | 评价 | 笔记 |
|---|---|---|---|
| **PPO** | Actor-Critic + GAE + clip surrogate | 最经典；4 个模型同时训练，复杂 | [[ppo]] |
| **GRPO** | 去掉 critic，用组内归一化作为 advantage | DeepSeek-R1 主力；显存友好 | [[grpo]] |
| **RLOO** | REINFORCE Leave-One-Out | 更简单 | [[RLOO]] |
| **REINFORCE++** | REINFORCE 改进，稳定 | 比 GRPO 更简单 | [[REINFORECE++.md]] |
| **ReMax** | REINFORCE + argmax baseline | 简化版 | [[ReMAX（REINFORCE argmax）]] |

### 9.3 GRPO 家族（DAPO / CISPO / SAPO / GVPO）

> [!note]+ DAPO 4 个改动（折叠）
> 1. **Clip-Higher**：`ε_high` 从 0.2 → 0.28，缓解熵坍塌
> 2. **Dynamic Sampling**：过滤全对/全错的 prompt，保证每批有有效梯度
> 3. **Token-Level Loss**：去掉 1/|o|，每个 token 等权（解决长度偏差）
> 4. **Overlong Reward Shaping**：超长回答给 soft penalty
> 详见 [[dapo]]

其它变体：
- **CISPO**：重要性采样裁剪放在 token 梯度上而非 ratio
- **SAPO**：自适应 clip
- **GVPO**：用组方差而非标准差

笔记：[[dapo]] / [[CISPO]] / [[SAPO]] / [[GVPO]] / [[DCPO]]

### 9.4 DPO / SimPO 系列（无 RL 路径）
- **DPO**：直接把 reward 写成策略比 → 不需要 RL；详见 [[OPD]] 仓库整理
- **SimPO**：去掉 reference model，长度
- **ORPO / KTO / IPO**：各种 reward-free 对齐变体

笔记：[[OPD]]

### 9.5 Process Reward Model（PRM）/ MCTS

- **ORM**（Outcome）：整序列一个 reward
- **PRM**（Process）：每个 step 一个 reward → 配合 MCTS 做 step-level search
- **应用**：o1 / R1 类推理模型的核心训练信号

笔记：[[Agent/Reasoning/MCTS和PRM]] / [[Research/Papers/Reasoning/BEYOND CORRECTNESS：HARMONIZING PROCESS AND OUTCOME REWARDS THROUGH RL TRAINING|Beyond Correctness]]

### 9.6 Reasoning RL 论文分类

> [!tip] 推荐阅读入口
> - [[Research/Papers/Reasoning/]] 下有 40+ 篇 Reasoning RL 笔记，按"算法改进 / 长度控制 / 探索 / 数据"分组
> - [[Research/Papers/LLM-RL/]]：通用 RL 后训练方法
> - [[Research/Papers/Agent/agent RL相关论文]]：Agent RL

按主题划分：
- **算法改进**：DAPO、CISPO、SAPO、GVPO、GSPO、GTPO、GRPO-S、EDGE-GRPO、CAPO
- **长度控制**：LAPO（长度自适应）、UloRL（超长输出 RL）、L1 / Think in Blocks、ThinkDial
- **探索**：First Return Entropy-Eliciting、ETTRL、EDGE-GRPO
- **信用分配**：CAPO（verifiable credit assignment）、CARFT
- **数据合成**：R-Zero（self-evolving zero data）、ProRL、TTRL（test-time RL）
- **过程监督**：Good Learners Think Their Thinking、Process Reinforcement

### 9.7 关键八股考点

- [ ] **PPO vs GRPO**：PPO 用 critic 网络预估 V；GRPO 用同 prompt 的 G 个采样作 baseline
- [ ] **GRPO 长度偏差**：因 1/|o| 归一化，答对时倾向更短、答错时倾向更长 → DAPO 用 token-level loss 解决
- [ ] **DPO 的核心**：直接用偏好数据回归策略比，最大似然对齐
- [ ] **K1/K2/K3 KL 估计**：无偏、有偏、k3 无偏估计器（见 [[grpo]] 配图）
- [ ] **on-policy vs off-policy**：GRPO 默认 on-policy；可以 PPO-style 多次迭代（num_iterations > 1）

### 延伸阅读
- **InstructGPT / RLHF 原始**：Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, 2022 — <https://arxiv.org/abs/2203.02155>
- **DPO**：Rafailov et al., *Direct Preference Optimization*, 2023 — <https://arxiv.org/abs/2305.18290>
- **GRPO**：Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024 — <https://arxiv.org/abs/2402.03300>
- **DAPO**：Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025 — <https://arxiv.org/abs/2503.14476>
- **R1**：DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025 — <https://arxiv.org/abs/2501.12948>
- **o1 / Reasoning**：OpenAI, *Learning to Reason with LLMs*, 2024 — <https://openai.com/index/learning-to-reason-with-llms/>
- **Lilian Weng RLHF**：<https://lilianweng.github.io/posts/2024/01/02/dpo/>
- **HuggingFace Alignment Handbook**：<https://github.com/huggingface/alignment-handbook>
- **综述（中文）**：[[Research/Papers/Reasoning/]] 仓库

---

## §10 推理与压缩 (Inference)

> [!quote] 一句话
> 推理优化的目标：**低 TTFT（首 token 延迟）+ 高 TPOT（每 token 延迟）+ 高 throughput**。

### 10.1 Prefill vs Decode
- **Prefill**：一次性处理 prompt，compute-bound
- **Decode**：自回归逐 token，memory-bound
- **PD 分离**：Prefill / Decode 拆到不同 GPU，详见 [[KV cache]]

### 10.2 KV Cache

> [!note]+ 显存估算
> - KV cache 大小 = `2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_elem`
> - 例：Llama 3 70B（int8），128K token → ~21 GB（详见 [[KV cache]]）
> - GQA / MLA 改 n_kv_heads；量化改 bytes_per_elem

笔记：[[KV cache]]

### 10.3 PagedAttention / vLLM
- **PagedAttention**：把 KV cache 按固定 page 大小分块，逻辑块 → 物理块映射
- **Continuous Batching**：每个 decode step 动态加入/退出请求
- **Prefix Caching**：相同 prefix 共享 KV
- **Chunked Prefills**：长 prompt 分块进 Pre-fill
- **Speculative Decoding**：小模型 draft + 大模型 verify，加速 2-4×

笔记：[[paged attention]] / [[vllm高效推理全链路]] / [[nano-vllm阅读笔记]] / [[一些参数]]

### 10.4 采样策略

| 方法 | 思路 | 用途 |
|---|---|---|
| **Greedy** | argmax | 确定性 / 评测 |
| **Beam Search** | 维护 k 个候选 | 翻译 / 摘要 |
| **Top-k** | 截取前 k | 控制多样性 |
| **Top-p (Nucleus)** | 累积概率 > p | 动态截取 |
| **Temperature** | softmax(logits/T) | 控制尖锐度 |
| **Repetition Penalty** | 出现过的 token 降权 | 减少重复 |
| **Frequency / Presence Penalty** | 按频次 / 是否出现 降权 | 同上 |

笔记：[[generate相关]]

### 10.5 量化

| 方法 | 粒度 | 代表 |
|---|---|---|
| **GPTQ** | weight-only, group-wise | GPTQ-for-LLaMA |
| **AWQ** | activation-aware weight | MIT Han Lab |
| **GGUF** | llama.cpp 格式 | llama.cpp / ollama |
| **SmoothQuant** | activation + weight | 推理部署 |
| **KV Cache 量化**（FP8/INT4） | cache 层 | vLLM / TensorRT-LLM |

笔记：（合在 [[vllm高效推理全链路]] / [[一些参数]]）

### 10.6 长度外推
- **PI**（位置插值）：`θ → θ/s`
- **NTK-aware**：放大 RoPE base
- **YaRN**：按频率维度分组插值
- **LongRoPE**：搜索最优 per-dim 缩放
- **ALiBi**：天然友好（无需改动）

笔记：[[长度扩展（外推）]]（**占位，待完善**）

### 10.7 Speculative Decoding
- 大模型 verify 小模型的多个 draft token
- 自投机（Medusa / Lookahead）
- Eagle 系列
- 不改变输出分布

笔记：（合在 §10.3 vLLM 笔记）

### 八股考点
- [ ] KV cache 为什么能省 FLOPs？→ 答：避免每 step 重算历史 token 的 K/V
- [ ] vLLM 比 HuggingFace 快在哪？→ 答：PagedAttention + Continuous Batching
- [ ] Prefill 和 Decode 为什么放不同 GPU？→ 答：两者计算特性不同（compute-bound vs memory-bound），PD 分离可以独立优化

### 延伸阅读
- **vLLM**：Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — <https://arxiv.org/abs/2309.06180>
- **FlashAttention**：Dao et al., *FlashAttention*, NeurIPS 2022 — <https://arxiv.org/abs/2205.14135>
- **Flash-Decoding**：<https://pytorch.org/blog/flash-decoding/>
- **Speculative Decoding**：Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2023 — <https://arxiv.org/abs/2211.17192>
- **Medusa**：Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024 — <https://arxiv.org/abs/2401.10774>
- **Chunked Prefills 解读**：<https://zhuanlan.zhihu.com/p/14689463165>
- **PD 分离 DistServe**：<https://zhuanlan.zhihu.com/p/706761664>

---

## §11 Agent

> [!quote] 一句话
> Anthropic 定义：**Workflow = 预编排的 LLM+tools 流程；Agent = LLM 动态决策的工具使用**。两者的边界在 2024-2026 越来越模糊。

笔记：[[agent概览]]（含 Anthropic Workflow 五种范式：Prompt Chaining / Routing / Parallelization / Orchestrator-workers / Evaluator-optimizer）

### 11.1 Agent 四大能力

| 能力 | 描述 | 代表笔记 |
|---|---|---|
| **Reasoning** | 思维链 / Tree of Thought / MCTS | [[Agent/Reasoning/MCTS和PRM]] / [[思维链压缩]] |
| **Planning** | 分层规划 / 长期目标 | [[Research/Papers/Planning/HIPLAN：Hierarchical Planning for LLM Agents with Adaptive Global-Local Guidance]] |
| **Memory** | 短期 / 长期 / 情景 / 程序 | [[Agent/Memory/memory论文整理]] |
| **Tool Use** | Function call / Search / Code | [[Research/Papers/Agent/Search-R1：Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning]] |

### 11.2 Context Engineering（上下文工程）

> [!note]+ 四件事
> - **写入（Write）**：把信息加进 context
> - **读取（Read）**：从 context 取出
> - **压缩（Compress）**：摘要 / 修剪
> - **隔离（Isolate）**：多 agent / 子 context
> 详见 [[context_engineering]]

笔记：[[context_engineering]]

### 11.3 Memory 体系
- **Mem0**：长时记忆生产级方案，详见 [[Research/Papers/Agent/Mem0：Building Production -Ready AI Agents with Scalable Long-Term Memory]]
- **MemAgent**：RL 训练的 memory agent
- **MemTool** / **Memp** / **ZEP**（时序知识图谱）
- **ReasoningBank**：self-evolving memory

笔记：[[Agent/Memory/memory论文整理]]

### 11.4 Planning
- **Plan-and-Act**：先 plan 后 act
- **HiPLAN** / **HIPLAN**：分层规划
- **PilotRL**：global plan-guided progressive RL
- **Learning When to Plan**：动态分配 test-time compute

笔记：[[Research/Papers/Planning/H2R：Hierarchical Hindsight Reflection for Multi-Task LLM Agents|H2R]] / [[Research/Papers/Planning/agent planning综述|Planning 综述]]

### 11.5 Agent RL
- **Search-R1** / **ReSearch**：LLM 学会用 search
- **Tool-R1** / **ToolRM**：工具调用
- **RAGEN**：多轮 RL
- **MUA-RL** / **SimpleTIR**：多轮交互
- **ZeroSearch**：不需要真实 search engine
- **Agent Lightning**：通用 agent RL 框架
- **RAG-RL** 探索（rStar2-Agent）
- **WebExplorer** / **WebThinker**：web agent

笔记：[[Agent/Reasoning/agentic RL]] / [[Research/Papers/Agent/agent RL相关论文|agent RL 相关论文]]

### 11.6 Multi-Agent
- **AWorld**：动态多 agent
- **Chain-of-Agents**：端到端 multi-agent distillation
- **MAS via Joint Alignment**：多 agent 协同对齐

### 11.7 World Model
- 在 agent 内部维护一个"环境模型"，用来 plan / simulate
- **WebEvolver**：web agent + coevolving world model
- **SIMURA** / **RLVR-World** / **Think in Games**

笔记：[[world_model]] / [[Research/Papers/World-model/]]

### 11.8 八股考点
- [ ] Workflow vs Agent 区别？→ 答：Workflow 路径预定义；Agent 动态决策
- [ ] ReAct vs Plan-and-Act？→ 答：ReAct 边想边做，Plan-and-Act 先完整规划
- [ ] Function calling 是怎么实现的？→ 答：JSON schema → tool dispatch → result 拼回 prompt
- [ ] Tool RL 怎么训？→ 答：reward = 任务成功；GRPO 组内归一化

### 延伸阅读
- **Anthropic** *Building Effective Agents* — <https://www.anthropic.com/engineering/building-effective-agents>
- **Anthropic** *How we built our multi-agent research system* — <https://www.anthropic.com/engineering/built-multi-agent-research-system>
- **Chip Huyen** *Agents* — <https://huyenchip.com/2025/01/07/agents.html>
- **Phil Schmid** *Zero to One: Learning Agentic Patterns* — <https://www.philschmid.de/agentic-pattern>
- **Lilian Weng** *LLM-powered Autonomous Agents* — <https://lilianweng.github.io/posts/2023/06/23/agent/>
- **ReAct**：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 — <https://arxiv.org/abs/2210.03629>
- **Toolformer**：Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, 2023 — <https://arxiv.org/abs/2302.04761>

---

## §12 VLM（多模态）

> [!quote] 一句话
> VLM = **Vision Encoder（ViT / SigLIP）+ Projector + LLM**。视觉 patch 编码 → 投影到 LLM token 空间 → LLM 处理多模态序列。

笔记：[[patch详解]]

### 关键组件
- **Vision Encoder**：ViT / SigLIP / ConvNeXt
- **Patch 切分**：16×16 / 14×14 / dynamic patch
- **Projector**：MLP / Q-Former / Cross-Attention
- **LLM**：通常沿用 text LLM
- **训练**：pretrain alignment（frozen LLM）+ SFT + RLHF

### 代表模型
- **LLaVA / LLaVA-OV**：MLP projector
- **InternVL** / **Qwen-VL** / **MiniMax-VL**
- **GPT-4V / GPT-4o** / **Gemini**

笔记：[[patch详解]] / [[LLM Architecture/VLM/]] / [[Engineering/coding/pai-megatron/llavaov_1_5|llava-ov 笔记]] / [[qwen3vl+deepseek]]

### 延伸阅读
- **LLaVA**：Liu et al., *Visual Instruction Tuning*, 2023 — <https://arxiv.org/abs/2304.08485>
- **Qwen-VL**：Bai et al., *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Reading and More*, 2023 — <https://arxiv.org/abs/2308.12926>

---

## §13 工程实践 (Engineering)

> [!quote] 一句话
> 工业 LLM 训练 / 推理 = **分布式框架 + 内存优化 + 通信库 + 编译加速** 的组合。

### 13.1 训练框架

| 框架 | 用途 | 笔记 |
|---|---|---|
| **Megatron-LM** | NVIDIA 大模型训练 | [[Engineering/coding/pai-megatron/ai阅读megatron]] |
| **DeepSpeed** | Microsoft ZeRO + 优化 | [[zero]] / [[fsdp]] |
| **FSDP2 (PyTorch)** | 原生 FSDP | [[fsdp]] |
| **verl** | RLHF / Agent RL 训练 | [[verl总体概览]] / [[veRL_Ray执行流程源码解析]] |
| **trl** | HuggingFace RLHF | [[grpo]] |
| **OpenRLHF** | 分布式 PPO | [[ppo]] |
| **LLaMA Factory** | 一站式 SFT/RLHF | （合在 [[verl]] 笔记） |

### 13.2 推理引擎

| 引擎 | 特点 | 笔记 |
|---|---|---|
| **vLLM** | PagedAttention + Continuous Batching | [[vllm高效推理全链路]] / [[nano-vllm阅读笔记]] |
| **SGLang** | RadixAttention + 结构化生成 | （无独立笔记） |
| **TensorRT-LLM** | NVIDIA 编译优化 | （无独立笔记） |
| **TGI**（Text Generation Inference） | HuggingFace 生产部署 | （无独立笔记） |

### 13.3 训练 vs 推理 Engine 不一致问题

> [!warning] 训练引擎 vs 推理引擎 log_prob 对不齐
> 同一段序列，推理引擎（vLLM/SGLang）的 log_prob 与训练引擎（HF / FSDP）的 log_prob 会**有细微差异**。原因：
> - 推理要快，训练要准
> - kernel 实现差异、并行策略差异、attention 实现差异
> - **解法**：在 GRPO 等 RL 算法里额外 forward 一次算 old_log_prob（详见 [[grpo]]）

### 13.4 verl 重点

verl = Volcano Engine RL，是字节开源的 LLM RL 训练框架，深度集成 Ray + vLLM。

笔记：[[verl总体概览]] / [[veRL_Ray执行流程源码解析]] / [[DataProto]] / [[agent_loop]] / [[reward_mananger]] / [[batch_size解释]] / [[init_workers详解]] / [[verl V1 主线重构]] / [[verl总体概览]] / [[veRL_colocate实现源码解析]] / [[Dataset]] / [[metrics总结]] / [[ray前置知识]]

### 13.5 八股考点
- [ ] ZeRO-1/2/3 区别？→ 答：见 §8.3
- [ ] DP / TP / PP 怎么组合？→ 答：DP 外层、TP+PP 内层（N×P×D 网格）
- [ ] vLLM 为什么吞吐高？→ 答：PagedAttention + Continuous Batching
- [ ] FSDP 和 DDP 区别？→ 答：DDP 每卡一份完整参数 + 同步梯度；FSDP 切分参数 + 通信

### 延伸阅读
- **The Ultra-Scale Playbook**（HF）：<https://huggingface.co/spaces/nanotron/ultrascale-playbook>（笔记 [[UltraScale Playbook 笔记]]）
- **PyTorch FSDP**：<https://pytorch.org/docs/stable/fsdp.html>
- **DeepSpeed**：<https://www.deepspeed.ai/>
- **vLLM docs**：<https://docs.vllm.ai/>
- **verl**：<https://verl.readthedocs.io/>

---

## §14 速查表（一句话定义）

> [!success] 30 秒自检清单
> 用这些定义检查自己能不能在 30 秒内讲清楚：

| 概念 | 一句话定义 | 笔记 |
|---|---|---|
| **Self-Attention** | Q·Kᵀ/√d → softmax → ·V | [[MHA]] |
| **Multi-Head Attention** | 把 Q/K/V 拆 h 头分别 attention 后 concat | [[MHA]] |
| **MQA** | 所有 Q head 共享 1 个 KV head | [[MQA]] |
| **GQA** | Q head 分组共享 KV head（MHA/MQA 之间） | [[GQA]] |
| **MLA** | KV 压到低维 latent，差异信息吸收到 Q | [[MLA]] |
| **FlashAttention** | tiling + online softmax，IO-aware exact attention | [[flash attention]] |
| **PagedAttention** | 像 OS 虚拟内存分页一样管理 KV cache | [[paged attention]] |
| **RoPE** | 复数旋转注入位置，attention 内积天然含相对位置 | [[rope]] |
| **RMSNorm** | 只做 RMS 不做 mean-center | [[Layer Norm]] |
| **BPE** | 反复合并最高频相邻子词对 | [[tokenization]] |
| **MoE** | 多 FFN 专家 + 路由器，每 token 选少量激活 | [[MoE（Mixture of Experts）]] |
| **KV Cache** | decode 阶段缓存历史 K/V，避免重算 | [[KV cache]] |
| **Speculative Decoding** | 小模型 draft + 大模型 verify | [[vllm高效推理全链路]] |
| **量化** | FP32 → INT8/INT4 节省显存 | §10.5 |
| **Prefix Cache** | 相同 prefix 共享 KV 块 | [[vllm高效推理全链路]] |
| **RLHF** | 用人类偏好训 reward model，再用 PPO 优化策略 | [[ppo]] |
| **GRPO** | 去掉 critic，用同 prompt G 个采样作 baseline | [[grpo]] |
| **DPO** | 直接回归策略比，无需 reward model + RL | [[OPD]] |
| **PRM** | 每 step 一个 reward，配合 MCTS 做 step-level search | [[Agent/Reasoning/MCTS和PRM]] |
| **FSDP** | 切分参数 + 梯度 + optimizer state 跨卡 | [[fsdp]] |
| **ZeRO** | DeepSpeed 的参数切分方案（1/2/3 阶段） | [[zero]] |
| **TP / PP / DP / EP** | 切 matmul / 切层 / 切 batch / 切专家 | [[tensor_parallel]] / [[pipeline parallelism]] / [[data_parallel]] |
| **Workflow vs Agent** | 预编排 vs LLM 决策的工具调用 | [[agent概览]] |
| **ReAct** | Reasoning + Acting 交替 | [[Research/Papers/Agent/]] |

---

## §15 外部参考（汇总）

### 综述 / 大图
- Sebastian Raschka, *The Big LLM Architecture Comparison* — <https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison>
- Sebastian Raschka, *Build a Large Language Model (From Scratch)* — <https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167>
- Lilian Weng 博客（[Agents](https://lilianweng.github.io/posts/2023/06/23/agent/) / [RAG](https://lilianweng.github.io/posts/2024/07/07/the-illusion-of-thinking/) / [DPO](https://lilianweng.github.io/posts/2024/01/02/dpo/)）
- Anthropic *Building Effective Agents* — <https://www.anthropic.com/engineering/building-effective-agents>
- HuggingFace *NLP Course* — <https://huggingface.co/learn/nlp-course>

### 必读论文（按主题）

| 主题 | 论文 |
|---|---|
| **Transformer 起源** | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **位置编码** | [RoPE](https://arxiv.org/abs/2104.09864), [ALiBi](https://arxiv.org/abs/2108.12409), [YaRN](https://arxiv.org/abs/2309.00071) |
| **Attention 优化** | [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention](https://arxiv.org/abs/2309.06180), [GQA](https://arxiv.org/abs/2305.13245) |
| **高效架构** | [MoE 原始](https://arxiv.org/abs/1701.06538), [Switch Transformer](https://arxiv.org/abs/2101.03961), [GShard](https://arxiv.org/abs/2006.16668), [DeepSeekMoE](https://arxiv.org/abs/2401.06066) |
| **MLA** | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| **Linear/SSM** | [Mamba](https://arxiv.org/abs/2312.00752), [RWKV-7](https://arxiv.org/abs/2503.14456) |
| **Scaling Law** | [Chinchilla](https://arxiv.org/abs/2203.15556), [Kaplan Scaling](https://arxiv.org/abs/2001.08361) |
| **分布式** | [Megatron-LM](https://arxiv.org/abs/1909.08053), [ZeRO](https://arxiv.org/abs/1910.02054), [FSDP](https://arxiv.org/abs/2304.11277) |
| **RLHF / DPO** | [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290) |
| **RL 后训练** | [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |
| **Agent** | [ReAct](https://arxiv.org/abs/2210.03629), [Toolformer](https://arxiv.org/abs/2302.04761), [Mem0](https://arxiv.org/abs/2504.19413) |
| **VLM** | [LLaVA](https://arxiv.org/abs/2304.08485), [Qwen-VL](https://arxiv.org/abs/2308.12926) |

### 视频 / 课程
- **3Blue1Brown** Transformers 系列 — <https://www.3blue1brown.com/topics/neural-networks>
- **Andrej Karpathy** *Let's reproduce GPT-2* — <https://www.youtube.com/watch?v=l8pRSuU81PU>
- **HuggingFace** *NLP Course* — <https://huggingface.co/learn/nlp-course>
- **李沐** *D2L + 论文精读* 系列（B 站）

### 中文资料
- **苏剑林** *科学空间*（苏神博客） — <https://kexue.fm/>
- **July_v** CSDN 位置编码综述 — <https://blog.csdn.net/v_JULY_v/article/details/134085503>
- **知乎 PPO / GRPO 专栏**
- **机智流** 推理系列
- **Vlon** vLLM 解析 — <https://zhuanlan.zhihu.com/p/706761664>

### 模型仓库 / Blog
- **Sebastian Raschka** *Ahead of AI* — <https://magazine.sebastianraschka.com/>
- **Lilian Weng** — <https://lilianweng.github.io/>
- **Jay Alammar** *Illustrated* 系列 — <https://jalammar.github.io/>
- **Chip Huyen** *AI Blog* — <https://huyenchip.com/>

### 工程资源
- **HF Ultra-Scale Playbook** — <https://huggingface.co/spaces/nanotron/ultrascale-playbook>
- **PyTorch 官方教程** — <https://pytorch.org/tutorials/>
- **vLLM 文档** — <https://docs.vllm.ai/>
- **verl 文档** — <https://verl.readthedocs.io/>
- **HuggingFace TRL** — <https://huggingface.co/docs/trl>
- **OpenRLHF** — <https://github.com/OpenRLHF/OpenRLHF>

---

## §16 关联笔记

- **入口** [[面经_2026]] — 2026 算法岗面试手册（本文是其八股导航子模块）
- **基础 ML/DL** [[Foundation/机器学习/机器学习的一些概念]] / [[Foundation/深度学习/优化/AdamW|AdamW]] 等
- **代码实现** [[Transformer]] / [[grpo]] / [[ppo]] / [[vllm高效推理全链路]]
- **研究论文** [[Research/Papers/]]（按 Agent / Reasoning / LLM-RL / Planning / World-model / Transformer 分组）

---

## §17 TODO / 待完善

> [!warning] 占位笔记
> 以下相关章节的笔记**尚未完善**，对应本文里的链接点开是 TODO 状态：
> - [[长度扩展（外推）]] — §8.5 / §10.6
> - [[context_engineering]] — §11.2
> - 其余 MLA / MoE / DAPO 等关键笔记均已完善

> [!tip] 如何扩展本文
> 1. 新增主题章节：模仿 §1-§13 的格式（Quote + 八股考点 + 延伸阅读 + wikilink）
> 2. 在 §14 速查表补一行新概念
> 3. 在 §15 外部参考补新论文链接

---

> [!quote] 阅读顺序建议
> 1. 第一次读：§0 + §1 + §14（30 分钟）
> 2. 第二次读：§2-§7（架构）+ §10（推理）（2 小时）
> 3. 第三次读：§8-§9（训练）（2 小时）
> 4. 按方向选读：Agent 走 §11、VLM 走 §12、工程走 §13（各 1 小时）
> **总计 1-2 天可覆盖一轮；考前 1 周刷一次速查表 + 八股考点即可。**
