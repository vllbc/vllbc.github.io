# 从 GPT-2 到 Kimi K3：大模型架构如何演进


> [!info] 来源
> 整理自 ali（[@waterloo_intern](https://x.com/waterloo_intern)）于 2026-07-27 发布的 X Article：[22580: From GPT2 to Kimi3, Explained](https://x.com/waterloo_intern/status/2081762065392541951)。本文不是逐句翻译，而是对原文论证主线的中文重组。

## 一句话总结

从 GPT-2 到 Kimi K3，不只是把参数量扩大了 22580 倍。更关键的演进是：模型不断重新设计**信息如何存储、旧信息如何被更新或遗忘，以及需要的信息如何被检索**。

Kimi K3 最终组合了四种互补能力：

1. **KDA**：以固定大小的循环状态保存长序列信息；
2. **周期性 MLA**：保留对 token 上下文的精确 softmax 检索；
3. **稀疏 MoE**：增加参数容量，但避免每个 token 激活全部参数；
4. **AttnRes**：让深层网络选择性读取较早层的表示。

## 文章主线

```text
GPT-2 / Softmax Attention
        ↓ KV cache 随序列增长，访存开销大
Linear Attention
        ↓ 固定状态高效，但信息会相互干扰
DeltaNet
        ↓ 能精确替换关联，但缺少全局遗忘
Gated DeltaNet
        ↓ 增加可学习衰减，但控制仍较粗
Kimi Delta Attention（KDA）
        ↓ 逐通道精细控制记忆衰减
Kimi K3
        = KDA + 周期性 MLA + Latent MoE + AttnRes
```

## 1. GPT-2：标准 Decoder-only Transformer

GPT-2 的输入由 token embedding 和 positional embedding 相加得到。每个 Transformer block 包含 self-attention 和 MLP，并通过残差连接组织起来。最后的 hidden state 经 language-model head 映射为词表 logits。

自回归生成时，每一步只用最后一个位置的 logits 选择下一个 token，但模型需要利用此前全部 token。若每次都重新计算历史 token 的 K、V 投影，会产生大量重复工作，因此引入 **KV cache** 保存历史 token 的 key 和 value。

KV cache 解决了重复计算，却带来新的问题：

- cache 容量随序列长度 $N$ 线性增长；
- decode 每一步都需要从 HBM 读取历史 K、V；
- 长上下文下容易形成显存容量与内存带宽瓶颈。

原文以约 124M 参数的 GPT-2 为基准：词表约 50k、12 层、12 个注意力头、hidden size 为 768。若按 Kimi K3 的 2.8T 总参数计算，一个 Kimi K3 约等于 22580 个 GPT-2 的参数量。

> 22580 这个数字揭示的是尺度差异，但全文真正讨论的是：这些新增容量被放在了哪里，以及它们解决了什么问题。

## 2. Linear Attention：把增长的 KV cache 压缩为固定状态

标准 softmax attention 可以简写为：

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^T)V
$$

softmax 作用在 $QK^T$ 之后，因此不能简单改变矩阵乘法顺序。Linear Attention 对 $Q$、$K$ 分别应用非负特征映射 $\phi$，例如 $\mathrm{ELU}(x)+1$：

$$
\operatorname{Attn}(Q,K,V)
\approx
\frac{\phi(Q)(\phi(K)^TV)}{\phi(Q)(\phi(K)^T\mathbf{1})}
$$

利用乘法结合律，可以先维护固定大小的状态：

$$
S_t=S_{t-1}+\phi(k_t)^Tv_t
$$

随后用 query 从状态中读取：

$$
o_t=\phi(q_t)S_t
$$

这样，历史 K、V 不再沿 token 维不断拼接，而被折叠进一个固定的 $D\times D$ 状态。其优势是 decode 状态不随序列长度增长。

代价是：$\phi$ 对 softmax kernel 的表达能力有限，因此可能损失精确性。Linear Attention 保留了“非负打分—归一化—对 value 加权平均”的基本契约，但使用了更容易重排计算、也更弱的特征映射。

## 3. DeltaNet：固定容量记忆需要覆盖策略

Linear Attention 的核心状态是有限容量的关联记忆。若不断执行加法更新：

$$
S_t=S_{t-1}+k_t^Tv_t
$$

不同 key-value 关联会被写入同一个矩阵。当序列长度远大于状态容量时，关联会互相干扰，而且旧信息从未被移除。

Delta rule 的思路不是直接叠加新 value，而是先读取当前 key 已经指向的内容，再只写入差值：

$$
v_{old}=k_tS_{t-1}
$$

$$
u_t=\beta_t(v_t-v_{old})
$$

$$
S_t=S_{t-1}+k_t^Tu_t
$$

其中 $\beta_t$ 控制写入强度。直观上，这相当于：

1. 在 key 指向的位置读取旧关联；
2. 计算目标 value 与旧 value 的差；
3. 删除旧信息并写入新信息。

因此，DeltaNet 能对某个特定关联做更精确的替换，而不是让信息无止境叠加。

### Chunk-wise 并行

Delta rule 看起来是严格递归的：第 $t$ 步依赖第 $t-1$ 步状态，不利于 GPU prefill。DeltaNet 通过数学重参数化和 chunk-wise 算法提高并行性：

- 序列被切成大小为 $C$ 的 chunk；
- chunk 内采用类似 masked attention 的并行矩阵运算；
- chunk 间通过固定状态递归传递信息。

两个极端是：

- $C=N$：退化为标准 $O(N^2)$ attention；
- $C=1$：退化为逐 token 的 linear attention。

中间的 $C$ 在 FLOPs 与 GPU 矩阵乘法效率之间折中。原文指出实践中常用 64 或 128，因为这类粒度更适合 tensor core。$C=1$ 的 FLOPs 最少，但 wall-clock time 不一定最好。

## 4. Gated DeltaNet：在精确替换之外加入遗忘

Delta rule 能替换某个 key 对应的关联，却不擅长：

- 上下文切换时一次性清除多条旧关联；
- 让整个记忆随时间衰减；
- 主动释放有限状态中的容量。

Mamba 类更新提供了整体衰减思路：

$$
S_t=\alpha_tS_{t-1}+k_t^Tv_t
$$

其中 $0\leq\alpha_t\leq1$：

- $\alpha_t\approx1$：保留旧状态；
- $\alpha_t\approx0$：清空旧状态。

但统一的标量衰减会等比例遗忘所有关联，无法区分不同信息的重要性。Gated DeltaNet 因而将两者结合：

- Delta rule 负责定向替换某一关联；
- gate 负责让旧状态自适应衰减。

其本质是同时具备“写什么”和“忘多少”的能力。

## 5. Kimi Delta Attention：从标量遗忘到逐通道遗忘

Kimi Linear 的关键改进是将粗粒度标量 gate 变为更细粒度的逐通道 decay。也就是说，不再让整个状态按同一比例遗忘，而是让每个 channel 学习不同的衰减值。

这给固定容量记忆增加了更有针对性的容量：不同通道可以保留不同时间尺度、不同重要程度的信息。

原文将 Kimi Linear 相对 DeltaNet Transformer 的变化概括为：

1. 使用 KDA 与 **Multi-head Latent Attention（MLA）** 交错的混合结构；
2. 以 **Mixture-of-Experts（MoE）** 替代普通 MLP；
3. 通过 alpha projection 为 DeltaNet 增加逐通道记忆控制能力。

这里的扩容不是盲目堆参数，而是把容量放到有明确功能的位置：提高对记忆衰减的控制精度。

## 6. Kimi K3：混合记忆、检索与稀疏容量

据原文介绍，Kimi K3 的 language backbone 包含 23 个四层 macrocycle。每个 macrocycle 中：

- 前 3 层使用 Kimi Delta Attention；
- 第 4 层使用 Multi-head Latent Attention；
- 第 1 层使用 dense FFN；
- 其余层使用 latent MoE。

相对 Kimi Linear，Kimi K3 还加入或强化了：

- 更大规模；
- 每 12 层一次的 blockwise AttnRes；
- MLA query LoRA 与 output gating；
- latent-space MoE；
- SiTU activation；
- Gated MLA。

### 6.1 KDA 与 MLA 为什么共存

两种机制承担不同角色：

- **KDA** 将历史压缩到固定大小状态，decode 高效，但必然丢失部分信息；
- **MLA** 保留基于 softmax 的 token 级检索，可以精确访问上下文，但成本随上下文增长。

因此，Kimi K3 不是完全用 linear attention 替代 softmax attention，而是采用混合结构：大部分层使用常量状态的 KDA，周期性插入 MLA 修复精确检索能力。

### 6.2 Latent MoE

传统 MoE 用 router 将 token 分配给少数专家。原文给出的 Kimi K3 配置为：

- 共 898 个专家；
- 2 个 shared experts 处理所有 token；
- 其余 896 个专家中，每个 token 选择 16 个。

MoE 增加总参数容量，但每个 token 只激活一小部分专家，从而避免计算量与总参数量同比增长。

Kimi K3 还让专家在压缩后的 latent space 中工作。原文称这种设计接近将 FLOPs 减半，并抵消新 activation 带来的部分 kernel 开销。

### 6.3 SiTU 与工程实现

Kimi K3 用 SiTU 替换传统 SwiGLU 路径中的激活方式。原文强调了一个重要的系统事实：**架构上的数学改进不等于实际推理更快**。

如果没有 fused kernel，新 activation 路径可能接近原路径的 3 倍慢；因此模型架构必须与 kernel 实现共同设计。latent-space experts 的计算压缩，就是抵消额外操作的一种方式。

## 7. AttnRes：沿网络深度选择性读取历史表示

普通残差流可以写成：

$$
h_l=h_1+\sum_{i=1}^{l-1}f_i(h_i)
$$

其中所有前层输出以同等权重累加。这样有两个问题：

1. 不同层可能需要不同类型的早期表示，但无法选择；
2. 随深度增加，残差不断累积，后层需要产生更大的输出才能影响整体状态，可能造成 residual dilution、hidden-state growth 和训练不稳定。

AttnRes 为每项增加可学习权重：

$$
h_l=\alpha_0h_1+\sum_{i=1}^{l-1}\alpha_if_i(h_i)
$$

权重 $\alpha_i$ 由 query-key 点积得到，并经归一化后形成对早期 residual states 的加权组合。因此当前层不必只依赖直接前驱，而能选择性检索更早、更有用的深度表示。

Kimi K3 并非每层执行一次 AttnRes，而是在固定 block 边界应用。原文描述为每 12 层一次；23 个四层 macrocycle 共 92 层，对应 8 个 block 表示。这样在控制开销的同时，保留大部分收益。

原文给出的结果是：AttnRes 约增加 2% 推理延迟，同时带来：

- 对早期表示的选择性检索，缓解 residual dilution 与 hidden-state growth；
- 约 1.25 倍 compute advantage。

AttnRes 与 MLA 解决的是同一类问题的两个方向：

- MLA 沿**序列维度**检索此前 token；
- AttnRes 沿**网络深度**检索此前层表示。

## 8. 如何理解整条演进链

| 架构 | 主要收益 | 暴露的问题 | 下一步改进 |
| --- | --- | --- | --- |
| Softmax Attention + KV cache | token 级精确检索 | KV cache 随 $N$ 增长，访存成本高 | 固定状态 |
| Linear Attention | 常量大小循环状态 | 加法写入导致关联干扰 | Delta rule |
| DeltaNet | 精确替换指定关联 | 缺少整体遗忘和上下文切换能力 | Gate / decay |
| Gated DeltaNet | 写入与整体衰减结合 | 标量衰减过于粗粒度 | 逐通道 decay |
| KDA / Kimi Linear | 精细记忆管理 | 固定状态仍会丢信息 | 周期性 MLA |
| Kimi K3 | 混合检索、稀疏容量与深度检索 | 系统复杂度与 kernel 实现要求提高 | 软硬件协同优化 |

## 核心洞见

### 1. 固定容量记忆必须有淘汰策略

只要状态维度固定，容量就一定有限。纯加法更新迟早会使不同关联互相干扰。因此模型必须学习：

- 哪些信息值得保留；
- 哪些关联应该替换；
- 何时需要衰减或清空；
- 固定状态丢失的信息应从哪里重新检索。

这也是 gate、routing、decay 和 attention 反复出现的共同原因。

### 2. Attention 的核心价值是选择性读取

attention 不只是一种 $O(N^2)$ 算子，更重要的是它提供了内容寻址的 selective read：根据当前 query 选择与之相关的信息。

Kimi K3 在三个维度使用了“选择”：

- KDA 的 gate 选择保留或忘记哪些状态；
- MoE router 选择激活哪些专家；
- MLA 与 AttnRes 分别在 token 维和 depth 维选择读取哪些表示。

### 3. Scaling 仍然重要，但容量必须有功能位置

22580 倍参数差距并不能单独解释架构进步。原文的主张是：每一次扩容都应该针对前一代系统的明确瓶颈，例如：

- 用状态矩阵替代无限增长的 cache；
- 用 Delta rule 修复关联覆盖问题；
- 用 channel-wise decay 提升记忆控制精度；
- 用 MoE 增加稀疏容量；
- 用 MLA 和 AttnRes 补回固定状态无法保存的信息。

### 4. 理论 FLOPs 最低不代表实际最快

chunk size、tensor core 粒度、kernel fusion、HBM 访问和中间张量布局都会影响实际性能。因此理解现代模型不能只看公式，还要同时看：

- prefill 与 decode 的不同瓶颈；
- 计算复杂度与访存复杂度；
- GPU 是否能高效执行该算子；
- 新结构是否有成熟 fused kernel。

## 我的归纳

可以把 Kimi K3 看成一个多层级记忆系统：

- **KDA 是工作记忆**：容量固定、更新快，适合大多数 token；
- **MLA 是上下文档案检索**：成本更高，但能精确回看原始 token；
- **AttnRes 是深度记忆检索**：从早期网络层恢复被后续残差稀释的表示；
- **MoE 是知识容量池**：按需激活不同参数子集。

这解释了为什么最终方案是混合架构，而不是让某一种 attention 完全取代其他机制：不同机制在容量、精度、计算和访存之间有不同的 Pareto 前沿。

## 阅读时需要注意

- 原文是一篇个人 worklog，包含作者基于论文与实现的解释，不等同于 Kimi 官方技术报告。
- “2.8T”“898 experts”“AttnRes 开销约 2%”“1.25x compute advantage”等数字均按原文记录；严谨引用时应继续回溯 Kimi K3 官方报告或代码。
- 原文包含大量示意图和实现片段。本笔记保留了概念与关键公式，没有逐张搬运图片或逐段复制代码。
- 原文标题写作 `GPT2`、`Kimi3`，正文写作 `GPT-2`、`KimiK3`；本文统一记为 GPT-2 与 Kimi K3。

## 参考

- [ali：22580: From GPT2 to Kimi3, Explained](https://x.com/waterloo_intern/status/2081762065392541951)
- [ali 的 X 主页](https://x.com/waterloo_intern)

