# The Ultra-Scale Playbook 学习笔记


> 来源：[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（Hugging Face nanotron 团队，2025）
>
> BibTeX: Tazi et al., "The Ultra-Scale Playbook: Training LLMs on GPU Clusters", 2025.
>
> 配套仓库：[nanotron](https://github.com/huggingface/nanotron)（生产级）、[picotron](https://github.com/huggingface/picotron)（教学级）
>
> 原书基于 **4100+ 组分布式实验、16k+ 测试运行** 的实测数据写成，是当前最系统的分布式 LLM 训练手册。本文按主线整理：先把"为什么会爆显存"的代价讲清楚，再讲 5 种并行 + ZeRO 怎么对付这些代价，最后给出选型决策三步法。所有公式、表格、决策规则均保留原文细节。

## 01 高层概览：三大挑战与权衡

整本书的所有技术都围绕三个挑战展开：

1. **显存使用（硬约束）**：单步放不下，训练直接挂。
2. **计算效率**：让硬件尽量在算，而不是在等数据 / 等别的 GPU。
3. **通信开销**：通信是 GPU 闲置的根源。要分别利用好 node 内（NVLink，快）和 node 间（IB/Ethernet，慢）的带宽，并尽量把通信**与计算 overlap**。

核心 insight：这三者可以**互相交换**——比如 activation recomputation 用算力换显存，TP 用显存换通信。**找平衡** 是分布式训练的关键。

## 02 单卡训练基础

### Batch Size 的三种单位

LLM 社区普遍按 token 报 batch size（而不是 sample 数）：

- `bst = bs * seq`（token 总数）
- `bs = gbs = mbs × grad_acc`（单卡或单节点）
- `bs = gbs = mbs × grad_acc × dp`（分布式）

近期 LLM 训练的最佳 batch size 区间是 **4M – 60M tokens / step**。Llama 1 用 ~4M tokens 训 1.4T tokens；DeepSeek-V3 用 ~60M tokens 训 14T tokens。DeepSeek-V3 还做了 batch 渐进：从 3072 序列 → 15360 序列（前 469B tokens），之后保持 15360。

### 显存占用四大件

训练时显存里有四样东西：

| 项目 | 说明 |
|---|---|
| 模型权重 | 推理、backward 都要用 |
| 梯度 | backward 算出 |
| 优化器状态 | Adam 至少要存 momentum + variance |
| 激活值 | forward 留的中间张量，backward 要用 |

额外开销：CUDA kernel 通常占 1-2 GB（`torch.ones((1,1)).to("cuda")` 后用 `nvidia-smi` 可见），加上 buffer / fragmentation，可忽略但要心里有数。

### 权重 / 梯度 / 优化器状态

Transformer 参数总量公式（不含 positional embedding）：

$$
N = h \cdot v + L \cdot (12 h^2 + 13 h) + 2h
$$

- $h$ = hidden dim，$v$ = vocab size，$L$ = 层数
- 主导项是 $h^2$，所以模型做大时 $h^2$ 先爆

**FP32 全精度**训练：

$$
m_{params} = 4N,\quad m_{grad} = 4N,\quad m_{opt} = (4+4)N
$$

**BF16 混合精度**训练（默认）：

$$
m_{params} = 2N,\quad m_{grad} = 2N,\quad m_{params\_fp32} = 4N,\quad m_{opt} = (4+4)N
$$

注意：`m_{params_fp32}` 是所谓的 **master weights**。有些库（如 nanotron）把梯度也用 FP32 累加，牺牲内存换数值稳定。

直观感受：

| 模型规模 | FP32 或 BF16 w/o FP32 grad | BF16 w/ FP32 grad |
|---|---|---|
| 1B | 16 GB | 20 GB |
| 7B | 112 GB | 140 GB |
| 70B | 1120 GB | 1400 GB |
| 405B | 6480 GB | 8100 GB |

> **7B 模型就已经超过单张 H100（80 GB）了**——这是后面所有分布式方案的动机。

### Activation 显存

每个 transformer layer 的 activation 显存（mixed precision）：

$$
m_{act} = L \cdot seq \cdot bs \cdot h \cdot \left(34 + \frac{5 \cdot n_{heads} \cdot seq}{h}\right)
$$

- 随 $bs$ **线性** 增长，随 $seq$ **二次** 增长
- **长序列 / 大 batch 下，activation 是最大头**——比 weights / grads / optimizer state 加起来还多

实测观察（Llama 系，$bs=1$）：短序列（≤2K）activation 可忽略；到 4K+ 开始显著。

### Activation Recomputation（gradient checkpointing）

基本思路：forward 时丢一些 activation，backward 时再算回来——**用算力换显存**。

| 策略 | 做法 | 代价 | 收益 |
|---|---|---|---|
| **Full** | 每个 transformer layer 边界处 checkpoint | +30-40% compute | 最大显存节省 |
| **Selective** | 丢 attention（重算便宜），checkpoint FFN | GPT-3 175B：**70% activation 显存减少，仅 2.7% compute** | 几乎免费 |

DeepSeek-V3 还把 attention 激活本身也压得更小（**MLA，Multi-Head Latent Attention**），进一步减少选择性 checkpoint 要存的量。

两个易混概念：
- **HFU (Hardware FLOPS Utilization)**：含 recomputation 的实际 FLOPs 算利用率
- **MFU (Model FLOPS Utilization)**：只算 forward + backward 必要的 FLOPs 算利用率——更反映**模型本身的硬件利用**

跨硬件对比时更推荐 MFU（如果某卡显存大到能省去 recomputation，那它的 HFU 低但训练更快是合理的）。

现代训练基本默认用 **Flash Attention**，它原生把 attention 矩阵的 S / P 用 online softmax 不落地，所以绝大多数用户已经在享用 selective recomputation。

### Gradient Accumulation

把全局 batch 切成 micro-batch，依次做前向 / 反向，**累积梯度** 再做一步 optimizer：

$$
bs = gbs = mbs \times grad\_acc
$$

关键 insight：求和 / 求平均是等价的（按 $grad\_acc$ 数除即可），所以 GBS 一定时训练动态完全一致。Gradient accumulation **不改变训练，只解决显存**——是分块处理大 batch 的最朴素办法。

## 03 Data Parallelism（DP）

把模型复制到 $N$ 个 GPU 上，每个 GPU 跑不同 micro-batch（不同数据切片）。最朴素 DP 的通信是 backward 后做一次 all-reduce 同步梯度。

### 三个 DP 优化

**优化 1：把梯度同步与 backward overlap**

用 `register_post_accumulate_grad_hook` 在每个参数上注册 hook，**该参数的梯度一算完就立刻启动 all-reduce**——而不是等全部 backward 完再统一同步。bucket 内是连续触发。

**优化 2：梯度 bucketing**

GPU 操作对**大 tensor**更友好。把参数按 bucket（如默认 25 MB）分组，每个 bucket 内部触发一次 all-reduce，减少 kernel launch 次数。

**优化 3：与 Gradient Accumulation 的配合**

只在**真正需要 reduce 的那次** backward 上同步，其余 micro-step 用 `model.no_sync()` 跳过通信：

$$
bs = gbs = mbs \times grad\_acc \times dp
$$

### DP 的局限

每个 DP rank **仍然要存一整套** optimizer state、grad、param。复制造成显存冗余——这就是 ZeRO 出场的动机。

## 04 ZeRO（Zero Redundancy Optimizer）

沿 DP 轴把 optimizer state、grad、param **分片**到各 rank，消除冗余；分片代价是额外通信。

| Stage | 分片内容 | 通信量 |
|---|---|---|
| ZeRO-1 | optimizer state | 2Ψ（等价 DP）|
| ZeRO-2 | + gradient | 2Ψ |
| ZeRO-3（FSDP） | + parameter | **3Ψ** |

通用内存公式（每 rank）：

$$
m_{rank} = \frac{2\Psi + 2\Psi + k\Psi}{N_d}
$$

$\Psi$ 是参数量，$k$ 是 optimizer state 的内存倍数（Adam 是 12），$N_d$ 是 DP 度。

> ZeRO-3 在 PyTorch 里也叫 **FSDP（Fully Sharded Data Parallel）**。下文混用。

### ZeRO-1

- forward / backward 跟朴素 DP 一样，每 rank 都有完整 bf16 参数 + 梯度
- backward 后：**reduce-scatter**（注意是 reduce-scatter，**比 all-reduce 快 2 倍**）梯度 → 每 rank 只保留 $1/N_d$ 梯度
- optimizer step：每 rank 只更新自己那 $1/N_d$ 个 FP32 master weight
- optimizer 之后：**all-gather** bf16 参数到所有 rank

`reduce-scatter` 是 ZeRO 引入的第三个集合通信原语（前面 DP 只有 all-reduce）。

优化：all-gather 的通信可以与 optimizer step overlap（先把已更新的部分 gather），或与 forward 重叠（每 layer 用之前才 gather）。

### ZeRO-2

把梯度也按 $1/N_d$ 分片——backward 直接 reduce-scatter，省掉一份全量梯度。通信与 ZeRO-1 相同但显存更低：

$$
\frac{2\Psi + (2\Psi + k\Psi)}{N_d}
$$

通常 **ZeRO-2 是性价比最高的选择**。

### ZeRO-3

参数也分片。流程：

- forward：进入 layer $i$ 之前 all-gather 该 layer 参数，用完立刻丢弃
- backward：再 all-gather（PyTorch 实现里 backward 也需要完整参数算 Wgrad）
- backward 末尾 reduce-scatter 梯度

每层通信 `2 * num_layers - 1` 次额外 all-gather，总通信量 **3Ψ**（vs ZeRO-2 的 2Ψ）。但可以用 **prefetching**：forward layer $n$ 时后台 all-gather layer $n+1$；backward layer $n$ 时后台 all-gather layer $n-1$。**DP 度不宜超过 ~512**，否则通信延迟 > 计算时长，prefetch 撑不住。

### DP / ZeRO 总结

DP 只能 scale 到"单层能塞进一张 GPU"的模型。ZeRO 把参数 / 梯度 / 优化器状态按 DP 轴分片，但**分不了 activation**——activation 仍随 sequence length / batch size 涨。要继续打破天花板，得换维度——TP。

## 05 Tensor Parallelism（TP）

### 数学基础

矩阵乘 $A \cdot B$ 的两种分片方式：

1. **列分片**：$A \cdot [B_1, B_2, \dots] = [AB_1, AB_2, \dots]$——结果直接拆开，不需通信
2. **行分片**：$[A_1, A_2, \dots] \cdot [B_1; B_2; \dots] = \sum_i A_i B_i$——结果求和，需 **all-reduce**

### Transformer Block 内的 TP 切法

一个 transformer block 有两次大矩阵乘：

- **Attention**：Q、K、V 三个矩阵按**列分片**（Q、K、V 各自独立切），输出投影按**行分片**
- **FFN**：先 Column Linear（gate + up），再 Row Linear（down）

约束：
- **TP 度不能超过 Q/K/V 的 head 数**（GQA 下要 ≤ KV head 数）。比如 Llama-3 70B 用 GQA，KV head 8，所以 TP=8 恰好到顶。
- TP 通信发生在**每一次 forward / backward**——所以 TP 度大会被通信拖垮。

### 通信模式小结

| 矩阵类型 | 切法 | forward 通信 | backward 通信 |
|---|---|---|---|
| Column Linear | 切输出维 | all-gather 输出 | reduce-scatter 梯度 |
| Row Linear | 切输入维 | reduce-scatter 输出 | all-gather 梯度 |

每次 Column + Row 一对共 `2 * seq * bs * h / TP` 字节/层，**远多于** ZeRO-3（`Ψ = L * 16h²`）。所以 TP 通常**只放在 node 内部**（8 卡 NVLink 域内），跨 node 改用其他维度。

## 06 Sequence Parallelism（SP）

TP 把 hidden 维切了，但 **Dropout、LayerNorm 这些 per-token 算子** 没切到，它们在 TP 通信间隙里反而是 full sequence 的 full hidden——浪费。

SP 把 TP 区域**外**（即 LN / Dropout / Residual）按 sequence 维切，配合 TP 通信交错：

| 阶段 | TP only | TP + SP |
|---|---|---|
| Enter TP（Column Linear 前） | h sharded, s full | h sharded, s all-gather to full |
| TP 区域 | h sharded, s full | h sharded, s full |
| Exit TP（Row Linear 后） | h full + all-reduce, s full | h full + reduce-scatter, s reduce-scatter to sharded |
| SP 区域 | h full, s full | h full, s sharded |

**关键收益**：Row Linear 之后那个 all-reduce 被替换成 reduce-scatter，省下一半通信量（因为接下来 SP 区域只需要 sharded 状态）。Megatron-LM 的论文里 SP 带来 ~10-15% 的吞吐提升。

## 07 Context Parallelism（CP）

当 sequence 长度到 128K+，即使有 SP + full recomputation，**activation 仍然随 seq length 线性涨**——单 node 装不下。

CP 的核心：把 **TP 内部** 也按 sequence 切。MLP、LayerNorm 这些 per-token 算子天然能处理 sequence 切片，不需要通信；**只有 attention 需要全 sequence 的 K/V**——这就是 Ring Attention 出场的地方。

### Ring Attention

每个 GPU 在每个时间步做三件事：

1. **非阻塞地**把自己的 K/V 发给下一个 GPU
2. **本地**算当前拥有的 Q·Kᵀ·V
3. 等待上一个 GPU 的 K/V 到位，回到步骤 1

4 个 GPU、4 个 token 要做 4 个 step，每 step 是 send + local compute + recv 的流水。

### Zig-Zag Ring Attention

朴素 Ring Attention 在 causal mask 下**负载不均**：第一个 GPU 一开始就拥有最早的 token，几乎不用等别人；最后一个 GPU 要等三轮。**Zig-Zag 把 token 按交叉顺序分配**到 GPU，让每个 GPU 都有"早 token + 晚 token"的混合，attention mask 上的工作量均匀。

### AllGather vs All-to-All 实现

两种方式实现 CP 通信：

- **AllGather**：一次同步收齐所有 K/V——**显存峰值高**（每卡临时存完整 K/V），但通信模式简单
- **All-to-All（Ring）**：按 ring 一片一片传——**显存友好**（每卡只多存一片），但有多步基础延迟

一般 **All-to-All 更省显存**，AllGather 更简单。

### 与 Flash Attention 的关系

CP 和 Flash Attention 都用 **online softmax** 思想避免存完整的注意力矩阵：Flash Attention 是单卡内的"算完一块丢一块"，CP 是多卡间的"算完一片丢一片"。可以叠用。

## 08 Pipeline Parallelism（PP）

当模型**单卡装不下**，再叠一张卡也装不下时，需要把**整层**切到多卡：layers 1-4 在 GPU 0，layers 5-8 在 GPU 1……

PP 通信只发生在 layer 之间传 activation，**通信频率远低于 TP**，所以跨 node 也能跑（IB 带宽够）。

### Bubble 问题

朴素 forward→backward，所有 GPU 流水线排开会出现大量"等下一批 micro-batch"的空白——**bubble**。

设 $p$ = pipeline 度，$m$ = micro-batch 数，$t_f$ / $t_b$ = 单 micro-batch 在单 stage 的 forward / backward 时间（一般 $t_b \approx 2 t_f$）：

$$
r_{bubble} = \frac{(p-1)(t_f+t_b)}{m(t_f+t_b)} = \frac{p-1}{m}
$$

### 调度方案演进

| 调度 | 思路 | 显存 | Bubble |
|---|---|---|---|
| **AFAB**（All-Forward-All-Backward） | 先做完全部 forward，再做 backward | 存 $m$ 个 micro-batch 激活（**爆炸**）| $\frac{p-1}{m}$ |
| **1F1B** | 稳态阶段交叉做 forward / backward | 存 $p$ 个 micro-batch 激活（**可控**）| $\frac{p-1}{m}$ |
| **Interleaved** | 每 GPU 持有 $v$ 个 model chunk，micro-batch 在 chunk 间循环 | 同 1F1B | $\frac{p-1}{v \cdot m}$（$v$ 倍降幅）|

Llama 3.1 用 1F1B + Interleaved Stages + 可调 depth-first / breadth-first 调度。

实测经验：跨 node 时 PP 比 TP 强很多——同样从 1 node（p=8）扩到 2 node（p=16），PP 只损失 ~14%，TP 要损失 ~43%。**所以跨 node 用 PP 是更好的默认选择**。

### ZeroBubble 与 DualPipe

进一步把 backward 拆细：**矩阵乘的 backward 实际是两件事**——输入梯度 B（下层 backward 需要）和权重梯度 W（只在 optimizer step 前需要）。W 可以灵活地**填进 bubble**里。

- **ZeroBubble**（论文）：这种细粒度切分，理论上能达到 ZB-H2 schedule 的零 bubble
- **DualPipe**（DeepSeek-V3）：在 ZeroBubble 基础上加**双向流**——两股 micro-batch 从 PP 维两端向中间走，进一步压缩空闲时间。DeepSeek V3 用 DualPipe 报告 "near-zero all-to-all overhead"

但调度本身需要 ILP 求解——工程复杂。

## 09 Expert Parallelism（EP）

专门给 MoE。每个 expert 的 FFN 互相独立，**按 expert 把 FFN 拆到不同 GPU**——比 TP 轻得多（不用切矩阵乘，只路由 token）。

关键 trick：DeepSeek-V3 限制每个 token **最多发到 M=4 个 node**，让 token 尽量留在同一 node，**减少跨 node 的 all-to-all**。

通常 EP 与 DP 联合使用：DP 切 batch，EP 切 expert——两个维度正交。

> 有些人把 EP 视为 DP 的子集（"用 specialized routing 而非同构复制"），但概念上是独立的。

## 10 5D Parallelism 综合

五种并行 + 三档 ZeRO 的对比：

| 方法 | 分片维度 | 主要省什么 | 主要代价 |
|---|---|---|---|
| DP | batch | activation（按 mbs） | 受最大 batch 限制 |
| PP | model layers | params | bubble + 调度复杂 |
| TP/SP | hidden dim / seq | params + activation | 要高带宽通信 |
| CP | seq | activation | attention 通信开销 |
| EP | expert dim | expert params | 要 MoE + 路由通信 |
| ZeRO-1 | DP 副本间分片 | optimizer state | params 通信 |
| ZeRO-2 | + gradients | 同上 | 同上 |
| ZeRO-3 | + params | 同上 | 同上 |

### PP vs ZeRO-3

两者都是"沿模型深度切 param"，但有重要区别：

| | ZeRO-3 | Pipeline Parallelism |
|---|---|---|
| 每卡存的 | 层的 $1/N_d$ | 完整的一层（或多个 chunk） |
| 通信的是 | 权重 | activation |
| 编排 | 模型无关 | 模型无关（但要平衡层数） |
| 实现难点 | 分片 + 通信调度 | 高效 PP schedule |
| Scaling 偏好 | 大 mbs / seq_len（藏通信）| 大 grad_acc（藏 bubble）|

DeepSeek-V3 训练用 **PP + ZeRO-1**（注意是 ZeRO-1 不是 ZeRO-3），是 PP 与 ZeRO-1 / ZeRO-2 组合的典型例子。**ZeRO-3 与 PP 不常组合**——叠加通信代价太大，除非你能把 GBS 撑得很高。

### 组合原则

- **TP (with SP) ↔ PP / ZeRO-3**：天然互补，TP 拆权重、PP / ZeRO-3 拆层
- **TP 通信频繁 → 留在 node 内（NVLink）**
- **PP / ZeRO-3 通信少 / 可 prefetch → 跨 node**
- **CP ↔ EP**：互相独立，可一起用
- **CP ↔ TP**：互补，TP 切 hidden、CP 切 seq
- **EP ↔ DP**：标准组合

经验法则：组织 GPU group 时，**TP 组 ≤ node 大小（8）**，PP / ZeRO 跨 node。

## 11 选型决策三步法

### Step 1：把单卡训练跑通

**GPU-rich case**（卡多）：

| 模型规模 | 推荐 |
|---|---|
| < 10B | 单种并行即可：TP 或 ZeRO-3 / DP + Full Recompute，跨 8 卡 |
| 10B-100B（> 8 卡）| 选一个：(TP=8 + PP) / (TP=8 + ZeRO-3) / 纯 ZeRO-3 |
| 512+ 卡 | 纯 ZeRO-3 通信代价上来了——考虑 DP + TP 或 DP + PP |
| 1024+ 卡 | 推荐 **TP=8 + DP（ZeRO-2）+ PP** |

特殊情况：
- 长序列 → 叠 CP（跨 node）
- MoE → 叠 EP（跨 node）

**GPU-poor case**（卡少）：
- 开启 full activation recomputation（牺牲 30-40% 速度换显存）
- 增大 grad_acc

### Step 2：凑到目标 GBS

GBS 不对就要调：

- **GBS 太小**：扩 DP / 增 grad_acc；长序列下可以加 CP
- **GBS 太大**：减 DP；长序列下减 CP

### Step 3：优化吞吐

- TP 度一路向上加到接近 node size（8），把其他并行度降下来
- DP + ZeRO-3 同步扩，保持 GBS 不变
- DP 通信撑不住时换 PP
- 单变量微调不同并行度
- 多试几个 mbs，找 GBS / 模型 / 计算 / 通信的最佳平衡

### 实测经验（4100+ 实验总结）

实验条件：seq=4096，GBS=1M tokens，8 卡 / node 的 H100 集群。

1. **节点越多效率越掉**——小模型尤其明显（compute-to-model ratio 低），GBS 上限 1M 也限制了用大 batch 补偿
2. **大模型反而被 memory 卡**——节点少时装不下，装得下也跑在 GPU 显存极限附近，效率低
3. **实现质量决定胜负**：刚开始 TP 比 PP 快，PP 优化完后反超，再优化 TP 通信 overlap 又回到领先——**没有永远的最优**，要看实现

> 想在自己的集群复现可以用 [nanotron](https://github.com/huggingface/nanotron) 的 benchmark 脚本。

## 12 GPU 底层：Kernel、Flash Attention、Mixed Precision

### GPU 体系结构速览（NVIDIA）

- **计算侧**：132 个 SM（Streaming Multiprocessors），每 SM 128 个 core，H100 共 16,896 core
- **存储侧**（按层级）：Registers（线程私有）→ Shared Memory / L1 cache（SM 内共享）→ L2 cache（SM 间共享）→ Global Memory（HBM，80GB，最慢）
- **Kernel**：跑在 GPU 上的代码，用 CUDA / Triton 写，编译成 PTX
- **调度**：threads → warps（32 threads）→ blocks（一 block 在一 SM 上）→ grid

### Kernel 优化的四个核心

1. **Memory Coalescing**：相邻 threads 访问连续地址，合并成一次内存事务
2. **Tiling**：把大矩阵切成小块，让数据块留在 shared memory 里复用（GEMM 的核心）
3. **Thread Coarsening**：每个 thread 处理多个数据元素，摊销调度开销
4. **Minimizing Control Divergence**：让 warp 内 threads 走相同分支，避免 SIMT 的半 warp 闲置

### Fused Kernel

CPU 与 GPU 操作可以**异步**——host 不必等 kernel 完成。Fused Kernel 把连续的点状操作（LayerNorm、激活函数等）**打包进一个 kernel**，避免 GPU ↔ HBM 来回倒数据。

> Horace He 的 [Making Deep Learning Go Brrr From First Principles](https://horace.io/brrr_intro.html) 是经典入门。

### Flash Attention 1 / 2 / 3

朴素的 attention 要在 HBM 上**物化** $S$ 和 $P$ 两个大矩阵——HBM 带宽成了瓶颈。

Flash Attention 的 trick：
- 把 S 切成能在 SM shared memory 装下的小块
- **online softmax**：不物化完整 S/P，只保存 max、sum 这些归一化所需的标量
- 直接在 SRAM 里累加部分 O 输出

效果：
- **显存大幅降低**（不存 $S$、$P$）
- 顺带去掉 $S^2$ 的 naïve attention 复杂度影响

Flash Attention 2、3 主要是**进一步压榨 GPU**：减少非 matmul 操作数（FA2）、更精细的 warp / block 划分（FA2）、针对 Hopper Tensor Core 的 FP8 优化（FA3）。

> FA 对 attention pattern 有约束，要更灵活的 variant 看 [FlexAttention](https://pytorch.org/blog/flexattention/)。

### Mixed Precision Training

| 格式 | bits | sign | exponent | mantissa | 备注 |
|---|---|---|---|---|---|
| float32 | 32 | 1 | 8 | 23 | 默认 |
| float16 | 16 | 1 | 5 | 10 | 范围小 |
| bfloat16 | 16 | 1 | 8 | 7 | brain float，范围同 fp32 |
| fp8 e4m3 | 8 | 1 | 4 | 3 | 精度好但范围小 |
| fp8 e5m2 | 8 | 1 | 5 | 2 | 范围大但精度差 |

bf16 保留 fp32 的范围，牺牲精度——是当前混合精度的默认。

#### FP16 / BF16 训练的三个 trick

1. **FP32 master weights**：小 weight 不会 underflow 到 0
2. **Loss scaling**：梯度通常 < 1，提前把 loss 放大 N 倍 → backward 后再 unscaled
3. **累加用 FP32**：求和 / 平均时用 FP32 累加避免精度损失

#### FP8 训练

H100 上 FP8 GEMM 比 BF16 **理论 2× FLOPS**——数字诱惑巨大，但不稳定。

第一次公开大规模成功的是 **DeepSeek-V3**。关键 trick：
- 按 **per-tile** 量化：输入/激活用 $1 \times 128$ tile，权重用 $128 \times 128$ tile，减少 outlier 影响
- 主权重、累加梯度保持高精度（BF16/FP32），只有 GEMM 操作本身用 FP8

各方案显存对比：

| 方案 | Master | Grad acc | W | Grad | Opt | 总(bytes/param) |
|---|---|---|---|---|---|---|
| BF16 + FP32 mixed（baseline）| bf16 | fp32 | fp32 | bf16 | fp32+fp32 | **20** |
| 同上但不 FP32 grad acc | bf16 | fp32 | n/a | bf16 | fp32+fp32 | 16 |
| Transformer Engine | fp8 | n/a | n/a | fp32 | fp32+fp32 | 16（-20%）|
| FP8-LM O3 | fp8 | fp16 | fp16 | fp8 | fp8+fp16 | **9**（-55%）|
| DeepSeek-V3 | fp8 | fp32 | fp32 | fp8 | bf16+bf16 | 15（-25%）|
| nanotron FP8 | fp8 | bf16 | fp32 | fp8 | fp8+fp8 | 10（-50%）|

到 2025 年初 FP8 仍算"实验性技术"，但趋势明确——Blackwell 已宣布支持 **FP4**。开源参考实现：nanotron [FP8 PR](https://github.com/huggingface/nanotron/pull/70)。

## 13 Conclusion（结语）

整本书走完一遍：单卡 → DP → ZeRO → TP → SP → CP → PP → EP → 5D 综合 → GPU 底层 → Flash Attention → Mixed Precision / FP8。最终能读懂 Llama-3 的 4D 并行配置图。

分布式训练不再是少数 pretrain 团队的专利——AI builder 社区扩大 + 模型规模继续增大，**任何 fine-tuning / inference / 训练 都越来越可能用上分布式技巧**。这正是该读这本书的时机。

## Appendix A：集合通信原语速查

| 原语 | 功能 | 在 LLM 训练中的典型用途 |
|---|---|---|
| Broadcast | 1→N 广播 | 主节点广播初始权重 |
| Reduce | N→1 归约 | 单节点收梯度 |
| AllReduce | N→N 归约 + 广播 | DP 梯度同步 |
| Gather | N→1 收集 | 收集 log / 监控 |
| AllGather | N→N 收集 | ZeRO-3 gather params |
| Scatter | 1→N 分发 | 初始化分发 |
| ReduceScatter | N→N 归约并分片 | ZeRO-1/2 gradient 同步 |
| Ring AllReduce | AllReduce 的带宽最优实现 | 大规模 DP |
| Barrier | 同步屏障 | checkpoint 同步 |

实现：NCCL（NVIDIA Collective Communications Library）是事实标准。

## Appendix B：典型 LLM 训练规模（数量级速查）

- **Input tokens**：每 batch 处理 `seq * mbs` 个 token
- **Activation**（单 layer）：`seq * mbs * h` 个元素
- **权重 / 梯度**：每个线性层约 $h^2$ 元素
- **Optimizer 状态**（Adam + mixed precision）：每权重矩阵 $6h^2$（FP32 momentum + variance + FP32 master weight）
- **Transformer block 总参数**：$16h^2$（含 GLU 的 MLP），无 GLU 是 $12h^2$
- **整模型参数**：$16h^2 \cdot L$（含 GLU）
- **Forward FLOPs**：$\approx 2 \cdot \text{num\_tokens} \cdot \text{num\_params}$
- **Backward FLOPs**：$\approx 2 \times \text{forward}$

## Appendix C：通信 vs 计算 Overlap 的数学判断

设 $\text{peak\_flops}$ 是 GPU 算力（FLOPs/s），$\text{peak\_bw}$ 是互联带宽（B/s）。两者之比决定 overlap 是否成立：

### DP（baseline）

每 bucket 通信：$t_{comm} = \frac{bucket\_size \cdot 2(DP-1)}{DP \cdot \text{peak\_bw}}$

Backward 计算：$t_{compute} = \frac{4 \cdot \text{num\_tokens} \cdot \text{num\_params}}{\text{peak\_flops}}$

Overlap 条件（通信 / 计算 ≤ 1）：

$$
\frac{\text{num\_params}}{2 \cdot \text{num\_tokens}} \cdot \frac{DP-1}{DP} \cdot \frac{\text{peak\_flops}}{\text{peak\_bw}} \le 1
$$

### ZeRO-3（FSDP）

每 block 通信总量 $3 \cdot 16h^2 / DP$，所有 layer 总 $3L \cdot 16h^2 / DP$。

$$
\frac{t_{comm}}{t_{compute}} = \frac{1}{2 \cdot seq \cdot mbs} \cdot \frac{DP-1}{DP} \cdot \frac{\text{peak\_flops}}{\text{peak\_bw}} \le 1
$$

——通信能藏进计算的临界条件与 sequence × mbs 成反比。

### TP

每 block 通信 $8 \cdot seq \cdot mbs \cdot h / TP$：

$$
\frac{t_{comm}}{t_{compute}} = \frac{TP-1}{2h} \cdot \frac{\text{peak\_flops}}{\text{peak\_bw}} \le 1
$$

**只与 hidden dim 和 TP 度有关，与 seq/mbs 无关**——所以 TP 的通信能否被盖住主要看 hidden size。

### PP

每 microbatch 通信 $4 \cdot seq \cdot mbs \cdot h$，所有 grad_acc 步总和 $4 \cdot grad\_acc \cdot seq \cdot mbs \cdot h$：

$$
\frac{t_{comm}}{t_{compute}} = \frac{\text{peak\_flops}}{32 \cdot h \cdot L_{next} \cdot \text{peak\_bw}} \le 1
$$

也跟 seq / batch 无关——靠**隐藏维 × 下一 stage 的层数**决定 PP 的可扩性。

## Appendix D：参考资料

### 里程碑论文

- ZeRO（[Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054)）
- Megatron-LM（TP / SP；[Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)）
- Ring Attention（[Liu et al., 2023](https://arxiv.org/abs/2310.01889)）
- Flash Attention（[Dao et al., 2022](https://arxiv.org/abs/2205.14135)；FA2 [2023](https://arxiv.org/abs/2307.08691)；FA3 [2024](https://arxiv.org/abs/2407.08608)）
- Pipeline Parallel schedules：GPipe、1F1B、PipeDream、ZeroBubble
- DeepSeek-V3 Technical Report（FP8 + DualPipe）
- FP8-LM（FP8 pretraining）

### 训练框架

- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [nanotron](https://github.com/huggingface/nanotron)
- [picotron](https://github.com/huggingface/picotron)

### 内部交叉引用

- [[Qwen2.5-3.7 系列最全总结]]：Qwen 实际训练用到的并行配置示例
- [[Knowledge Distillation]]：同样依赖大模型分布式部署

## 阅读建议

- 第一次读：跳过 Appendix 数学推导，先把 1-11 章过一遍，建立全局图
- 第二次读：精读 ZeRO 与 TP 章节，对照 picotron 源码
- 工程实践：先看 nanoGPT → picotron → nanotron 的递进
- 决策调优：跑通后用 nanotron benchmark 脚本在自己集群扫配置
