# RL 面试题 2026


> 来源: 整理自 [@sheriyuo](https://x.com/sheriyuo/status/2063295181131247674) 的长文 **RL Interview Questions 2026** (原文链接) 与 [@vivek_2332](https://x.com/vivek_2332/status/2063566811749331353) 的逐题答案。
>
> sheriyuo 在知乎也发过中文版: <https://zhuanlan.zhihu.com/p/2046740446353811230>
>
> sheriyuo 在原文中提到几点: 没严格区分 LLM RL 与 Agentic RL; 几乎每道题都能继续追问, 没有给出参考答案; 现代 RL 招聘越来越要求全栈理解, 算法岗也会问 infra, 反之亦然; 数据相关问题没收录 (太看经验); 死记硬背不够, 深度理解更重要。
>
> Vivek 在一些题上明确说 *"not sure"* 或 *"haven't read the paper"*, 我在对应位置标注了 ⚠️。

## 算法 (Algorithm)

### 1. 为什么用 Actor-Critic 而不是纯 Critic?

纯 value-based 方法 (DQN 这类) 处理不了大的 / 连续的动作空间, 而且只能给出 greedy policy。纯 policy gradient 能处理它们, 但更新方差高。Actor-Critic 直接保留策略, 用 critic 来压低这个方差。两全其美。

### 2. KL 散度、交叉熵、MLE 之间有什么关系?

$CE(p,q) = KL(p,q) + H(p)$, 最小化任意一项最终收敛到同一处。MLE 本质上就是数据分布和模型分布之间的 KL。

### 3. 不同 RL 场景下奖励该怎么设计?

主要看你建的是什么环境。可验证的 (数学、代码) 比不可验证的 (写作) 容易奖励得多, 原因显而易见。最重要的是环境不能被 reward hack, 不能有一条通往错误解的捷径。数据和奖励都得仔细写。

### 4. importance sampling、rejection sampling、其它 Monte Carlo 方法在 RL 里怎么用?

IS 用于 async RL 复用更早 policy 产生的 off-policy 样本。RS 用于数据过滤, 可以过滤 SFT 数据, 也可以扔掉过易 / 过难的样本。广义上都是 Monte Carlo, 用样本估期望而不是精确计算。

### 5. PPO 和 GRPO 里 advantage 怎么算? 为啥要减 baseline? std 归一化真的必需吗?

PPO 有 critic + reward model。GRPO 干掉这两个, 它的 "critic" 来自多次 rollout 取均值当 baseline。baseline 降低方差, 这样不是每个 rollout 都得到正向推动, 构成组内零和。std 归一化不必要, 参考 [Dr.GRPO](https://arxiv.org/abs/2503.20783)。

### 6. RL 训练和 test-time scaling 在探索方式上有什么不同?

RL 是把领域 intelligence 注入权重。test-time scaling 是在已有 intelligence 前提下, 找最优解的路径, 用更多 thinking token 去探索不同方案。一个塑权重, 一个花推理预算。

### 7. PPO clipping 怎么工作的? 为啥取 min? 不 clip 会怎样? CISPO 区别在哪?

clip 是为了防止策略突变, 不同样本的 ratio 方差会变得很高。min 来自 TRPO 的 trust region 思想, 它让更新保持悲观, 只在有利时才动。CISPO 不一样, 它 clip importance weight 而不是整个目标, 这样所有 token 仍有梯度流过, 保留更多信号。

### 8. GRPO 为什么加 KL 惩罚? KL 怎么算? 为什么 DAPO、GSPO 等方法把它去掉?

KL 在是因为已经有一个不错的基座模型, 你不想离它太远。但又希望通过 RL 学新领域。它被去掉是因为同样的约束反而阻止你学到足够远, 这不是我们想要的。所以去掉 KL 在 RLVR 下反而更稳。

### 9. ⚠️ LLM 训练时 loss 不小心被 AllReduce 多次会怎样?

Vivek 自己说没看懂这题。

### 10. DPO 的 reward function 是什么? 会出现 reward hacking 吗? 怎么缓解?

DPO 数据里隐式带 reward (标记好坏样本), 不像 PPO 需要单独的 reward model。Reward hacking 还是会发生 (见 [@willccbb](https://x.com/willccbb)), 因为你追逐的解有一条后门。目前没有单一解法, 环境必须严格压测。可观察 reward 突然飙升或 rollout 长度暴涨这类信号来抓。

### 11. MoE 模型里 train-inference mismatch 的解决方法有哪些?

trainer-inference mismatch 来源有几个: 不同 engine、不同 expert 选择、不同 kernel、量化、舍入 (参考 [Thinking Machines 博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-llm-inference/))。一种修法是 replaying expert selection, 让推理和训练选同样的 expert。广义上是要让 rollout policy 和训练 policy 数值上一致, 否则 importance weight 会错。

### 12. RL 训练时 group size、learning rate、PPO epochs、generation length 怎么选?

- group size 就是更多 rollout、更多可学习的路径, 常用 8/16。
- lr 可控, $1 \times 10^{-6}$ 量级。
- PPO epochs 通常 1, 多会让数据更 off-policy 不稳定。
- generation length 完全看任务, 不交代任务没法量化。

### 13. 对比 GRPO, Dr.GRPO / DAPO / GSPO / CISPO / SAPO / DPPO / MaxRL / SimKO 各自怎么改进了训练? 有什么局限?

基本都是 GRPO 的变体, 主要改稳定性。lab 规模下选哪个差别不大, 但有差异:

- **Dr.GRPO**: 去 std 归一化。
- **DAPO**: clip-higher、动态采样、token 级 loss、超长过滤。
- **GSPO**: sequence-level importance sampling 而非 token-level, 对 MoE 好很多。
- **CISPO**: clip weight 不 clip objective。
- **MaxRL**: 解决多样性坍缩。
- **SAPO / DPPO**: ⚠️ Vivek 没读过论文。

### 14. TRPO、DPPO、AReaL 怎么在 RL 目标上施加 trust-region 约束?

TRPO 是带约束的优化, 留在 KL 边界内。PPO 用 clip 近似的也是这个。AReaL 在 async 设置下用 staleness / off-policyness 代替硬 KL。

### 15. RL 能从根本上扩展 LLM 的能力边界吗?

目前论文证据倾向于 RL 主要锐化基座模型已有能力, 而不是形成新能力。不过这是一个有趣的方向, 只要探索够多、样本够多样, 我不完全确定它做不到。

### 16. ⚠️ 基于 ProRL 等工作, 怎么思考 RL 训练的边界扩展?

Vivek 没读这篇。

### 17. OPD 比传统 RL 和 SFT 有什么改进? 应用场景?

OPD 把 RL 的 on-policy 探索跟 SFT 的稠密信号结合起来。RL 的弱点之一是稀疏奖励, OPD 让 teacher 在学生自己 (on-policy) 的 rollout 上打分 (关键是 on-policy), 就修了这个。用来低开销迁移能力, 最近是这个方向的热趋势。

### 18. LLM 推理能力在哪个训练阶段出现?

不能完全确定, 但 CoT 起到很大作用。还有预训练给基线 intelligence, 后训练用 reasoning 和 test-time compute 放大 intelligence。

### 19. ⚠️ 从 DeepSeek R1 到 V3.2 到未来 V4, 引入了哪些 RL 相关改进? MoE 上 RL 有什么不同?

Vivek 不记得 V4 的细节。

---

## 基础设施 (Infrastructure)

### 1. 忽略 CPU offload, GRPO 训练时内存里有几份模型副本? 各种优化能省多少?

主要取决于算法和 KL。GRPO 带 KL 时需要训练 policy、计算 KL 的 reference policy、推理模型。优化内存可以从去掉 KL 开始, 干掉 reference model。还有一些优化比如 NVFP4 量化省显存, 分布式训练 (FSDP / EP / TP 把模型和 optimizer state 切到多卡) 也帮很大。可以把推理那份量化, 把 optimizer state 切片每 rank 只持有部分。

### 2. 分布式推理: KV cache 传输优化和多 GPU 通信策略?

KV cache 传输对 disaggregated prefill / decode 很重要。通过 NVLink 或逐层 overlap 传输 KV / 权重来优化。多 GPU 看是同节点还是跨节点。总的来说 TP 帮助很大, 重的 KV 移动尽量留在单节点内。

### 3. INT8 vs FP8 的权衡? 训练和推理各偏好哪种精度?

FP8 的指数位多, 更适合训练。INT8 在固定范围内精度更高, 更适合推理权重。所以训练用 FP8, 推理 serving 用 INT8。

### 4. RL rollout 里的长尾问题是什么? 怎么处理?

batch 内 rollout 长度长尾。可用 PipelineRL、continuous batching 或早期截断。常见于 async 场景。

### 5. Continuous batching 在 RL 训练里引入什么问题? vLLM 和 SGLang 的区别?

Continuous batching 把不同 next-token 预测阶段的 decoding 混合在一起。vLLM 用 PagedAttention, SGLang 用 RadixAttention (前缀共享)。具体读 paper。RL 这里要注意序列在不同时间结束时, 对齐完整轨迹和 logprobs。

### 6. 怎么衡量 vLLM 和 SGLang 的利用率? 训练时怎么评估 KV cache 利用率?

通过吞吐量 (tok/s)、KV cache 利用率、GPU 利用率。KV cache 看 occupancy。也看是否内存受限, 以及 rollout 是否被驱逐。GPU 利用率低通常意味着在等东西 (rollout 同步或 CPU) 而不是计算受限。两个 engine 都暴露这些 scheduler / cache 指标。

### 7. 大规模多节点 RL 训练里 backprop 怎么实现?

跟常规 backprop 差不多, 加一些 trick: FSDP、EP、TP。超长上下文加 CP (context parallel)。理想上没人用 PP, 太复杂, 大多数规模下没必要。

### 8. 有哪些异步 RL 框架? 它们解决什么同步瓶颈?

异步 RL 框架: Prime-RL、verl、TRL。瓶颈归结为三件事:

1. rollout 的 staleness 及管理方式 (IS + staleness bounds);
2. trainer-inference mismatch;
3. token-in token-out。

解决 mismatch 的办法: 让 rollout engine 和 trainer 数值上保持一致, replaying expert / router 决策, 在 trainer 侧算 logprobs 而不是信推理引擎。

### 9. ⚠️ AReaL 或其它部分 rollout 框架里, 是否保留旧 policy 的 KV cache?

Vivek 不确定 AReaL, 但有些方法保留, 有些没有。两种做法之间实际没太大性能差异。

### 10. Expert Parallelism (EP) 如何影响 MoE 吞吐量?

EP 对 MoE 模型影响很大, 把一批 experts 放到一个节点或一组 GPU 上, 每个持有一小部分。它通过并行 expert FFN 提升吞吐量, 但要为 token 路由到对应 expert 增加 all-to-all 通信。实际收益是 all-to-all 能跟计算 overlap, 以及 expert 负载均衡程度。

### 11. 长上下文训练里 compute-communication overlap 怎么设计? Megatron 和 FSDP 在并行策略上有什么区别?

关键是通信和计算并行起来。FSDP 把 grads / params / optim 切片, 按需 gather, 简单很多。Megatron 用 DP + PP + TP (3D 并行), 通信更显式。Megatron 这块我不太熟, 用得少。

### 12. 怎么开启确定性执行? batch invariance 是什么? 是什么引起的? 跟 atomic add 有关吗? atomic add 能解决问题吗?

需要 reduction 顺序一致、相同的 kernel、batch invariant 的 op。参考 [Thinking Machines 博客](https://thinkingmachines.ai/blog/defeating-nondeterminism-llm-inference/)。

### 13. AReaL 和 Slime 对 RL rollout 瓶颈的理解有什么区别?

AReaL 我能聊: 它完全 async, 带 staleness 控制, 把 rollout 生成和训练解耦, 让 trainer 不用等最慢的 rollout, 然后用 max off-policy steps 或 importance sampling 或过阈值的 sample 直接丢弃来控制 staleness。Slime 这块 ⚠️ Vivek 没仔细读, 需要再读一下。

### 14. 完全异步 RL 训练里 staleness 怎么理解? 实际典型值?

Staleness = 生成 rollout 的 policy 和当前训练 policy 之间的差距。我跑的实验通常是落后 1–4 步。Staleness 大了会让 IS 不稳定, ratio 会爆。

### 15. ⚠️ slime 里的数据流怎么走? 怎么跟 Megatron 整合? loss 怎么算?

Vivek 同样不确定 Slime。

### 16. VeRL、TRL、Unsloth、AReaL、slime 之间选一个怎么选?

都不用, 用 [Prime Intellect](https://x.com/PrimeIntellect)。(原推的吐槽/梗)

---

## 参考链接

- sheriyuo 长文 (英文): <https://x.com/sheriyuo/status/2063295181131247674>
- sheriyuo 知乎版: <https://zhuanlan.zhihu.com/p/2046740446353811230>
- vivek 答案: <https://x.com/vivek_2332/status/2063566811749331353>
- Thinking Machines (defeating nondeterminism): <https://thinkingmachines.ai/blog/defeating-nondeterminism-llm-inference/>
- Dr.GRPO 论文: <https://arxiv.org/abs/2503.20783>

