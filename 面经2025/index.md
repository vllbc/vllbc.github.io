# 面经2025

# 常见问题总结

## 多轮对话sft样本怎么构造？

- [大模型微调样本构造 trick]（https://zhuanlan.zhihu.com/p/641562439)
    
    - 多轮对话的传统组织方式：将多轮对话拆分为多条独立的训练样本，如 Q1A1/Q2A2/Q3A3 可拆分为 Q1—>A1， Q1A1Q2->A2， Q1A1Q2A2Q3->A3 三条样本。
        
    - 将整个 session 的对话内容拼接成一个长文本序列，例如：Q1 A1 Q2 A2 Q3 A3。这样，整个 session 被表示为一个连续的文本序列，而不是多条独立的样本。
        
    - 构造时计算损失有一些坑，详见 https://zhuanlan.zhihu.com/p/721652210
        

## SFT 是否可以注入知识？

- continue pretrain 注入知识。
    
- sft 对齐输出格式。（sft 在一些特定的场景下确实是可以注入知识的）
    

## 如何解决灾难性遗忘？

1. 保留通用数据：在进行领域数据训练时，仍然需要保留一部分通用数据用于模型训练。这样可以确保模型仍然能够学习到通用的语言和知识，从而保持一定的通用能力。
    
2. 增量学习：使用增量学习（Incremental Learning）的方法，将领域数据与通用数据逐步交替进行训练。这样可以在学习新领域的同时，保持对通用知识的记忆。
    
3. [数据重采样](https://zhida.zhihu.com/search?content_id=234154468&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E9%87%8D%E9%87%87%E6%A0%B7&zhida_source=entity)：在进行领域数据训练时，可以使用数据重采样的方法，使得模型在训练过程中能够更多地接触到通用数据，从而缓解遗忘通用能力的问题。
    
4. 强化学习：使用强化学习的方法，通过给模型设置奖励机制，鼓励模型在领域任务上表现好，同时保持一定的通用能力。
    
5. [领域适应技术](https://zhida.zhihu.com/search?content_id=234154468&content_type=Article&match_order=1&q=%E9%A2%86%E5%9F%9F%E9%80%82%E5%BA%94%E6%8A%80%E6%9C%AF&zhida_source=entity)：使用领域适应技术，如[领域自适应](https://zhida.zhihu.com/search?content_id=234154468&content_type=Article&match_order=1&q=%E9%A2%86%E5%9F%9F%E8%87%AA%E9%80%82%E5%BA%94&zhida_source=entity)（Domain Adaptation）和领域对抗训练（Domain Adversarial Training），帮助模型在不同领域之间进行迁移学习，从而减少遗忘通用能力的问题。
    
6. [SDFT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.13669)：微调前，让大模型将任务数据集重写一遍。这样的话，重写后的任务[数据集](https://zhida.zhihu.com/search?content_id=234154468&content_type=Article&match_order=2&q=%E6%95%B0%E6%8D%AE%E9%9B%86&zhida_source=entity)的分布和大模型的差异就小了很多。在这样的数据集上微调对大模型分布上的改变会小很多，对大模型通用能力的损害也会降低。
    
7. Llama-Pro:在原始模型中每个Transformer块或者某几个Transformer块后增加一个Transformer块，但为了保持扩展后的模型输出保持不变，需要增加的块为恒等块（输入输出相同）
    

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MjQwZGU5ZWMyMGI2YmI0MDNiMGRiZDY1NTZiYjBmYTNfT093bXI0bmVSU1FER2ZHMzZzTUo5VUV3WnF3VHhyemNfVG9rZW46RlJ3TGJoTWtFb05WSm54eXB1RmNIdzJRbnRnXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

## 为什么大模型都是 Decoder-Only 架构？（https://www.zhihu.com/question/588325646/answer/3357252612）

- 泛化性能更好：[ICML 22](https://zhida.zhihu.com/search?content_id=640486327&content_type=Answer&match_order=1&q=ICML+22&zhida_source=entity)的[What language model architecture and pretraining objective works best for zero-shot generalization?.](https://link.zhihu.com/?target=https%3A//proceedings.mlr.press/v162/wang22u/wang22u.pdf) 在最大5B参数量、170B token数据量的规模下做了一些列实验，发现用next token prediction预训练的decoder-only模型在各种下游任务上zero-shot泛化性能最好
    
- 苏神强调的注意力满秩的问题，双向attention的注意力矩阵容易退化为低秩状态，而 causal attention的注意力矩阵是下三角矩阵，必然是满秩的，建模能力更强；
    
- @yili大佬强调的预训练任务难度问题，纯粹的decoder-only架构+next token predicition预训练，每个位置所能接触的信息比其他架构少，要预测下一个token难度更高，当模型足够大，数据足够多的时候，decoder-only模型学习通用表征的上限更高；
    
- @mimimumu 大佬强调，上下文学习为decoder-only架构带来的更好的few-shot性能：prompt 和demonstration的信息可以视为对模型参数的隐式微调，decoder-only的架构相比encoder-decoder在in-context learning上会更有优势，因为prompt可以更加直接地作用于decoder每一层的参数，微调的信号更强；
    
- 多位大佬强调了一个很容易被忽视的属性，causal attention（就是decoder-only的单向 attention）具有隐式的位置编码功能，打破了transformer的位置不变性，而带有双向 attention的模型，如果不带位置编码，双向attention的部分token可以对换也不改变表示，对语序的区分能力天生较弱。
    
- decoder-only支持一直复用KV-Cache，对多轮对话更友好，因为每个token的表示只和它之前的输入有关，而encoder-decoder和PrefixLM就难以做到。
    

## Transfomer attention计算为什么除以根号 d？

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NWU0MTA2YzUyZjJhZDhhOTdmM2JkZDAwN2Y4NWE4NDBfOFlSRjM5eVdEOVh5MmVTS2dXSktlSllUN1hUSHZFNVBfVG9rZW46TE5vcmJ4MWZSb0ljM2N4ak82UmM5Y1dBbk1nXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

## 为什么不在rollout阶段保存logp而是再计算一遍logp？

因为forward的时候和generate的时候logprob由于推理引擎（vllm）和训练引擎（fsdp）的优化目标不一样，会造成两者对不上，因此需要做两次。

batch 算子的细微差异，都会造成这两个 log_prob 不完全一致。推理引擎要的是快速出 token id，训练引擎需要保证一定的log_prob 精度。

## 为什么需要 Server 来完成 rollout？

为了配合 agentic LLM 的训练，在现有的PPO/GRPO 算法的基础上，从 single turn rollout 改动为 environment interactive multi-turn rollout 的需求非常强烈。

  

这一过程中，policy 与 environment 的交互存在绝对不可忽视的延迟，turn 之间的等待时间很长。一直用 Engine 做 rollout 的话（ engine.generate），可能连 continuous batching 都组不起来。所以，改用 server 来通过 https 做 rollout的需求就呼之欲出了。实际上，这也是 最自然的工作方式。除此之外，environment 的交互往往也是通过 https 请求来完成的。譬如，众多 coding sandbox 都是 environment 自己启动一个 server 暴露一个 port，然后往里面发请求来实现交互的。

  

总之，为了在 training engine,rollout 和 environment 三个子进程中保持良好的通讯和交互，选择 server 势在必行。

## MCP和fuction call的区别？（https://zhuanlan.zhihu.com/p/1898326676087223572）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NDlkNjY4ZDUyNjNkOTY1ZjVkNTc5ZWE3Y2ExMGFmYTNfUHMwcVcyNlZiTE5PaFhGSXEzV1dydHNXc0ZHVkdLSk9fVG9rZW46QUxMNGJabktQb0x2bEt4bmlna2N3d042bm9iXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

- 函数调用是一种机制，它允许 LLM 根据用户的输入识别它需要什么工具以及何时调用它。
    
- MCP（即模型上下文协议,[Model Context Protocol](https://zhida.zhihu.com/search?content_id=256840389&content_type=Article&match_order=1&q=Model+Context+Protocol&zhida_source=entity)）试图标准化此过程。 MCP (Model Context Protocol):是一个开放协议和标准，旨在标准化AI 应用（MCP 客户端）如何发现、连接和与外部工具/数据源（实现为 MCP 服务器）进行交互。它关注的是系统间的通信和集成，解决 Function Calling 指令生成后，如何高效、安全、可扩展地执行这些调用。
    

## 为什么R1不使用模型reward？

- reward hacking
    
- 训练资源问题，成本过高。
    

## 为什么PRM+MCTS这条路走不通？（https://zhuanlan.zhihu.com/p/19623772462）

- 在reasoning任务中如何显式定义step，比如以`\n` 还是以推理逻辑来划分step？
    
- 如何定义step正确性，将影响step labeler来高效标注
    
- PRM容易reward hacking
    

  

- LLM比象棋搜索空间大太多
    
- MCTS价值影响模型生成质量（不如纯CoT采样）
    

## GRPO中可以去掉KL项吗？

可以。

1. 去除KL项, 意味着不需要ref-model, 减少一个模型的显存，减少一次前向ref_policy的计算。
    
2. 没有KL的约束，那么可以将过大的梯度进行裁剪(max_grad_norm)，避免优化的不稳定性(这也是另一种层面的clip)。
    
3. 没有KL的约束，参数的优化更加自由，更容易探索到好的回答
    

## GRPO 的损失为什么会为负（参考 https://zhuanlan.zhihu.com/p/28326620566）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YmNkMDNkNTM1ZWM0ZGU3NWY1ZjhjMmY1YWRlZGVlZGZfb3FxVTFqSUVxdWYyOGhJeTdjUHZVTTBEWDZYZGwxM3ZfVG9rZW46RHdqT2J1UVlDb05hbkJ4WVdqMWNxMmlsblZlXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

总结：组奖励变化对 loss 上升影响不直观。优化策略远离 ref，KL 变化大，导致 Loss 上升明显。

## Qwen模型为什么随机奖励也能work？（https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f）

- 模型依赖性：研究发现，RLVR的有效性更多地依赖于模型的预训练能力，而不是监督信号的质量。Qwen模型在预训练期间学会了特定的推理策略，这些策略可以通过RLVR轻易地被激发出来，而其他模型则不具备这些策略。
    
- 代码推理策略：Qwen-Math模型在预训练阶段就频繁地使用Python代码来解决数学问题，即使在没有代码执行器的情况下，也能生成正确的代码输出和答案。RLVR训练（无论奖励质量如何）进一步增加了这种代码推理的频率，从而提高了性能。
    
- 奖励信号的作用：不可靠奖励通过放大模型在预训练期间学到的有用推理表示来发挥作用。这些奖励信号并没有教会模型任务质量，而是触发了一种集中效应，使模型专注于其现有的推理模式分布。
    

  

## 为什么DPO里Chosen和Rejected概率会同时下降?（https://zhuanlan.zhihu.com/p/6327313416）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWNkNTA5N2U4NTM0YjExNTM1MjE1YTFhMjU2YWIwMzZfdUNYSDNmbnA2bTNzbmxqMDJIQ3YzckYyaFFlQUFPRFBfVG9rZW46UHpEbmJFRTFvb3pNUUJ4aUV4bWNsQjBzbmpjXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmVhMDM1OTZiMGZiN2U5ZGU3ODRiYWE0NWNhN2Q2ZWFfVFBEMFdlNHdESWtSaWtFelVFajY4TlhMTndCQkxvcWNfVG9rZW46VWZ5QWJzOGowb0d4M0Z4VHY3ZWN6VEIwbk5jXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

## reward hacking如何解决？

  

## [梯度累积两次，跟 batch size 增大 2 倍，在多数情况下，效果一样吗？](https://www.zhihu.com/question/583011902/answer/7205474551)（loss 的 3 次平均）

理论上，[梯度累计](https://zhida.zhihu.com/search?content_id=694556710&content_type=Answer&match_order=1&q=%E6%A2%AF%E5%BA%A6%E7%B4%AF%E8%AE%A1&zhida_source=entity)在数学上应该等同于[全批量训练](https://zhida.zhihu.com/search?content_id=694556710&content_type=Answer&match_order=1&q=%E5%85%A8%E6%89%B9%E9%87%8F%E8%AE%AD%E7%BB%83&zhida_source=entity)，但实际发现 loss 并不匹配。( [Gradient accumulation yields worse results than the equivalent batch size · Issue #2175 · huggingface/trl](https://link.zhihu.com/?target=https%3A//github.com/huggingface/trl/issues/2175))

一般情况下，loss 计算会经历三次平均

1. micro batch 维度，分母是这个 micro batch 中的所有 label 不是 -100 的 token 数**（不同 token 之间 loss 的平均）**
    
2. DP 维度，分母是 DP size **（和 GPU 数量相关，不同机器之间 loss 的平均）**
    
3. 梯度累加维度，分母是梯度累加数。**（不同 batch 之间的 loss 的平均）**
    

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkxZDY5NzFlYTM1ZjdiODJmNzM4NTE0NjBiNDgzNTFfNkRicjNyRWVZNFpxQXVmcG1lamw2SzR0ZGxoNGRzRG9fVG9rZW46SWY4UGJMRzlkb2l5c1h4NEFHZGNLRkxjbjVmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

## 为什么不用大模型做 embedding？

- 大模型主要训练的预测 next token 的能力，而非判断整个句子 embedding 的好坏。因此使用 LLM 做嵌入效果不理想。
    
- 部署成本高。
    

# 基础知识

## Post-Training总结（https://mp.weixin.qq.com/s/VLWU3YnJa1SZRCySZQc4Hw）

## Flash Attention

  

## MoE

  

## KV cache

  

## 从 MHA 到 MLA

  

## Transformer 相关

### 手撕各模块代码

### free-running mode 和 teacher forcing mode（https://zhuanlan.zhihu.com/p/630356292)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDdhODlkYTVmMTI5NmYyODNiNGNhYjI1ZDg4NDNkMWRfc0RiVUJDOVVsWFFnWTR6SnpQeXAyT0xYYVlGZmlxMDhfVG9rZW46T25xbGJ3bnF3b2x0UmF4MXhnWmNqV244bkhmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### Transformer 参数量计算

  

## overthinking 怎么解决

### 相关文献

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=OTBhZjk0NDk5YTc5YTkyN2NmNDBiMzAyNTYzOWNjNDdfeWpKMzVEUThkNjNoMTFTZGI5elpGUzQ5Y2hicDNGRVlfVG9rZW46VGhvYWJLMVVRb3puSVZ4NWxJY2NkRGJKbjRjXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### 方法

- 推理链压缩：在保持推理质量的同时，减少生成的 token 数量。
    
- 长度预算控制：在推理过程中动态调整生成内容的长度，以提高效率。
    
- 系统切换与模型切换：根据任务需求灵活选择不同的模型或推理路径。
    
- 并行搜索：利用并行计算加速推理过程。
    

## DeepSpeed（zero 显存优化）

### 模型显存占用（bf16/fp16）（https://zhuanlan.zhihu.com/p/665172400）

#### 训练时

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MTZkZTU5ZTM4NmZiNmMwMWU5NDRjZDNjZDNhZmFjNmVfckJHRVZabkVzZGhab1BIejhvdlhNRFVTVTl6SWRHanFfVG9rZW46U0FYNmJlRGMyb2xvdzh4dFVBQmNZeW1JbmphXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWU0YTEzZjIwNDNiNWIzNjM1NTRkYWJmNzVjZDg3YTVfOGdpTWJXeG5qajZLaFJJdlRqR25BVDk1QmZlZDlRT2ZfVG9rZW46RFdTMGJwaVgybzZoSWh4bkZjTGM5Rjk5bm1iXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

可见激活值占大头。

优化方法如下：

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTMzMGY2ZjNmYTkwNzRhYjk0OWE1ZjE0MTk0YzhlNGRfSE0xekJPVEFQZHhmNTRnOXNWQTd0ZlZ2YllXaWdTTTFfVG9rZW46TkxHTGJ0QXdJb2dqbWN4MVc4WmNNNUZ6bjZmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

#### 推理时

一个经验法则是：推理时的峰值显存大致是模型参数显存的 1.5 - 2.5 倍（尤其在处理长序列或大批次时）。更精确的估计需要结合具体模型和输入

- 输入/输出的 Token 存储：需要显存存储输入的 Token 嵌入（embedding）和生成的输出 Token。
    
- 中间激活值（Intermediate Activations）：前向传播过程中每一层的输出（如 Attention 的 Key/Value 缓存、FFN 的中间结果等）。
    
- Key-Value 缓存（KV Cache）：自回归生成时，为避免重复计算历史 Token 的 Key/Value，需缓存这些中间结果（显存占用与输入+输出长度成正比）
    

  

### zero3

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MDhiYWZiNDg4ZWM2OTE0ZjMwZGNiZjNkNGMyMzZlM2ZfakI3MnlCd3h5RkRyUGRlV1NRZjhrMVB0ZkNVZ3lPMFdfVG9rZW46VldsRmJnSnJ1b1VxNW54M0pvYmN2MG5rbmJmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NWI2Njk5YjliYmNlZWEyMmM1MmIxYWI1NTEzNWZlMjJfT1BtRnZ2VWxQZ3dMNFExNXZpR0dYMlJZMFNQMnJjRDdfVG9rZW46QUdGRGIyOGdzb2ZET1d4eU1JaGM5T0dtbjNjXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

这是inter-layer + broadcast + reduce-scatter的实现方法，是很久之前官方的实现，现在官方的真正做法是：**intra-layer + all-gather+ reduce-scatter（与fsdp一致）**

  

## 位置编码（RoPE）

  

## LoRA（手撕）

  

## R1 相关

### 训练整体流程

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YjgxMjE5YzI4ZGMwNDQ5NTM5N2IxY2I3MjUyODRkOTJfcjBTbXhwQUxZZVZoQktZOXJtMU1SaG9nMFlmUm5KQjVfVG9rZW46R3R6UWJSN0xxb0RTMkV4N2Fyd2NvZjg4bk5lXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### aha monment

在训练过程中，研究团队使用数学问题来训练和评估模型的逻辑推理能力。在观察模型输出时，研究人员正是在下面这个数学方程的解题过程中捕捉到了一个引人注目的"顿悟时刻"，充分展现了模型通过强化学习自然获得的自主反思能力：

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YmQ2MjVkZWZlZmM4YmQ5NWY2YmY5ZGE1NWUxMzhjOGRfaVlWZnQ0cjUzMGJpUXlnTlMyU0hoZmxueHJpbEdSbmZfVG9rZW46T0txVWJWWUZEbzc1Z3B4SG9aZmNWYmgybm5iXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### k1 k2 k3

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MDBmMjNjYmU0YjhkOWYzMWFkNzI0ZDg0YTlkNDc3ZDVfeVIxUHVLUjBmM2hSVWlES3BtenNuUHlqMHFvNmxaZG1fVG9rZW46Q0p6RWJsWkk5b2h6ZVJ4MFFBRGN2d1FhbnFIXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### R1-zero

- qwen2.5-0.5B 模型，num_generations 为 2，gsm8k 数据集准确率 0.45489006823351025。num_generations 为 4，准确率为 0.47763457164518575
    

### Rule-based reward

- 准确性奖励（Accuracy Rewards）：答案是否正确
    

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=M2Y2NGU1NDFlMzU2NmRmOTViNWNhMWZjYmY3N2Y3OGNfeWlOVzVVOUo4THA0WTNZdUdGMzd3dGx5VDdCbXIzcndfVG9rZW46Vk11TGJpVnU3b1RZekx4RFNqVWNvREl4bmYxXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

- 格式奖励（Format Rewards）：是否有思考标签<think>
    
- 语言混合问题：对语言一致性进行打分
    

  

## 梯度累计

### 代码（https://zhuanlan.zhihu.com/p/423359955)

暂时无法在飞书文档外展示此内容

gradient_accumulation_steps 是梯度累积次数，累积几次，原本的 loss 就要除以几，这是为了对多个批次的数据的梯度做累积。

有人说，应该有这一行代码才算累加

暂时无法在飞书文档外展示此内容

这样理解是错误的。

要明白最重要的一点是，梯度累加，累加的并不是损失，而是根据损失得到的梯度。

  

梯度累计如何节省显存

- 减少瞬时激活值显存：每次仅处理小批量的数据，激活值显存占用降低为原来的 `1/k`（例如 `k=4` 时，显存占用降至 25%）。
    
- 复用显存：每次小批量计算完成后，释放当前激活值显存，供下一次计算使用（显存占用峰值始终为小批量对应的量）。
    
- 梯度显存不变：模型参数和梯度的显存占用与批量大小无关，因此不受影响（但需额外存储累积梯度的变量，这部分开销极小）。
    

  

## Adam 和 AdamW

  

## RLHF 相关（PPO、RLOO、REINFORCE++、ReMax、GRPO、SAC）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MGQ5NzAzZjc3MWY4OGJmYzMyNWFhNTY3Yjc1MDVjY2JfdWRyU1dQNXlLTndMZ2Z0bzVWbUJMSkIxb0VxWGRMYjdfVG9rZW46QUVvVmJUQzRub2Y4WEp4MkRaT2NPNTFzblFnXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### on-policy 和 off-policy（参考 https://zhuanlan.zhihu.com/p/346433931）

- Off-policy: the learning is from the data off the target policy（引自《Reinforcement Learning An Introduction》）。也就是说 RL 算法中，数据来源于一个单独的用于探索的策略（不是最终要求的策略）。（Off-policy 方法中不一定非要采用重要性采样，要根据实际情况采用（比如，需要精确估计值函数时需要采用重要性采样；若是用于使值函数靠近最优值函数则不一定））
    
- On-policy: the target and the behavior polices are the same. 也就是说 on-policy 里面只有一种策略，它既为目标策略又为行为策略。
    

总结：PPO 算法，虽然有 2 个 policy，用$\pi_{old}$采样去更新 pi，但是由于 pi_old 的参数是从 pi 复制的，本质上还是属于同一个策略。所以 PPO 是一个看起来很像 off-policy 的 **on-policy** 算法。

### online 和 offline

更新的数据是否是最新 agent 的模型生成的数据

- Online: 更新的数据为最新模型采样得到的（采样的数据一次性用完）
    
- Offline: 更新的数据为 x 次更新前模型采样得到的（采样的数据更新多次）
    

### online、offline、on-policy、off-policy

online 学习中可以是 on/off policy 的。而 offline 学习中除了第一次更新模型的学习可能是 on policy 的，之后的所有学习只有仍是 offline 的则一定是 off policy。

### critic 和 reward 区别（参考 https://www.zhihu.com/question/1900547615495545054/answer/1901411039406457541）

- reward model 评估整个 response 质量，给出整体奖励信号，无法直接映射到每个 token 的贡献。
    
- critic model 估计价值函数，预测未来可能获得的累积奖励，为策略更新提供稳定的 advantage 信号
    

reward 扮演的是**环境**的角色，而 critic 属于 llm 这个智能体的一部分，就好比在考试中，你自己检查卷子和老师给你打分的区别。

### clip 细节

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTBmMDI0ZDIxMjA3MjUwZWJiMjdmYzhlOTVjYWQ5MmZfc3F5MXNUOERrVzZ2dThKZFhaWlZPc1lndmNkR09HWmJfVG9rZW46UnBncGJQOTdqb2hYTFJ4b3QzaGNjZWFobm1iXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTc5M2M3Y2RhYjNkMmExMWZhZGZkZmVkNWQ2MzgyN2NfbGdLc2JTUDZVTzAwUUJQNXAza2JKNGh6MzlpbHRScmNfVG9rZW46TDdDTWJaMG5Qb0RiaFV4RnZEeGNQU1hZbmxnXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### PPO 和 GRPO 的区别

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YWY0Mzc3MmI3ZWY0NTVhZTYwZjJhOGY5OTZmNDhlMTdfd2FSTnNwaWxTYUdNYW5SRmF2Y0lLRTdrMjJBeUJibG5fVG9rZW46V0dzR2JYbG9qb3pCYmN4YTRLa2N6TG9JbjBmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

#### 优势值区别

#### kl 散度区别

- PPO 的 KL 计算是在每个 token 的生成过程中发生的，不断计算当前 token 和 ref_model 的 KL 散度。
    
- GRPO 的 KL 计算是在一个回答生成结束后发生的，一次性对句子中的每个 token 计算 KL 散度，并参与最终 loss 的计算；
    
- PPO 的 KL 塞到 reward 里（reward shaping）
    
- GRPO 的 KL 是独立的损失项。
    
- advantage 角度来看，PPO 的 KL 惩罚是 token-level 的
    
- GRPO 的 KL 惩罚是 sentence-level（但是也是逐个 token 算 kl 再取 mean）的
    

### REINFORECE++

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YTY3NmIzYzM1NTM5MGZkMTdmNjlkMjUzN2Y3Y2Q1MTFfazM3Nm1zZFJjNjY2amxMQkZFS05WOVJlV3hBY2l4UFZfVG9rZW46SDMwdGJzajNCb2lCc2t4NFp6d2NvRnZqblFoXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### RLOO（REINFORCE leave-one-out）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YTE3MjllMTI2Nzk3YjZmZGM5OGJkOGY0M2I0NDdmYmVfTUdya09tWFBRMDFvTktLM09ZNGtIbGlJdHNxQjN0VEhfVG9rZW46R0FIUGJXTTBVb09mNzB4VFdaUGNJd0k2bjFnXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### ReMAX（REINFORCE argmax）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTQ2YjUxOGZjMzIyZDg2MTdkZWRkNzg4Mjk0ZTIxMjFfb3hMdXBOMjZra3hWQmNlWWt3WG11RHduVDdrUThlbTZfVG9rZW46RWtLVGJjREFEb0ZQc0l4UGVoR2NyMjgzbmdmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### DPO

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=YjdlMWUyYzdiM2ZiMjdkYTg2NjQ2MjZmYjNhMmViZjhfQXJIWmZtMzhIV0NiSVd5NU91VTRqN0JtVllJTDFEcHlfVG9rZW46Unpsa2JKWXRxb2s4OWd4bFdqaWNJend5blZmXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NTg0YWJkZjYyNDI1MmU3YjcwNjA2NjA3YTA4ZDI5ZmVfdzdlR2k1ZXM2SmJOZ3E0U2xBNENxdGlLZWx4RGQ2aW1fVG9rZW46TEttMWJLVjlTb3RBMEt4MXNEMWNlcm5QbnJjXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

a变大，b变小，理想情况，chosen response概率提升，rejected response概率下降

## 针对GRPO的改进

### DAPO（参考 https://www.zhihu.com/question/1895273986537014226/answer/1899582779408245950）

DAPO 是对 GRPO 的改进。DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，即解耦裁剪和动态采样策略优化）的优化点有四个（其中前 2 个是主要亮点，是命名的来源）：

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=MmM5MWQwMjBkMzU1MzEwOWYzZTY2NTAwNjk0M2Y5ZDVfZTFJa3RNVGlJMTVJa2o5ZHhGSUNMclVsRFQ3MkxuMHZfVG9rZW46Uk41RWJhWWgzbzMxUHh4dnVRZmNONnY2bjhkXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

#### 更高裁剪

#### 动态采样（Dynamic Sampling）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=N2Q1ZTBkNTEyNmIxNWE0Nzc3Nzc1ODg3YTE0ZDRlMzZfUUJOcGhPRVVZSnRUaTZrSFVpTUxoUlpwNFdDVDVPS0lfVG9rZW46RUx6MGJFTVMwb3pxV2V4MFVNbWNJankxbjVlXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

#### Token 级策略梯度损失（Token-Level Policy Gradient Loss）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjZiNTE4NzI5ZTk4NDYxMTY5MDBmNThiOWJlMjVlZjJfeERiMmlTc0JUbDAxWnhFYklMQnV5NXd0OUJwR2xrNlZfVG9rZW46Q1o4S2JYUkU0b1VaZHR4NVdqUGMwbGtoblRkXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

#### 超长奖励塑造（Overlong Reward Shaping）

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=NGU2N2U4MGFhOTBlOGE4M2Y4ZTM2ZDFkNTFmMTIyYTNfYXBIa1g5N3JrVFRVNjR0Z0xLSm5nMW84QlV0NklrR0tfVG9rZW46U2lYV2JVamJDb045TEp4bk8wcGM1RG9IbkVjXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### DR.GRPO

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDNjY2FhYmMyODQ3MTQyYmE0ODMwZWQ0NDQwYzhhYTFfVlUwSVByTElRNG91MllRbjVwYVg1SnUxQncyUExXMEVfVG9rZW46WmlhOWJPYjJRb2t3cEl4RDRlWmNNekUxbnhlXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

### GPG

GPG彻底使用policy-based的方法，去除了其他的PPO小trick

![](https://q0r4cw0t2o.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDFhODQxMDlhYjM2YzMwZDNiNDQ3OTdkYjMxZGYxNDVfamY0MjFpWTFkNmhzSW9Ia1VudG85aHpMZzRWbTE3T3BfVG9rZW46Q2FNZ2JYU2pLb2RMTEF4ZDFoWmNZVXR0blpnXzE3NTE0NjEyMDQ6MTc1MTQ2NDgwNF9WNA)

## FP16 和 BF16 的区别

  

## Tokenize 相关

  

  

  

## 面经

[大模型微调经验](https://zhuanlan.zhihu.com/p/690824731?utm_psn=1808294053495853057)

[ai 八股](https://github.com/WeThinkIn/AIGC-Interview-Book/tree/main?tab=readme-ov-file)
