# Review：自进化Agent的RL Memory与Skill方法批判


> 一篇对当前 "用 RL 提升 memory / skill refinement" 这条线的批判性综述。作者把现状分成 4 类：SkillRL、MemRL、纯 prompt memory、search-augmented RL。本笔记整理作者的核心观点,并与 vault 内已有的相关论文笔记交叉链接。

## 总论点

**真正的"真问题"不是 memory 本身,而是 credit assignment。**

- 在 agent RL 中,难点不是"存点东西",而是**哪一步该被奖励,哪段经验对哪一步有用**。
- 当前大量 "self-evolving / 自进化" 的工作,核心 memory / skill 主要靠 prompting 或外部模型生成,把"外部经验整理"说成"RL 学会自己整理出 skill"——**over claim 重于 RL 实质**。
- 普遍现象:benchmark 节节攀高,但 per-step latency、retrieval infra、memory poisoning / staleness、cache invalidation、rollout 成本、offline / online mismatch 几乎都不讨论。baselines 也偏弱(no-memory、naive full-context、passive semantic retrieval)。

> ⚠️ 关键公平性提醒: 既然号称"self-evolving",**就不应该在 runtime 之前对模型做黑箱 SFT**。一旦做了 SFT,模型就已经从历史轨迹中学到环境细节,根本没有 "evolve" 的意义。

---

## 一、SkillRL

**Paper:** [arxiv.org/pdf/2602.08234](https://arxiv.org/pdf/2602.08234) — Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

### 公开定位
原始 trajectory → 蒸馏成层级 SkillBank → RL 训练时检索进 prompt。在 ALFWorld、WebShop 和 7 个 search tasks 上优于 baseline。

### 值得肯定
- **真问题**: raw trajectory 太噪、太长、太贵。抽象成 rule / heuristic 再注入,确实比直接塞原始日志更省 token、更高效。
- 蒸馏方向是对的。

### 但水分很大
1. **"self-evolving" ≠ RL 学 skill**
   - SkillBank 是通过 experience-based distillation 构建和更新的。
   - 不是 end-to-end 把 skill writer / retriever 放进 policy gradient 里学。
   - 本质是**外部抽取 + prompt 注入**,不是 RL 学 skill。
   - 代码里用 gpt-o3 这种闭源大模型做 skill 蒸馏,效果能不好吗?
   - **更像是 experience distillation pipeline**,不是"RL 学 skill"系统。
   - 有效部分主要来自表示层的 abstraction,不来自 RL 机制本身。

2. **评测不公**
   - 没有 test set,直接用 validation set 评测。
   - 失败的 cases 拿去更新 skill bank,接着再评测。
   - 跟 Reflexion 类似的"路数",现实里几乎不可能这样做。

### 相关工作
- **SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization** — 其实是 GiGPO (verl-agent) 项目上的一点魔改,加了 memory 机制。沿用 GiGPO 的那几个 benchmark。
- **GiGPO 原项目**: alfworld 只评测了 in-distribution split,**没有测 out-of-distribution split**。后续这些 incremental 文章一律避而不谈。

---

## 二、SLEA-RL

**Paper:** [arxiv.org/pdf/2603.18079](https://arxiv.org/pdf/2603.18079) — Step-Level Experience Augmented Reinforcement Learning for Multi-Turn Agentic Training

### 核心主张
在多轮环境里,task-level retrieval 会失配,所以要做 **step-level retrieval**,并配合 **step-level advantage**。

### 方向上对的两个点
1. **retrieval 与 state 真正绑定** — 不再跟初始 instruction 锁死,而是跟当前 observation 走。这是对的。
2. **正面碰 step-level credit assignment** — 比很多"memory 论文"更重要的一步。

### 工业落地的顾虑
- **clustering 脆**: 用 observation clustering 做 cluster-indexed retrieval。论文环境里有效,但 UI 变、DOM 变、tool schema 变怎么办? observation 模态一换,cluster 还能沿用吗?
- **系统太重**: 同时引入 per-step retrieval、clustering / index、memory admission、step-level advantage estimation。latency 增加,failure mode 增加,**可解释性反而下降**。
- **对 benchmark 友好,对 production drift 敏感**: 论文环境 ≠ 线上环境,这种 mismatch 决定了很多"方向没错"的论文工业价值不高。

### 局限
- "self-evolving" 依然不是 fully learned memory。experience library 通过 semantic analysis 演化,而非 gradient update。
- 虽然比 SkillRL 更靠近 credit assignment,但 memory 演化本身仍**不是 end-to-end 学出来的**。

### 定位
适合回答"step-level retrieval 值不值?""credit assignment 能不能更细?"这类**偏 research prototype 的问题**,而非"线上 agent 怎么搭建"。

---

## 三、MemRL

**Paper:** [arxiv.org/pdf/2601.03192](https://arxiv.org/pdf/2601.03192) — Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory

### 公开主张
- **冻结 LLM**,避免改参数。
- 把 plasticity 放到 episodic memory 上。
- **Two-Phase Retrieval**: 先语义相关,再按 learned Q-value / utility 选。
- utility 通过环境反馈持续更新。

### 评价:开创了一个方法
- 至少在学一个**真正该学的东西**: 用环境反馈给 memory item 学 utility。
- 这比纯 embedding retrieval 前进了一大步。
- **语义相似 ≠ 决策有用**,表面相似 ≠ 可复用。MemRL 方向上更接近"真正的 memory RL"。

### 工业契合度
- 把"模型能力稳定"和"记忆可塑"分开,和工业实践很像。
- 现实里不想频繁动基座模型(API 模型贵、合规要求高、灰度上线严格),把 learning 放到外部 memory / retrieval 层通常更现实。
- 工程上: 一个主模型 + 一个 embedding provider + API 接口就能跑起来,除 vllm 推理外不需要 GPU 算力,**实作可行**。

### 严重问题
1. **utility / Q-value 估计依赖环境反馈,反馈稀疏时还是难**。
2. memory item 的 **non-stationarity** 很麻烦。
3. 论文省略了"sparse memory repo 到底能存多大,会不会爆炸"的讨论。

4. **最严重 — 高度怀疑是不是真的 RL**:

   在当前框架下,retrieval 路径基本是**确定性的**:
   - 第一阶段: 给定 query,embedding 确定的 → 相似度排序确定。
   - 从 query 扩展到 memory ID: top query groups 选出后,memory IDs 按既定规则查出,这一步也确定。
   - 第二阶段打分: candidate memories 给定后,similarity 和 Q 值固定,hybrid scoring 和排序也确定。
   - 最终选择: vanilla topk 是确定性的,tri-channel bucket selection 也是确定性的。

   在 `configs/rl_alf_config.yaml` 中 `epsilon = 0`(LLM temperature 也为 0),**没有 exploration**。

   **结论**: 这不是 RL 意义上的 exploration method,而是:
   - 对 memory bank 的 exploration 较弱。
   - 对 memory 的 credit coverage 也较弱。

5. **path-dependent 严重**:
   - Epoch 1 构建初始 memory bank。
   - Epoch 2 起,retrieval 已经受 Epoch 1 memory bank 强烈影响。
   - retrieval 基本 greedy,早期被选中的 memories 不断被重复使用和更新。
   - **Epoch 1 对后续训练轨迹的影响 disproportionately 大**。
   - 后续 epochs 不是独立学习轮次,而是在已带偏置的 memory state 上做 refinement。
   - 后果: 早期好的 memories 被放大;早期差的、噪声 memories 也被锁定;大量 memories 始终没被充分测试。

---

## 四、Prompt-based memory (工业上最有用,学术上最容易被低估)

代表工作:
- **Mem0** — [arxiv.org/pdf/2504.19413](https://arxiv.org/pdf/2504.19413)
- **ExpeL** — [ojs.aaai.org/.../29936](https://ojs.aaai.org/index.php/AAAI/article/view/29936)
- **MemP** — [arxiv.org/abs/2508.06433](https://arxiv.org/abs/2508.06433) — vault 笔记: [[Memp：Exploring Agent Procedural Memory]]
- **SimpleMem** — [arxiv.org/abs/2601.02553](https://arxiv.org/abs/2601.02553)

### 共同形态
外部 memory store + retrieval + in-context guidance。都属于**prompt 注入流派**,不是端到端把 memory 学进参数。

### 为什么工业价值最高

满足工业最看重的四件事:

1. **不依赖训练**: 不需要每周重训 policy,不需要昂贵 rollout,不需要复杂 GRPO infra。
2. **模型无关**: 尤其适合黑盒 API 模型。ExpeL 动机里就明确提到 proprietary APIs 下无法做 parametric update。
3. **可控**: memory schema 可查; retrieval 可调; 错了能删; 支持灰度和回滚。
4. **延迟 / 成本可优化**: SimpleMem 这类把 token budget 当一等公民处理,工业里非常重要。

### 明显上限
- 不能真正改变 base policy。
- 只能"提醒"模型,不一定真学会。
- retrieval 质量差时会把噪声喂进去。

### 落地建议
> **今天最该先上什么**: Mem0 / SimpleMem 这类外部 memory 系统,远比 fancy 的 RL-memory 联训更值。
> 不是因为理论更强,而是满足工业最重要的指标: ROI、可控性、维护成本、上线速度。

vault 内已有相关笔记:
- [[Mem0：Building Production -Ready AI Agents with Scalable Long-Term Memory]]
- [[Memp：Exploring Agent Procedural Memory]]
- [[ReasoningBank：Scaling Agent Self-Evolving with Reasoning Memor]]

---

## 五、Search-augmented RL

代表工作: **ZeroSearch**、StepSearch、EvolveR

### 定位
让模型在 step-by-step reasoning 中**主动生成搜索查询**,用 outcome-based reward 做 RL。学的是 search / retrieval 本身,不是积累经验记忆。

### 适用场景
open-domain QA、citation-heavy search、product / document search、multi-hop retrieval。

### 优点
- reward 比较干净(rule-based: 是否搜到、是否答对、是否命中)。
- 没什么 delay,适合可验证检索任务。

### 局限
- 现实场景复杂得多: action correctness、tool parameterization、session state drift、auth / permission、UI fragility、exception handling。
- **学会检索 ≠ 学会 agent**。

vault 内已有:
- [[ZEROSEARCH：Incentivize the Search Capability of LLMs without Searching]]
- [[Search-R1：Training LLMs to Reason and Leverage Search Engines with RL]]

---

## 六、整体批判

1. **灌水 / 噪音多** — 普遍披着 "self-evolving RL" 的 over claim 外衣讲故事。
2. **核心机制靠 prompt / 外部模型**,但表述成"RL 学会自己整理出 skill"。
3. **benchmark 节节攀高**,但 per-step latency、retrieval infra、memory poisoning、cache invalidation、rollout 成本、offline / online mismatch 几乎都不讨论。
4. **baselines 偏弱** — 常去比 no-memory、naive full-context、passive semantic retrieval 这种明显处于劣势的方法。
5. **最严重的公平性问题**: 既然号称 self-evolving,就**不应该在 runtime 之前做黑箱 SFT**。一旦 SFT,模型就已经从历史轨迹中学到了环境细节(好轨迹 / 坏轨迹、各种状态、工具 schema 都在 SFT 数据里),根本就没有 "evolve" 的意义。

---

## 七、本人总结

四类方法对比一览:

| 类别 | 核心思路 | RL 实质 | 工业落地 | 代表论文 |
|---|---|---|---|---|
| **SkillRL** | 蒸馏 trajectory → SkillBank → prompt 注入 | 弱(主要是 experience distillation) | 低 | SkillRL, SKILL0 |
| **SLEA-RL** | step-level retrieval + step-level advantage | 中(碰 credit assignment) | 中低(系统太重) | SLEA-RL |
| **MemRL** | 冻结 LLM,学 memory utility (Q-value) | 中(epsilon=0, exploration 弱) | 中(实作可行,但 path-dependent) | MemRL |
| **Prompt memory** | 外部 store + retrieval + in-context | 无 | **高** | Mem0, ExpeL, MemP, SimpleMem |
| **Search-augmented RL** | 学检索策略 + outcome reward | 中(rule-based reward) | 中 | ZeroSearch, StepSearch, EvolveR |

**两个判断标准**:
- **真 RL 还是假 RL**: 是否在 policy gradient 里端到端学 memory / skill writer / retriever? 还是只是 prompt 注入?
- **真公平还是假公平**: 有没有 test set? 有没有 out-of-distribution split? 有没有在 runtime 之前偷偷 SFT?

如果两问答案都偏负,这篇论文的工业价值大概率不如一篇朴素的 prompt-based memory 工作。

---

## 相关笔记

- [[RSI：Recursive Self-Improvement 分类图谱]] —— 同主题分类图谱：test-time / training-time RSI 的闭环逐层扩展，以及 "verifier 才是 closed-loop RSI 的真瓶颈"——可与本文 "credit assignment 才是真问题" 互证。
