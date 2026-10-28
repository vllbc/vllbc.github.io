# verl V1 主线重构 · 给后训练工作者的三问


> 来源：[verl 主线 V1 重构 · 给后训练工作者的三问](https://syt-nju.github.io/files/visual-reading/verl-v1-mainline-pipeline/)
>
> 仓库：[verl-project/verl](https://github.com/verl-project/verl)
>
> 关联笔记：
> - [[Engineering/coding/verl/verl总体概览]]
> - [[Engineering/coding/verl/agent_loop]]
> - [[Engineering/coding/verl/veRL_Ray执行流程源码解析]]
> - [[Engineering/coding/verl/veRL_colocate实现源码解析]]
> - [[Engineering/coding/verl/DataProto]]

verl 主线 pipeline 大改：我该关心的三件事
================================

主入口 `main_ppo.py` 已迁到 V1：**TransferQueue 数据仓库 + AgentLoop 服务化 rollout + 可插拔 trainer 模式**，默认 `trainer.use_v1=true`。本文只回答三个真正关心的问题——为什么改、老脚本还能不能跑、我的自由度提高了多少。

- **范围：** V0.8.0 前后主线重构
- **关键 PR：** #6067 · #6710 · #6823
- **入口命令：** `python -m verl.trainer.main_ppo`（未变）

## TL;DR · 一屏结论

1. **为什么改：** 老主线假设「同步、定长、单轮」，为让 Agent（多轮/工具/变长/异步 off-policy）成为一等公民，把生成与训练解耦。
2. **老脚本还能跑吗：** 能。入口命令没变，默认 `use_v1=true + trainer_mode=sync` ≈ 老同步 PPO；常规 GRPO/PPO 脚本基本原样可跑。想要异步只是加几行配置。
3. **自由度：** 大幅提高。生成逻辑 / 采样策略 / 调度模式 / PPO 每个阶段都成了可插拔扩展点——多数定制不用再碰主循环。

---

## Q1：为什么要动主线？目的是什么？

一句话：老 pipeline 的三条隐含假设（同步、定长、单轮）挡住了 Agent 训练。大改就是把这三条拆掉。

- **吞吐**：生成与训练可并行，慢样本（多轮/长）不再拖住整批 GPU
- **Agent**：多轮对话 + 工具调用 + 变长/多输出轨迹成为一等公民
- **可插拔**：同步↔异步、on↔off-policy、各阶段实现都变成可替换配置

### 老范式 V0 的局限

- **同步生成**：trainer 阻塞等一整批 rollout 完成才继续。
- **定长批次**：一个大 tensor batch 在 trainer 内流转，假设样本形状规整。
- **单轮为主**：prompt→response 一来一回，rollout 与 trainer 共享进程/显存。
- **trainer 什么都管**：worker、资源池、数据集、训练循环全塞进一个 `RayPPOTrainer`。

### Agent 训练 V1 的需求

- **变长 + 多输出**：一条轨迹可能几十轮、带工具返回，每 session 多个 output。
- **异步 / off-policy**：允许用「稍旧权重」生成的轨迹继续训练，换取吞吐。
- **服务化 rollout**：推理引擎（vLLM/SGLang）作为独立 server，AgentLoop 像调 API 一样发多轮请求。
- **解耦数据通道**：生成侧与训练侧各自并行，中间要一个能存变长数据、带元数据的「仓库」。

### 图 A · V0 单体同步 → V1 解耦异步

> **V0 · RayPPOTrainer（单体 · 同步）**
> - Dataloader → Actor + Rollout（共卡）→ PPO 更新
> - 同一进程 · 同步 generate
> - 串行循环：① generate（阻塞整批）→ ② 训练 → ③ update → 下一轮
>
> **V1 · 生成 / 数据 / 训练 / 权重同步 四块解耦**
> - Dataloader → AgentLoopManager（fire-and-forget · 每样本一 async 任务）→ LLM Servers（vLLM / SGLang 副本）
> - TransferQueue：KV 仓库 · `{uid}_{sid}_{idx}` · tags(staleness)
> - `PPOTrainer.step()`：actor / critic / ref · 读字段→算→写回
> - CheckpointEngine：异步权重同步
>
> **说明：**上半（V0）虚线大框代表「一个进程内的 RayPPOTrainer」，Dataloader → 共卡的 Actor+Rollout → PPO 更新，串行循环必须等一整批 generate 阻塞完成才能训练，rollout 和训练抢同一批 GPU。下半（V1）生成侧：Dataloader → AgentLoopManager → LLM Server 副本，慢样本不再拖住快样本。中间深色块 TransferQueue 是唯一数据交汇点，key 形如 `{uid}_{sid}_{idx}`。下排训练侧 `PPOTrainer.step()` 与 TransferQueue 双向交互。右侧橙色虚线 = 权重同步，训练完的权重经 CheckpointEngine 异步回灌给 LLM Servers，这是生成侧与训练侧唯一的强耦合点。

V1 把老的「大 trainer」拆成三个组件：**生成侧**（AgentLoopManager + LLM Server）、**数据桥**（TransferQueue）、**训练侧**（PPOTrainer + CheckpointEngine）。

---

## Q2：老脚本还能跑吗？怎么快速适配？

**先给定心丸：**入口命令**没变**——还是 `python3 -m verl.trainer.main_ppo ...`。默认 `trainer.use_v1=true` + `trainer.v1.trainer_mode=sync`，行为≈老的同步 PPO。仓库里 `examples/grpo_trainer/*.sh` 没写 `use_v1` / `trainer_mode`，照样跑。临时退回旧实现：`trainer.use_v1=false`（走 `main_ppo_v0.py`），但它 deprecated、**计划 v0.9.0 删除**。

### 启动脚本新增旋钮

| 配置项 | 默认 | 作用 |
|---|---|---|
| `trainer.use_v1` | true | 走 V1 主线；false 回退旧 `main_ppo_v0.py`（将删） |
| `trainer.v1.trainer_mode` | sync | `sync` / `colocate_async` / `separate_async` |
| `trainer.v1.sampler.max_off_policy_threshold` | 8 | 一条轨迹最多允许跨几个权重版本 |
| `trainer.v1.sampler.max_off_policy_strategy` | drop | 超阈值：`drop` 丢弃 / `wait` 等待 |
| `trainer.v1.colocate_async.num_warmup_batches` | 1 | 异步前预热多少批生成 |
| `trainer.v1.separate_async.num_warmup_batches` | 4 | 分离异步的预热批数 |
| `trainer.v1.separate_async.parameter_sync_step` | 4 | 每几步把新权重同步给 rollout |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | single_turn_agent | 默认用哪种 agent loop |

### 同一个脚本：同步照跑；想开异步只加两行

```bash
# 常规同步（等价旧行为，无需任何改动）
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=... actor_rollout_ref.model.path=... \
    trainer.n_gpus_per_node=8 trainer.nnodes=1
# 默认已是 trainer.use_v1=true, trainer.v1.trainer_mode=sync

# 想开异步（共卡 + partial rollout）：追加两行即可
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1
```

### 迁移清单

- **直接子类化旧 worker**、依赖 `verl/interactions`、curriculum sampler、dynamic dataset、旧 tool examples：在 #6067、#6302 等 **[BREAKING]** 里被废弃/迁移，需改到 engine 与 AgentLoop 体系。
- **自定义过 rollout / 生成逻辑**：改写一个 `AgentLoop`（见 Q3 骨架）。
- **reward / critic**：reward 走 `RewardLoopManager`，配置路径 `reward.reward_model.*`。
- **`separate_async` 硬约束**：`train_batch_size == ppo_mini_batch_size`、`checkpoint_engine.backend` 必须非 naive（nccl/nixl/mooncake）、colocate reward 不支持（需 `reward.reward_model.enable_resource_pool=true`）。

### 选哪种 trainer_mode？

| trainer_mode | 布局 | partial rollout | 权重同步时机 | 适用场景 |
|---|---|---|---|---|
| sync | colocate（共享显存） | 关闭 | 每步：sample 后 sleep，train 后 update | 单轮/短轨迹、要稳要复现（默认） |
| colocate_async | colocate | 开启 | 每步 update + `resume_generation`；带 warmup | 共卡想提吞吐、容忍轻度 off-policy |
| separate_async | 物理分卡 + 空闲切换 | 开启 | 按 `parameter_sync_step` 周期同步 | 大规模 / 长 agent 轨迹、最高吞吐 |

### 图 B · 时间线视角

- **sync**：生成与训练完全不重叠 · 最 on-policy · partial rollout 关闭
- **colocate_async**：共卡 · 生成（partial rollout）在后台持续 + warmup 批次 · 与训练时间上重叠 · 轻度 off-policy
- **separate_async**：分卡 · Rollout pool 持续流式生成 + 持续训练 · 两组 GPU 全程并行 · 按 parameter_sync_step 周期同步权重

三行**主流程代码完全相同**，差异只在 `on_sample_end` / `on_step_end` 两个钩子里怎么调 CheckpointEngine。

### GPU 分配：共卡 vs 分卡

三种模式中**真正改变 GPU 分配的只有 `separate_async`**。

- **sync / colocate_async**：1 个池（训练卡 + rollout hybrid 时分复用），卡数 = `trainer.n_gpus_per_node × nnodes`
- **separate_async**：2 个池（独立 rollout 池 + 训练卡空闲兼职），卡数 = 训练池同上 **+** `rollout.n_gpus_per_node × rollout.nnodes`

### separate_async 三条硬约束的根因

1. **`train_batch_size == ppo_mini_batch_size`**：全异步要求每个训练步恰好推进一个策略版本——一次 `sample()` → 一次梯度更新 → 同步一次权重。若一批被拆成多个 mini-batch 做多次 SGD，一个 step 内策略就跳了好几个版本，staleness 记账失真。
2. **`checkpoint_engine.backend != naive`**：`naive`（ColocatedCheckpointEngine）只做同卡 colocated 的 `all_gather`。分卡后必须换成 nccl/hccl、nixl/mooncake 等跨设备/跨节点后端。
3. **colocate reward 不支持**：colocate reward 靠「rollout 睡觉、腾出显存」窗口临时加载 reward model。`separate_async` 的独立 rollout 卡从不停机，没有这个窗口，所以 reward model 必须自己占独立卡。

### 异步流水线：塞 1 取 1 + 预热

V1 主循环是生产者/消费者流水线：每步 `_add_batch_to_generate()`（非阻塞）塞入 1 批 prompt；`replay_buffer.sample()`（阻塞轮询）等到够 1 批才取走（优先取最旧的压 staleness）。

预热深度对比：

| 模式 | num_warmup_batches 默认 | 原因 |
|---|---|---|
| sync | 无（0） | 刻意不重叠 |
| colocate_async | 1 | 共卡浅浅重叠一步 |
| separate_async | 4（+ parameter_sync_step=4） | 分卡全程并行，流水线更深 |

**一句话：**预热深度 ≈ 流水线深度 ≈ off-policy staleness 预算。

### colocate_async 一个 step 的解剖

1. **塞入下一批 prompt**（非阻塞，rollout 后台开跑）
2. **`sample()`** 取走够一批的完成轨迹
3. **`on_sample_end`**：`abort_replicas()`（partial rollout）+ `sleep_replicas()`（rollout 交出显存/KV cache）
4. **训练各阶段**：此刻共享的卡归训练独占
5. **`on_step_end`**：`update_weights()` + `resume_generation_replicas()`（带新权重续跑半截轨迹）

**快速选型：**先 `sync` 跑通（几乎零改动）；生成把卡闲置严重、能接受轻度 off-policy，再上 `colocate_async`；轨迹很长/规模很大、且愿意配非 naive 权重后端，才上 `separate_async`。

---

## Q3：训练流程的自由度提高了多少？

V1 把「生成逻辑 / 采样策略 / 调度模式 / PPO 每个阶段」做成了可插拔扩展点。多数定制只需写一个子类 + 一行配置，**不用碰主循环**。

| 想改的东西 | 怎么接入 | 入口 / 配置 |
|---|---|---|
| 生成 / 多轮 / 工具逻辑 | 写一个 `AgentLoop` 子类 | `@register` + `rollout.agent.agent_loop_config_path` |
| 整个 rollout 调度器 | 自定义 `AgentLoopManager` | `rollout.agent.agent_loop_manager_class` |
| 采样 / off-policy 策略 | 自定义 `ReplayBuffer` 子类 | `trainer.v1.sampler.custom_sampler.{path,name}` |
| 调度模式（同步/异步布局） | 注册新 `trainer_mode` | `@register_trainer` + 生命周期钩子 |
| PPO 单个阶段（adv / logprob / …） | override `PPOTrainer._compute_*` | 子类 + `register_trainer` |
| off-policy 修正（IS / decoupled） | 配置 `rollout_correction` | `algorithm.rollout_correction.*` |
| reward（规则 / 模型） | `RewardLoopManager` / reward server | `reward.reward_model.*` |

**PPO 每个阶段都是独立方法：**

```python
# verl/trainer/ppo/v1/trainer_base.py :: PPOTrainer.step()
_add_batch_to_generate → replay_buffer.sample → [_compute_reward_colocate]
  → _balance_batch → _compute_old_log_prob → [_compute_ref_log_prob]
  → [_compute_values] → _compute_advantage → [_update_critic] → _update_actor

# 生命周期钩子：
on_train_begin / on_step_begin / on_sample_begin / on_sample_end / on_step_end
```

### 示例 1：改「生成逻辑」· 写一个 AgentLoop

内置参考：`single_turn_agent_loop.py`（单轮）、`tool_agent_loop.py`（ReAct 多轮 + 工具）。

```python
# my_agent_loop.py
from typing import Any
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopOutput, register,
)

@register("my_agent")
class MyAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)

        # 多轮就在这里 while 循环：生成 → 解析工具调用 → 追加观测 → 再生成
        out = await self.server_manager.generate(
            request_id=uuid, prompt_ids=prompt_ids, sampling_params=sampling_params,
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=out.token_ids,
            response_mask=[1] * len(out.token_ids),  # 工具返回段可置 0
            num_turns=2,
        )
```

接入：

```yaml
# my_agents.yaml
- name: my_agent
  _target_: my_agent_loop.MyAgentLoop
```

```bash
actor_rollout_ref.rollout.agent.agent_loop_config_path=my_agents.yaml \
actor_rollout_ref.rollout.agent.default_agent_loop=my_agent
```

### 示例 2：改「采样 / off-policy 策略」· 自定义 ReplayBuffer

```python
# my_sampler.py
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

class MyReplayBuffer(ReplayBuffer):
    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        return batch, metrics
```

```bash
trainer.v1.sampler.custom_sampler.path=my_sampler.py \
trainer.v1.sampler.custom_sampler.name=MyReplayBuffer \
trainer.v1.sampler.max_off_policy_threshold=8
```

### 示例 3：改「调度 / 权重同步时机」· 注册新 trainer_mode

```python
# my_trainer.py
from verl.trainer.ppo.v1 import PPOTrainer, register_trainer

@register_trainer("my_mode")
class MyTrainer(PPOTrainer):
    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas()
    def on_step_end(self):
        if self.global_steps % 2 == 0:
            self.checkpoint_manager.update_weights(self.global_steps)
```

接入：`trainer.v1.trainer_mode=my_mode`。改 PPO 阶段同理 override `_compute_advantage` 等方法。

---

## 番外 · 放到 slime / AReaL 里看

三家框架都在解同一道题——解耦生成与训练、加一个数据桥、异步同步权重、控制 staleness。差异主要在抽象哲学和解耦彻底程度。

verl 依赖的 `TransferQueue==0.1.8` 是可复用独立组件（Ascend 开源）；verl 的异步 off-policy 修正直接指向 AReaL 论文——`trainer_separate_async.py` 里写着 `# TODO: Support Decoupled PPO: https://arxiv.org/abs/2505.24298`。

### 图 C · 共同骨架

| 框架 | 生成 | 数据桥 | 训练 |
|---|---|---|---|
| verl | LLM Server + AgentLoop | TransferQueue | PPOTrainer（可插拔模式） |
| slime | SGLang + sgl-router | Data Buffer | Megatron 循环（裸写） |
| AReaL | rollout workers（可中断） | Replay Buffer | trainer workers（全异步） |

| 维度 | verl V1 | slime | AReaL |
|---|---|---|---|
| 数据桥 | TransferQueue | Data Buffer | Replay Buffer（用一次） |
| 解耦 / 布局 | sync / colocate_async / separate_async 三档 | 一个 `--colocate` 开关 | 完全分离集群 |
| off-policy 控制 | `max_off_policy_threshold` + rollout_correction | over-sample + APRIL | `max_head_offpolicyness` + decoupled PPO |
| trainer 抽象哲学 | 结构化基类 + 注册表 + 钩子 | 刻意不封装，循环裸露 | algorithm-first（AReaL-lite） |

### slime · 极简抽象

三模块 + Data Buffer（Megatron + SGLang + Data Buffer）；靠 Ray 的 `ray.get` 位置切同步/异步；训练循环裸露在入口脚本。slime 明确不用 trainer 类包裹；与 verl「结构化基类 + 注册表 + 钩子」正好相反。长尾优化（APRIL）：over-sample → 够了 abort 其余 → 半截轨迹存 buffer、下轮续跑。对应 verl 的 partial rollout + `min/max_global_steps`。

来源：[THUDM/slime](https://github.com/THUDM/slime) · [介绍博客](https://thudm.github.io/slime/blogs/introducing_slime.html)

### AReaL · 完全异步 + decoupled PPO

fully asynchronous：rollout worker 不等待、持续流式；trainer 攒够一批就更新，更新后同步权重回 rollout。两把稳定器：`rollout.max_head_offpolicyness`（=0 退化为同步，异步常用 2–8）+ `actor.use_decoupled_loss`（解耦行为策略 π_behav 与近端策略 π_prox）。论文报告相比同步系统最高 2.77× 加速且性能不降。verl `separate_async` 的 TODO 直指这篇；verl 的 `rollout_correction` 与 `bypass_mode`，本质就是这套修正在 verl 的落地。

来源：[inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) · [论文 arXiv:2505.24298](https://arxiv.org/abs/2505.24298)

---

## 一句话行动指南

- **先别慌：** 现有 GRPO/PPO 脚本大概率**照跑**——默认 V1 sync ≈ 老同步 PPO，入口命令没变。
- **要吞吐/长轨迹：** 加 `trainer.v1.trainer_mode=colocate_async`（或规模够大再上 `separate_async` + 非 naive 权重后端）。
- **要定制：** 优先走三个扩展点——**AgentLoop**（生成）/ **ReplayBuffer**（采样）/ **trainer_mode**（调度），别再改主循环。
- **别踩坑：** 若老代码子类化了旧 worker 或依赖 interactions / curriculum / dynamic dataset，需按 #6067 迁到 engine；`use_v1=false` 只是临时退路（v0.9.0 移除）。

---

*本页基于本地 verl 仓库源码（分支 `gsm8k` @ `5c24e81d`）梳理，面向已了解 verl 的后训练工作者。代码骨架基于当前源码 API，实验阶段接口可能微调，落地前请对照仓库确认。*

- **源码：** [verl-project/verl](https://github.com/verl-project/verl)
- **关键文件：** `verl/trainer/main_ppo.py` · `verl/trainer/ppo/v1/trainer_base.py` · `verl/experimental/agent_loop/agent_loop.py`
- **关键 PR：** [#6067](https://github.com/verl-project/verl/pull/6067) · [#6710](https://github.com/verl-project/verl/pull/6710) · [#6823](https://github.com/verl-project/verl/pull/6823)
- **对比框架：** [THUDM/slime](https://github.com/THUDM/slime) · [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) · [AReaL 论文](https://arxiv.org/abs/2505.24298)
