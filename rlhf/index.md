# RLHF

基础部分看猛猿大佬的[人人都能看懂的RL-PPO理论知识](https://zhuanlan.zhihu.com/p/7461863937)即可，通俗易懂，我写不出来比这个更好的了。本文是各RL算法笔记。

# PPO (openrlhf库)
重点记录一下experience的采集过程。训练其实很简单。actor 在 RLHF 会进行 auto-regressive decoding，而 critic, reward 和 reference 则只会 prefill，不会 decode。所以，我们将 actor 的推理特定称为 rollout，而其他模型的推理称为 inference。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604154855.png)

获取experience的总体流程：
```python
####################
        # 1. 调用Actor generate()方法获取Prompt的生成结果，把结果存储到Sample对象
        ####################
        samples_list = self.generate_samples(all_prompts, **generate_kwargs)
        torch.distributed.barrier()
        ####################
        # 2. 调用make_experience 对每个Sample做处理，组装Experience部分字段（除了advantage和return）
        ####################
        experiences = []
        for samples in samples_list:
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)
        ####################
        # 3. 通过从后往前回溯计算的方式，获取advantage和return值
        ####################
        for experience, reward in zip(experiences, rewards):
            num_actions = experience.info["num_actions"]
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
        return experiences
```
关于句子序列中的state和action定义如下：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604161135.png)
## prompt -> sample
首先对batch进行pad,注意推理时需要左pad。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604161337.png)
然后生成sequences，attention_mask, action_mask：
sequences：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604162039.png)
attention_mask： 
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604162132.png)
action_mask： 
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604162144.png)
至此sample数据就得到了。
## sample -> experience 
首先根据前面generate的prompt+response计算得到response部分的logp：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604163901.png)
同样的方式得到reference_model的logp，然后就可以计算kl散度。

Critic是预估状态的价值，看代码实现时，参考图3，先理解LLM中状态的起始位置。最终状态序列长度是num_actions(生成token的数量)，状态序列起始位置是Prompt的最后一个token，结束位置是最后eos token 前一个token， 所以计算出的Critic预估状态价值的数据为：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604165317.png)
注意，图里eos token和pad token的位置输出应该并不是0，是regression head的输出（小实数值），只是我们期望良好的价值函数在这些位置输出0

在RLHF中，Reward Model是一个ORM（outcome Reward Model） 也就是对完整的生成response输出一个打分。代码实现上取每个sequence eos token位置的预估打分值。如图11，图中"xx"也是会并行计算出的Reward值，单最终只取了序列最后eos位置的score作为完整序列的打分值。最后reward处理成[B, 1]格式，每个序列一个打分。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604165943.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604170000.png)
gae(Generalized Advantage Estimation)是PPO论文中实现的优势奖励值计算方法，可平衡优势预估的偏差和方差。结合公式和图片内容更容易理解：
```python
def get_advantages_and_returns(values: torch.Tensor, rewards: torch.Tensor,）
    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
        - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
        + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
```

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604170202.png)
计算returns：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604170231.png)

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250604170049.png)

此时我们就完成了experience的采集过程。
# GRPO (trl库)

## 重要参数

- num_generations: **Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.**
- generation_batch_size: **Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`.**
- steps_per_generation:Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps.
- num_iterations: Number of iterations per batch (denoted as μ in the algorithm).
- per_device_train_batch_size
- num_processes(world_size)

trl库的重要参数比较少。其中根据官方文档，generation_batch_size = `per_device_train_batch_size * num_processes * steps_per_generation
gradient_accumulation_steps一般就是steps_per_generation(对应verl中的mini_batch_size / n_gpus / ppo_micro_batch_size_per_gpu)，可以理解为per_device_train_bs(对应verl中的ppo_micro_batch_size_per_gpu)是使用梯度累计后的bs，乘gpu数，再乘梯度累计的steps就是总的batch_size（对应verl中的train_batch_size * rollout.n）。所以注意，总的batch_size(generation_batch_size) 是已经rollout采样后的bs，除以num_generations才是针对prompts的bs（verl中的train_batch_size）。
下面是_get_train_sampler方法的注释，对每一个prompt重复num_generations是该方法实现的。
```python
if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations, # 每个 prompt 生成 self.num_generations 个 completions
            # 例如，如果 per_device_train_batch_size=8, num_generations=2, steps_per_generation=4,
            # 则 generation_batch_size = 8 (per_device_train_batch_size) * 4 (steps_per_generation) = 32
            # 这里的 batch_size = 32 / 2 = 16，表示一个 "generation block" 中有16个不同的prompt。
            batch_size=self.args.generation_batch_size // self.num_generations,
            # 每个 "generation block" (包含16个不同prompt，每个prompt有2个completion) 会被用于 num_iterations * steps_per_generation 次更新
            # 例如 num_iterations=1, steps_per_generation=4, 则这个 block 会被重复 1*4=4 次，每次取出一个 per_device_train_batch_size 的数据进行训练
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )
```
结合下面的例子帮助理解，例子中梯度累计steps不等于steps_per_generation
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250602225722.png)

在GRPO_trainer中，最重要的方法是`_generate_and_score_completions` 方法，输入为input，输出为计算得到的优势值和old_logp用于计算ratio。一些核心的部分和注释如下：
```python
 with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    # prompt_ids: (B_gen_local, P_max)
                    # prompt_mask: (B_gen_local, P_max)
                    # prompt_completion_ids: torch.Tensor (B_gen_local, P_max + C_new), C_new 是 HF generate 生成的新 token 数量 (最大为 max_completion_length)
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1) # P_max
            # prompt_ids 保持不变: (B_gen_local, P_max)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            # completion_ids: torch.Tensor (B_gen_local, C_new_hf)
            completion_ids = prompt_completion_ids[:, prompt_length:]
```
上面为generate的过程，不过现在基本上使用vllm或者sglang加速推理。为了逻辑简单，这里展示了HF generate的过程。trl实现的时候，将一个prompt采样多次的逻辑实现在了get_train_dataloader方法中，即一开始就使用get_train_sampler方法对同一个prompt repeat了多次。因此这里不需要再进行repeat。
之后得到补充部分的mask:
```python
 # Mask everything after the first EOS token
        # is_eos: torch.Tensor (B_gen_local, C_new), C_new 是 completion 的实际长度 (C_max_vllm 或 C_new_hf)
        is_eos = completion_ids == self.processing_class.eos_token_id
        # eos_idx: torch.Tensor (B_gen_local,), 存储每个 completion 中第一个 EOS token 的索引，如果没有EOS则为序列长度
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # completion_mask: torch.Tensor (B_gen_local, C_new), 标记有效 token (EOS之前及EOS本身)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        # completion_ids_list: list[list[int]], 长度 B_gen_local, 移除了 padding 和 EOS 之后的 token
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]
```
然后根据generate的ids得到old_logp:
```python
with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                # old_per_token_logps: torch.Tensor (B_gen_local, C_new), 代表生成这些 completion 时所用模型的 log probabilities
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None # 在特定条件下，可以用当前模型的 logprobs.detach() 代替，以节省计算
```
在上一步 generate 的时候我们不是已经进行过完整 batch 的推理了么？为什么现在还要重复进行一次 forward 来计算 log_prob，而不是在 generate 的过程中就把 log_prob 保存下来？  
  
因为forward的时候和generate的时候logprob由于推理引擎和训练引擎的优化目标不一样，会造成两者对不上，因此需要做两次。
batch 算子的细微差异，都会造成这两个 log_prob 不完全一致。推理引擎要的是快速出 token id，训练引擎需要保证一定的log_prob 精度。

注意这里的很关键的一点是如果符合分支条件将old_logp设置成了None，那么后续计算ratio时就固定为1（old_logps  = logps.detach ）。如果num_iterations > 1，说明一批数据会被训练多次，ratio就不固定为1了。所以要保存生成训练数据的那个模型对应的logps。steps_per_generation > ga_steps也一样，因为steps_per_generation参数就代表一批数据训练多少次。ga_steps更新actor参数，这之后ratio就不为1了。一共要经历steps_per_gen / ga_steps参数更新。如果我们不设置steps_per_generation默认就是ga_steps，这里还是看这个图就可以理解了：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250602225722.png)

然后根据定义的奖励函数计算reward：
```python
for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    # GRPO一般不需要这部分
                else:
                    # 自定义奖励函数
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    # rewards_per_func[:, i]: torch.Tensor (B_gen_local,)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
```
得到了奖励，就可以计算组内优势了：
```python
  # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
rewards_per_func = gather(rewards_per_func) # (N_proc * B_gen_local, num_reward_funcs)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1) # (N_proc * B_gen_local,)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # (N_groups_total, G) -> (N_groups_total,)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # (N_groups_total, G) -> (N_groups_total,)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards)) # (N_groups_total,)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (N_groups_total * G,),即(N_proc * B_gen_local)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (N_groups_total * G,),即(N_proc * B_gen_local)
        advantages = rewards - mean_grouped_rewards
```
这样grpo训练所需要的experience就生产好了。下面进入训练阶段，计算kl散度：
```python
if self.beta != 0.0: # 仅当 beta 不为0时才需要计算 KL 散度
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            # per_token_kl: torch.Tensor (B_eff, C_new), 每个 token 的 KL 散度
            # KL(P || Q) = sum P(x) log(P(x)/Q(x)) 的一种估计形式
            # 使用 exp(log P - log Q) - (log P - log Q) - 1 来避免直接计算 P/Q 可能的数值不稳定
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            ) # k3
```
核心计算部分：
```python
# Compute the loss
        advantages = inputs["advantages"] # torch.Tensor (B_eff,)
        # old_per_token_logps: torch.Tensor (B_eff, C_new), 旧策略（生成数据时）的对数概率
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        # coef_1 (r_t(θ)): torch.Tensor (B_eff, C_new), 概率比率 exp(log_probs_new - log_probs_old)
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # coef_2 (clipped r_t(θ)): torch.Tensor (B_eff, C_new), 裁剪后的概率比率
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None: # GRPO 论文中的 δ 参数，用于额外限制概率比率的上限
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # per_token_loss1: torch.Tensor (B_eff, C_new), PPO 目标的第一项 r_t(θ) * A_t
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2: torch.Tensor (B_eff, C_new), PPO 目标的第二项 clip(r_t(θ), 1-ε, 1+ε) * A_t
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # per_token_loss: torch.Tensor (B_eff, C_new), PPO 损失的 surrogate 部分 -min(loss1, loss2)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            # 如果 beta 不为0，则加入 KL 散度惩罚项
            per_token_loss = per_token_loss + self.beta * per_token_kl
```
这样，我们就得到了补全部分的每一个有效token的损失。这次还可以加入entropy loss，指策略分布的熵 (Entropy)：策略对选择下一个动作（在这里是下一个 token）的不确定性程度。熵越高，表示策略输出的概率分布越均匀，选择各个动作的概率越接近，策略的探索性越强；熵越低，表示策略越倾向于选择少数几个高概率的动作，确定性越强。
`entropy_loss` 指 entropy 的 平均值，是一个标量，表示探索性高低。
得到token_loss后根据不同的方法计算batch损失：
```python
if self.loss_type == "grpo":
            # GRPO 论文中的标准损失：对每个序列的 token 损失求和后取平均，然后再对批次取平均
            # (sum_t (L_t * mask_t) / sum_t mask_t).mean_batch()
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            # BNPO (Batch Normalized Policy Optimization) 损失：对所有 token 的损失求和后，除以所有有效 token 的总数
            # sum_batch sum_t (L_t * mask_t) / sum_batch sum_t mask_t
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            # DR-GRPO (Dense Reward GRPO) 损失：对所有 token 的损失求和后，除以 (批次大小 * 最大完成长度)
            # sum_batch sum_t (L_t * mask_t) / (B_eff * C_max)
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
```

# DAPO
# RLOO

# REINFORCE++

# ReMax



## 学习路线

[如何理解Q值和V值](https://zhuanlan.zhihu.com/p/109498587)
(https://zhuanlan.zhihu.com/p/14569025663)


