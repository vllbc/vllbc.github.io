# grpo

# GRPO (trl 库)

## 重要参数

- Num_generations: **Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.**
- generation_batch_size: **Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`.**
- steps_per_generation: Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps.
- Num_iterations: Number of iterations per batch (denoted as μ in the algorithm).
- Per_device_train_batch_size
- Num_processes (world_size)

trl 库的重要参数比较少。其中根据官方文档，generation_batch_size = `per_device_train_batch_size * num_processes * steps_per_generation
Gradient_accumulation_steps 一般就是 steps_per_generation (对应 verl 中的 mini_batch_size / n_gpus / ppo_micro_batch_size_per_gpu)，可以理解为 per_device_train_bs (对应 verl 中的 ppo_micro_batch_size_per_gpu) 是使用梯度累计后的 bs，乘 gpu 数，再乘梯度累计的 steps 就是总的 batch_size（对应 verl 中的 train_batch_size * rollout. N）。所以注意，总的 batch_size (generation_batch_size) 是已经 rollout 采样后的 bs，除以 num_generations 才是针对 prompts 的 bs（verl 中的 train_batch_size）。
下面是_get_train_sampler 方法的注释，对每一个 prompt 重复 num_generations 是该方法实现的。
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
结合下面的例子帮助理解，例子中梯度累计 steps 不等于 steps_per_generation
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250602225722.png)

在 GRPO_trainer 中，最重要的方法是 `_generate_and_score_completions` 方法，输入为 input，输出为计算得到的优势值和 old_logp 用于计算 ratio。一些核心的部分和注释如下：
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
上面为 generate 的过程，不过现在基本上使用 vllm 或者 sglang 加速推理。为了逻辑简单，这里展示了 HF generate 的过程。Trl 实现的时候，将一个 prompt 采样多次的逻辑实现在了 get_train_dataloader 方法中，即一开始就使用 get_train_sampler 方法对同一个 prompt repeat 了多次。因此这里不需要再进行 repeat。
之后得到补充部分的 mask:
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
然后根据 generate 的 ids 得到 old_logp:
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
  
因为 forward 的时候和 generate 的时候 logprob 由于推理引擎和训练引擎的优化目标不一样，会造成两者对不上，因此需要做两次。
Batch 算子的细微差异，都会造成这两个 log_prob 不完全一致。推理引擎要的是快速出 token id，训练引擎需要保证一定的 log_prob 精度。

注意这里的很关键的一点是如果符合分支条件将 old_logp 设置成了 None，那么后续计算 ratio 时就固定为 1（old_logps  = logps. Detach ）。如果 num_iterations > 1，说明一批数据会被训练多次，ratio 就不固定为 1 了。所以要保存生成训练数据的那个模型对应的 logps。Steps_per_generation > ga_steps 也一样，因为 steps_per_generation 参数就代表一批数据训练多少次。Ga_steps 更新 actor 参数，这之后 ratio 就不为 1 了。一共要经历 steps_per_gen / ga_steps 参数更新。如果我们不设置 steps_per_generation 默认就是 ga_steps，这里还是看这个图就可以理解了：
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250602225722.png)

然后根据定义的奖励函数计算 reward：
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
这样 grpo 训练所需要的 experience 就生产好了。下面进入训练阶段，计算 kl 散度：
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
这样，我们就得到了补全部分的每一个有效 token 的损失。这次还可以加入 entropy loss，指策略分布的熵 (Entropy)：策略对选择下一个动作（在这里是下一个 token）的不确定性程度。熵越高，表示策略输出的概率分布越均匀，选择各个动作的概率越接近，策略的探索性越强；熵越低，表示策略越倾向于选择少数几个高概率的动作，确定性越强。
`entropy_loss` 指 entropy 的平均值，是一个标量，表示探索性高低。
得到 token_loss 后根据不同的方法计算 batch 损失：
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

# 常见问题

### GRPO 的初始 Loss 为0，整体趋势是从0上升后下降，这正常吗？当 loss 是0的时候，那岂不是没梯度了，还能正常训练吗？
当我们只执行一个step的时候，这时候 $\pi_{\theta_{\text{old}}}$和 $\pi_\theta$ 是相同的，所以目标函数可以被简化为：

$$

= \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_{i,t} - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

易得优势值（归一化的）相加后等于0。因此变成了：

$$
- \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

而在训练初期，目标策略和参考策略也是相同的，因此就导致了loss等于0。但这不代表梯度等于0

对GRPO的目标函数求导后，得到：

$$
\nabla_\theta J_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\hat{A}_{i,t} + \beta\left(\frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1\right)\right]\nabla_\theta\log\pi_\theta(o_{i,t}|q,o_{i,<t})
$$

中括号内的右边部分可以易得为0，这时候就变成了：

因为$\frac{1}{\mid o_{i} \mid} \sum_{t=1}^{\mid o_{i}\mid} \hat{A}_{i,t} = \hat{A}_{i}$
化简为：
$$
\nabla_\theta J_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[\hat{A}_i \cdot \left(\frac{1}{|o_i|} \sum{t=1}^{|o_i|} \nabla_\theta\log\pi_\theta(o_{i,t}|q,o_{i,<t})\right)\right] \neq 0
$$
因此loss为0不代表梯度为0。


### 为什么在GRPO训练后的模型会出现更倾向于长文本的回答呢？对于偏差是如何解决的呢？

参考[# GRPO为什么会使得模型的推理变长？](https://www.zhihu.com/question/1912123003191433189)

$$
  

\mathcal{J}_{\text{GRPO}}(\pi_\theta) = \mathbb{E}_{\mathbf{q}\sim p_Q,\{\mathbf{o}_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|\mathbf{q})} \left[\frac{1}{G} \sum_{i=1}^G {\frac{\color{red}{1}}{\color{red}{|\mathbf{o}_i|}}} \sum_{t=1}^{|\mathbf{o}_i|} \left\{\min\left[\frac{\pi_\theta(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}\hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right]\right\}\right]
$$

$$
\hat{A}_{i,t} = \frac{R(\mathbf{q}, \mathbf{o}_i) - \text{mean}(\{R(\mathbf{q}, \mathbf{o}_1),\ldots,R(\mathbf{q}, \mathbf{o}_G)\})}{\color{red}{\text{std}(\{R(\mathbf{q}, \mathbf{o}_1),\ldots,R(\mathbf{q}, \mathbf{o}_G)\})}},
$$

- 响应级别长度偏差：对积极的advantage，这种偏差导致较短的响应获得更大的梯度更新，从而使策略倾向于在正确答案中优先选择更简洁的表达。相反，对于消极的advantage，由于较长的响应具有更大的 |oi|，因此它们受到的惩罚较小，这导致策略在错误答案中倾向于选择较长的响应。
- 问题难度级别偏差：标准差较低的问题（例如，太简单或太困难的问题，结果奖励几乎全为 1 或 0）在策略更新时会被赋予更高的权重。问题级归一化导致不同问题在目标函数中的权重不同，从而在优化过程中产生了难度偏差。

如果是答对的样本，因为模型鼓励正确的样本，短的回复token权重更高，这时候模型会倾向于回复更短

如果是答错的样本，因为模型要避免错误的样本，短的回复token权重更高，这时候模型会避免回复太短，也就是倾向于回复更长。

因此可以看出，在一个step中，样本回答的正确率，会对模型回复的长度倾向有一定的影响。

1. 在step正确率高的情况下，模型的回复可能会倾向于越来越短
2. 在step正确率低的情况下，模型的回复可能会倾向于越来越长

  
DAPO的token-loss解决了这个问题。

### PPO 、GRPO、DAPO 的区别是什么? 它们依次进行了那些改动，说出他们的异同？
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250731132607.png)
- PPO 的 KL 计算是在每个 token 的生成过程中发生的，不断计算当前 token 和 ref_model 的 KL 散度。
- GRPO 的 KL 计算是在一个回答生成结束后发生的，一次性对句子中的每个 token 计算 KL 散度，并参与最终 loss 的计算；
- PPO 的 KL 塞到 reward 里（reward shaping）
- GRPO 的 KL 是独立的损失项。
- advantage 角度来看，PPO 的 KL 惩罚是 token-level 的
- GRPO 的 KL 惩罚是 sentence-level（但是也是逐个 token 算 kl 再取 mean）的。

DAPO与GRPO的区别详见[dapo](dapo.md)
### K1、K2、K3分别是什么以及区别？

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250731131801.png)


# 参考

[Some simple thoughts on GRPO](https://khazzz1c.notion.site/Some-simple-thoughts-on-GRPO-23fd29780b5880459892eea775682df1)
