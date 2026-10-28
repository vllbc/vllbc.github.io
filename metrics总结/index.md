# metrics总结

# 训练指标完整指南

## 目录

1. [Actor 指标 (策略网络)](#1-actor-指标策略网络)
2. [Critic 指标 (价值网络)](#2-critic-指标价值网络)
3. [奖励与优势指标](#3-奖励与优势指标)
4. [序列长度指标](#4-序列长度指标)
5. [Episode 指标](#5-episode-指标)
6. [时间性能指标](#6-时间性能指标)
7. [系统性能指标](#7-系统性能指标)
8. [训练控制指标](#8-训练控制指标)
9. [World Model 指标](#9-world-model-指标)
10. [验证指标](#10-验证指标)
11. [指标监控最佳实践](#11-指标监控最佳实践)

---

## 1. Actor 指标 (策略网络)

### 1.1 `actor/pg_loss` - 策略梯度损失

**含义**：PPO 的策略梯度损失，衡量策略改进的幅度

**计算公式**：
```python
ratio = exp(log_prob - old_log_prob)
pg_losses = -advantages * ratio  # 或使用 clipped ratio
pg_loss = mean(pg_losses)
```

**数学解释**：
- 本质上是带权重的对数似然损失
- 权重 = advantages（优势函数）
- 目标：增加高优势动作的概率，降低低优势动作的概率

**正常范围**：
- 初期：-0.01 到 0.01
- 稳定后：-0.001 到 0.001
- 接近 0 时表示策略更新幅度小

**趋势分析**：

| 趋势 | 含义 | 原因 | 建议 |
|------|------|------|------|
| 负值且绝对值大 | 策略正在积极改进 | 发现了更好的策略 | 正常，继续训练 |
| 正值且较大 | 策略在变差 | clip 机制生效或优势估计不准 | 检查优势计算，降低学习率 |
| 接近 0 | 策略趋于稳定 | 收敛或陷入局部最优 | 检查奖励，考虑调整探索 |
| 剧烈波动 | 训练不稳定 | 学习率过高或批次大小过小 | 降低学习率，增大批次 |

**示例**：
```
step:0 - actor/pg_loss:-0.005  # 初期，策略在改进
step:50 - actor/pg_loss:-0.002  # 改进幅度变小
step:100 - actor/pg_loss:0.000  # 接近收敛
```

**调试技巧**：
```python
# 如果 pg_loss 异常，检查：
print(f"Advantages mean: {advantages.mean()}")  # 应该接近 0
print(f"Advantages std: {advantages.std()}")    # 应该为 1（经过归一化）
print(f"Log prob diff: {(log_prob - old_log_prob).mean()}")  # 应该很小
```

---

### 1.2 `actor/pg_clipfrac` - 策略梯度裁剪比例

**含义**：被 PPO clip 机制裁剪的样本比例

**计算公式**：
```python
ratio = exp(log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-clip_range, 1+clip_range)
pg_clipfrac = mean(clipped_ratio != ratio)
```

**直观理解**：
- 衡量有多少样本的策略更新被限制住了
- 防止策略更新过大，保持训练稳定性

**正常范围**：
- 理想值：0.05 - 0.30 (5% - 30%)
- < 0.05：策略更新保守，可能收敛慢
- \> 0.50：策略更新激进，可能不稳定

**趋势分析**：

| 值域 | 含义 | 影响 | 建议 |
|------|------|------|------|
| 0 - 0.05 | 几乎没有裁剪 | 策略更新空间大，可能学习慢 | 可以增大学习率或 `clip_ratio` |
| 0.05 - 0.30 | 适度裁剪 | 训练稳定且高效 | 继续保持 ✅ |
| 0.30 - 0.50 | 较多裁剪 | 策略更新受限，进展可能慢 | 考虑减小学习率或增大 `clip_ratio` |
| \> 0.50 | 大量裁剪 | 策略更新被严重限制 | 减小学习率或检查奖励信号 |

**示例**：
```
step:0 - actor/pg_clipfrac:0.000  # 初期无裁剪
step:10 - actor/pg_clipfrac:0.120  # 适度裁剪 ✅
step:50 - actor/pg_clipfrac:0.450  # 过多裁剪 ⚠️
```

**与 clip_ratio 的关系**：
```yaml
# 配置文件中
actor_rollout_ref.actor.clip_ratio: 0.2  # 默认值

# 如果 pg_clipfrac > 0.5，考虑增大到 0.3
actor_rollout_ref.actor.clip_ratio: 0.3
```

---

### 1.3 `actor/ppo_kl` - PPO KL 散度

**含义**：新旧策略之间的 KL 散度，衡量策略变化幅度

**计算公式**：
```python
ppo_kl = mean(old_log_prob - log_prob)
```

**数学背景**：
- KL 散度衡量两个分布的差异
- 这里是近似 KL（基于重要性采样）
- 真实 KL = E[log(π_old) - log(π_new)]

**正常范围**：
- 理想值：0.001 - 0.01
- < 0.001：策略几乎不变
- \> 0.05：策略变化过大

**趋势分析**：

| 值域 | 含义 | 影响 | 建议 |
|------|------|------|------|
| < 0.001 | 策略几乎不变 | 学习速度慢 | 增大学习率 |
| 0.001 - 0.01 | 适度变化 | 平衡探索与稳定性 ✅ | 保持 |
| 0.01 - 0.05 | 较大变化 | 可能影响稳定性 | 监控性能 |
| \> 0.05 | 剧烈变化 | 训练不稳定 | 降低学习率 |

**示例场景**：

**场景 1：健康训练**
```
step:0 - actor/ppo_kl:0.003 - actor/pg_clipfrac:0.15
step:10 - actor/ppo_kl:0.004 - actor/pg_clipfrac:0.20
step:20 - actor/ppo_kl:0.002 - actor/pg_clipfrac:0.10
# 策略稳步改进 ✅
```

**场景 2：学习率过高**
```
step:0 - actor/ppo_kl:0.050 - actor/pg_clipfrac:0.80
step:1 - actor/ppo_kl:0.045 - actor/pg_clipfrac:0.75
# KL 过大，大量裁剪，需要降低学习率 ⚠️
```

**场景 3：收敛**
```
step:80 - actor/ppo_kl:0.0001 - actor/pg_clipfrac:0.01
step:90 - actor/ppo_kl:0.0001 - actor/pg_clipfrac:0.01
# 策略稳定，可能已收敛 ✅
```

---

### 1.4 `actor/pg_clipfrac_lower` - 负优势裁剪比例

**含义**：Dual-clip PPO 中，负优势样本被下界裁剪的比例

**计算公式**：
```python
# 对于 advantage < 0 的样本
lower_clip = -advantages * clip_ratio_c  # 通常 clip_ratio_c=3.0
pg_clipfrac_lower = mean((advantages < 0) & (lower_clip > clipped_loss))
```

**背景**：
- 来自论文：https://arxiv.org/pdf/1912.09729
- 防止负优势样本（不好的动作）被过度惩罚
- 增加训练稳定性

**正常范围**：
- 通常 < 0.1
- 较少触发（因为 clip_ratio_c=3.0 比较大）

**示例**：
```
step:0 - actor/pg_clipfrac_lower:0.02  # 很少触发
```

---

### 1.5 `actor/entropy_loss` - 熵损失

**含义**：策略输出分布的熵，衡量策略的随机性

**计算公式**：
```python
entropy = -sum(p * log(p))  # 对所有 token
entropy_loss = mean(entropy)
```

**直观理解**：
- 高熵 = 分布平坦 = 输出随机 = 探索多
- 低熵 = 分布尖锐 = 输出确定 = 利用多

**正常范围**：
- 初期：0.5 - 2.0（取决于任务）
- 训练中：逐渐下降
- 最终：0.1 - 0.5

**趋势分析**：

| 阶段 | 典型值 | 含义 | 期望行为 |
|------|--------|------|----------|
| 初始化 | 5.0 - 8.0 | 模型几乎随机 | 快速下降 |
| 早期训练 | 0.5 - 2.0 | 开始学习但保持探索 | 缓慢下降 |
| 中期训练 | 0.3 - 0.8 | 平衡探索与利用 | 稳定或小幅下降 |
| 收敛阶段 | 0.1 - 0.3 | 策略较确定 | 稳定 |
| 过拟合 | < 0.05 | 几乎确定性输出 | ⚠️ 可能需要正则化 |

**示例场景**：

**场景 1：正常训练（对话生成）**
```
step:0 - actor/entropy_loss:0.800
step:50 - actor/entropy_loss:0.650
step:100 - actor/entropy_loss:0.500
step:200 - actor/entropy_loss:0.400
# 保持一定多样性 ✅
```

**场景 2：过早收敛（数学推理）**
```
step:0 - actor/entropy_loss:0.600
step:10 - actor/entropy_loss:0.050  # 过快下降
step:20 - actor/entropy_loss:0.030
# 可能陷入局部最优，考虑增大 entropy_coeff ⚠️
```

**场景 3：探索不足**
```
step:0 - actor/entropy_loss:0.500
step:100 - actor/entropy_loss:0.480  # 几乎不变
# 策略没有学到东西，检查奖励信号 ⚠️
```

**与 entropy_coeff 的关系**：
```yaml
# 如果希望增加探索
actor_rollout_ref.actor.entropy_coeff: 0.01  # 从 0.001 增大

# 如果希望确定性输出
actor_rollout_ref.actor.entropy_coeff: 0.0   # 关闭熵正则化
```

---

### 1.6 `actor/kl_loss` - KL 散度损失

**含义**：当前策略与参考策略的 KL 散度（用于 GRPO 等算法）

**计算公式**：
```python
kld = kl_penalty(log_prob, ref_log_prob, kl_penalty_type)
kl_loss = mean(kld)
```

**使用场景**：
- GRPO 算法中使用（替代 critic）
- 防止策略偏离参考策略太远
- 类似于信任域约束

**正常范围**：
- 0.001 - 0.1

**示例**：
```
# GRPO 训练
step:0 - actor/kl_loss:0.005 - actor/kl_coef:0.001
```

---

### 1.7 `actor/grad_norm` - 梯度范数

**含义**：参数梯度的 L2 范数，衡量梯度大小

**计算公式**：
```python
grad_norm = sqrt(sum(grad_i^2 for all parameters))
```

**正常范围**：
- 通常：0.1 - 10.0
- 经过梯度裁剪后：< `grad_clip` 值（通常为 1.0）

**趋势分析**：

| 值域 | 含义 | 影响 | 建议 |
|------|------|------|------|
| < 0.01 | 梯度消失 | 学习停滞 | 增大学习率，检查奖励 |
| 0.01 - 1.0 | 健康梯度 | 训练正常 ✅ | 保持 |
| 1.0 - 10.0 | 较大梯度 | 可能被裁剪 | 监控训练稳定性 |
| \> 10.0 | 梯度爆炸 | 训练不稳定 | 检查数据，降低学习率 |
| NaN/Inf | 数值错误 | 训练崩溃 | 检查学习率，数据预处理 |

**示例**：
```
step:0 - actor/grad_norm:1.523
step:10 - actor/grad_norm:0.845
step:20 - actor/grad_norm:1.000  # 触发梯度裁剪
```

**调试**：
```python
# 如果频繁出现 grad_norm = grad_clip
# 说明梯度经常被裁剪，考虑：
1. 降低学习率
2. 增大 grad_clip (小心)
3. 检查是否有异常数据
```

---

### 1.8 `actor/lr` - 学习率

**含义**：当前的学习率（可能带有预热和衰减）

**示例**：
```
step:0 - actor/lr:1e-07   # warmup 阶段
step:100 - actor/lr:1e-06  # 达到峰值
step:1000 - actor/lr:5e-07  # cosine 衰减
```

**典型配置**：
```yaml
actor_rollout_ref.actor.optim:
  lr: 1e-6  # 峰值学习率
  lr_warmup_steps_ratio: 0.1  # 10% 步数用于预热
  warmup_style: constant  # 或 cosine
```

---

## 2. Critic 指标 (价值网络)

### 2.1 `critic/vf_loss` - 价值函数损失

**含义**：价值网络预测的准确度，使用 clipped MSE loss

**计算公式**：
```python
vpred_clipped = clip(vpred, values - cliprange_value, values + cliprange_value)
vf_loss1 = (vpred - returns)^2
vf_loss2 = (vpred_clipped - returns)^2
vf_loss = mean(max(vf_loss1, vf_loss2))
```

**正常范围**：
- 初期：10.0 - 100.0（取决于奖励尺度）
- 训练后：0.1 - 10.0
- 收敛：< 1.0

**趋势分析**：

| 趋势 | 含义 | 影响 | 建议 |
|------|------|------|------|
| 快速下降 | Critic 在学习 | 优势估计改善 ✅ | 继续训练 |
| 稳定在低值 | Critic 准确 | Actor 可以信赖优势 ✅ | 正常 |
| 不下降 | Critic 学不到 | Actor 训练受影响 | 增大 critic 学习率 |
| 先降后升 | 奖励分布变化 | Actor 策略变化导致 | 正常现象 |
| 剧烈波动 | 训练不稳定 | 优势估计不可靠 | 降低学习率 |

**示例**：
```
step:0 - critic/vf_loss:14.947
step:10 - critic/vf_loss:8.352
step:50 - critic/vf_loss:2.148
step:100 - critic/vf_loss:0.832  # 收敛良好 ✅
```

---

### 2.2 `critic/vf_clipfrac` - 价值函数裁剪比例

**含义**：被 clip 机制裁剪的价值预测比例

**计算公式**：
```python
clipped = (vpred_clipped != vpred)
vf_clipfrac = mean(clipped)
```

**正常范围**：
- 0.0 - 0.5
- 通常 < 0.2

**含义**：
- 过高（>0.5）：价值函数更新受限
- 过低（<0.05）：裁剪机制未起作用

**示例**：
```
step:0 - critic/vf_clipfrac:0.000
step:50 - critic/vf_clipfrac:0.384  # 较多裁剪
```

---

### 2.3 `critic/vpred_mean` - 价值预测均值

**含义**：Critic 预测的平均价值

**作用**：
- 监控价值尺度是否合理
- 应该与实际 returns 接近

**示例**：
```
step:0 - critic/vpred_mean:-2.056 - critic/returns/mean:0.003
# 初期预测不准

step:100 - critic/vpred_mean:2.340 - critic/returns/mean:2.450
# 预测接近真实值 ✅
```

---

### 2.4 `critic/vf_explained_var` - 解释方差

**含义**：价值函数解释 returns 方差的比例，衡量拟合质量

**计算公式**：
```python
vf_explained_var = 1 - Var(returns - values) / Var(returns)
```

**数学解释**：
- 类似于 R² 分数
- 1.0 = 完美拟合
- 0.0 = 预测不比常数好
- <0 = 预测比常数还差

**正常范围**：
- 初期：-1.0 到 0.2
- 训练中：0.2 到 0.8
- 良好收敛：> 0.7

**趋势分析**：

| 值域 | 质量 | 含义 | 建议 |
|------|------|------|------|
| < 0 | 极差 | Critic 没有学到 | 增大学习率，检查数据 |
| 0 - 0.3 | 差 | Critic 刚开始学 | 继续训练 |
| 0.3 - 0.7 | 中等 | Critic 基本可用 | 继续优化 |
| 0.7 - 0.9 | 良好 | Critic 准确 ✅ | 很好 |
| > 0.9 | 优秀 | Critic 非常准确 ✅ | 完美 |

**示例**：
```
step:0 - critic/vf_explained_var:-0.234  # 初期很差
step:50 - critic/vf_explained_var:0.512  # 开始学到
step:100 - critic/vf_explained_var:0.823  # 拟合良好 ✅
```

---

### 2.5 `critic/grad_norm` - Critic 梯度范数

**含义**：Critic 参数梯度的 L2 范数

**正常范围**：
- 0.1 - 100.0（通常比 Actor 大）
- 经过裁剪后：< `grad_clip`（通常为 1.0）

**示例**：
```
step:0 - critic/grad_norm:1023.278  # 初期很大
step:10 - critic/grad_norm:942.924
step:50 - critic/grad_norm:1.000  # 经常触发裁剪
```

---

### 2.6 `critic/lr` - Critic 学习率

**含义**：Critic 的学习率

**典型配置**：
```yaml
critic.optim:
  lr: 1e-5  # 通常比 Actor 大一个数量级
```

---

## 3. 奖励与优势指标

### 3.1 `critic/score/mean|max|min` - 原始分数统计

**含义**：每个序列的原始分数（token_level_scores 的总和）

**来源**：
```python
sequence_score = token_level_scores.sum(dim=-1)
```

**含义**：
- 奖励函数给出的原始分数
- 包含任务奖励 + KL 惩罚（如果启用）

**正常范围**：
- 取决于任务和奖励函数
- 数学任务：0.0 (错误) 或 1.0 (正确)
- 对话任务：连续值，如 -5.0 到 5.0

**趋势分析**：

| 趋势 | 含义 | 期望 |
|------|------|------|
| mean 上升 | 策略在改进 ✅ | 训练目标 |
| max 上升 | 最好样本改进 | 好的探索 |
| min 上升 | 最差样本改进 | 更稳健 |
| mean 稳定 | 收敛或瓶颈 | 检查是否最优 |

**示例**：

**场景 1：数学任务（GSM8K）**
```
step:0 - critic/score/mean:0.145 - max:1.0 - min:0.0
# 14.5% 正确率

step:50 - critic/score/mean:0.523 - max:1.0 - min:0.0
# 52.3% 正确率 ✅ 进步明显

step:100 - critic/score/mean:0.782 - max:1.0 - min:0.0
# 78.2% 正确率 ✅ 持续改进
```

**场景 2：对话任务**
```
step:0 - critic/score/mean:0.234 - max:4.5 - min:-3.2
step:50 - critic/score/mean:1.823 - max:4.8 - min:-1.5
# 平均分数提高，最差样本也改进 ✅
```

---

### 3.2 `critic/rewards/mean|max|min` - 奖励统计

**含义**：实际用于训练的奖励（score - KL penalty）

**计算公式**：
```python
rewards = scores - kl_coef * kl_divergence
```

**与 score 的关系**：
```python
# 如果不使用 KL 惩罚
rewards == scores

# 如果使用 KL 惩罚（在 reward 中）
rewards = scores - kl_penalty
```

**示例**：
```
step:0 - critic/score/mean:0.500 - critic/rewards/mean:0.485
# KL 惩罚：0.500 - 0.485 = 0.015
```

---

### 3.3 `critic/advantages/mean|max|min` - 优势函数统计

**含义**：优势函数，衡量动作相对基线的好坏

**计算公式（GAE）**：
```python
delta = rewards + gamma * V(s_{t+1}) - V(s_t)
advantages = GAE(delta, gamma, lambda)
advantages = (advantages - mean) / (std + eps)  # 归一化
```

**数学直觉**：
- 优势 > 0：动作比平均好，应该增加概率
- 优势 < 0：动作比平均差，应该降低概率
- 优势 ≈ 0：动作接近平均水平

**正常特性**：
- **mean ≈ 0**：归一化后的期望特性
- **std ≈ 1**：归一化后的特性
- max 和 min 表示优势范围

**趋势分析**：

| 指标 | 正常值 | 异常情况 | 可能原因 |
|------|--------|----------|----------|
| mean | -0.1 到 0.1 | \|mean\| > 0.5 | 归一化失败 |
| max | 1.0 到 5.0 | > 10.0 | 奖励方差过大 |
| min | -5.0 到 -1.0 | < -10.0 | 奖励方差过大 |
| max-min | 2.0 到 10.0 | > 20.0 | 策略差异大 |

**示例**：

**正常情况**
```
step:0 - critic/advantages/mean:-0.023 - max:2.360 - min:-2.280
# mean 接近 0，范围合理 ✅
```

**异常情况**
```
step:0 - critic/advantages/mean:5.234 - max:50.23 - min:-45.12
# mean 远离 0，范围过大 ⚠️
# 检查：奖励尺度、归一化、critic 训练
```

---

### 3.4 `critic/returns/mean|max|min` - 回报统计

**含义**：累积折扣奖励（advantages + values）

**计算公式**：
```python
returns = advantages + values
# 或直接计算
returns = sum(gamma^t * reward_t)
```

**作用**：
- Critic 的训练目标
- 应该与实际累积奖励接近

**与其他指标关系**：
```python
returns = advantages + values
# 理想情况
returns ≈ actual_cumulative_rewards
```

**示例**：
```
step:0 - critic/returns/mean:0.003 - critic/values/mean:-2.045
# advantages = 0.003 - (-2.045) = 2.048
```

---

### 3.5 `critic/values/mean|max|min` - 价值估计统计

**含义**：Critic 预测的状态价值

**作用**：
- 作为优势计算的基线
- 应该接近实际 returns

**趋势**：
- 训练过程中应该逐渐接近 returns/mean

**示例**：
```
step:0 - critic/values/mean:-2.045 - critic/returns/mean:0.003
# 初期相差大

step:100 - critic/values/mean:2.340 - critic/returns/mean:2.450
# 逐渐接近 ✅
```

---

### 3.6 `actor/reward_kl_penalty` - KL 惩罚

**含义**：策略与参考策略的 KL 散度

**计算公式**：
```python
kl = mean(old_log_prob - ref_log_prob)
```

**配置**：
```yaml
algorithm:
  use_kl_in_reward: True  # 启用
  kl_ctrl:
    type: fixed  # 或 adaptive
    kl_coef: 0.001
```

**示例**：
```
step:0 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty_coeff:0.001
```

---

## 4. 序列长度指标

### 4.1 `response_length/mean|max|min` - 生成长度统计

**含义**：生成的 response 的实际长度（不含 padding）

**正常范围**：
- 取决于 `max_response_length` 配置
- 理想：< max_response_length（未达到上限）

**趋势分析**：

| 情况 | 含义 | 影响 | 建议 |
|------|------|------|------|
| mean ≈ max | 经常截断 | 可能影响质量 | 增大 max_response_length |
| mean << max | 生成较短 | 效率高 | 正常 ✅ |
| mean 增长 | 生成变长 | 监控质量 | 正常演化 |
| max = max_response_length | 有截断 | 部分样本未完成 | 考虑增大上限 |

**示例**：

**正常情况**
```
# max_response_length = 512
step:0 - response_length/mean:239.1 - max:256.0 - min:77.0
# 平均 239，未达到上限 ✅
```

**截断过多**
```
# max_response_length = 256
step:0 - response_length/mean:253.8 - max:256.0 - clip_ratio:0.85
# 85% 样本达到上限，可能需要增大到 512 ⚠️
```

---

### 4.2 `response_length/clip_ratio` - 响应截断比例

**含义**：达到最大长度限制的样本比例

**计算公式**：
```python
clip_ratio = mean(response_length == max_response_length)
```

**理想值**：
- < 0.1 (10%)：大部分样本能自然结束
- 0.1 - 0.3：可以接受
- \> 0.5：需要增大 max_response_length

**示例**：
```
step:0 - response_length/clip_ratio:0.012  # 1.2% 截断 ✅
step:50 - response_length/clip_ratio:0.523  # 52.3% 截断 ⚠️
```

---

### 4.3 `prompt_length/mean|max|min` - 提示长度统计

**含义**：输入 prompt 的实际长度

**作用**：
- 监控输入数据分布
- 计算 token 吞吐量

**示例**：
```
step:0 - prompt_length/mean:123.5 - max:512.0 - min:45.0
```

---

### 4.4 `prompt_length/clip_ratio` - 提示截断比例

**含义**：达到最大提示长度的样本比例

**理想值**：
- < 0.05：数据准备合理
- \> 0.1：考虑增大 max_prompt_length

---

## 5. Episode 指标

### 5.1 `episode/reward/mean|max|min` - Episode 奖励统计

**含义**：完整 episode 的累积奖励

**应用场景**：
- 多轮对话任务
- 环境交互任务（AlfWorld, WebShop等）

**与 critic/score 的区别**：
- `critic/score`：单步生成的奖励
- `episode/reward`：整个 episode 的累积奖励

**示例**：
```
# AlfWorld 任务
episode/reward/mean: 0.65
episode/reward/max: 1.0  # 完成任务
episode/reward/min: 0.0  # 失败
```

---

### 5.2 `episode/length/mean|max|min` - Episode 长度统计

**含义**：Episode 包含的交互轮数

**示例**：
```
episode/length/mean: 15.3  # 平均 15 轮交互
episode/length/max: 50.0   # 达到上限
episode/length/min: 3.0    # 快速完成
```

---

### 5.3 `episode/success_rate_*` - 任务成功率

**含义**：特定任务的成功率指标

**示例**：
```
episode/success_rate_alfworld: 0.68  # 68% 成功率
```

---

## 6. 时间性能指标

### 6.1 `timing_s/*` - 各阶段耗时（秒）

**包含的阶段**：
- `timing_s/gen`：生成阶段（rollout）
- `timing_s/ref`：参考策略计算
- `timing_s/values`：价值函数计算
- `timing_s/adv`：优势函数计算
- `timing_s/update_critic`：Critic 更新
- `timing_s/update_actor`：Actor 更新
- `timing_s/step`：总步骤时间

**示例**：
```
step:0 - timing_s/gen:21.470 - timing_s/ref:4.360 - timing_s/values:5.800
       - timing_s/adv:0.109 - timing_s/update_critic:15.664
       - timing_s/update_actor:20.314
```

**瓶颈分析**：

| 阶段 | 占比 | 典型值 | 优化方向 |
|------|------|--------|----------|
| gen | 30-40% | 20-30s | 增大 batch size，优化采样 |
| ref | 5-10% | 3-5s | 使用 offload |
| values | 5-10% | 3-5s | - |
| update_actor | 25-35% | 15-25s | 优化 micro batch size |
| update_critic | 20-30% | 10-20s | 优化 micro batch size |

---

### 6.2 `timing_per_token_ms/*` - 每 Token 耗时（毫秒）

**含义**：归一化到每个 token 的处理时间

**计算公式**：
```python
timing_per_token_ms/gen = timing_s/gen * 1000 / num_response_tokens
timing_per_token_ms/ref = timing_s/ref * 1000 / num_total_tokens
```

**作用**：
- 更公平的性能对比（消除 batch size 影响）
- 识别效率瓶颈

**示例**：
```
timing_per_token_ms/gen: 41.67   # 每个生成 token 42ms
timing_per_token_ms/ref: 25.00   # 每个 token 25ms
timing_per_token_ms/values: 25.00
```

**性能基准**：
- 生成（Gen）：20-50 ms/token
- 前向（Ref/Values）：10-30 ms/token
- 训练（Update）：30-60 ms/token

---

## 7. 系统性能指标

### 7.1 `perf/total_num_tokens` - 总 Token 数

**含义**：一个 step 处理的总 token 数（prompt + response）

**计算公式**：
```python
total_num_tokens = sum(attention_mask)  # 所有有效 token
```

**示例**：
```
perf/total_num_tokens: 98304  # batch_size * seq_length
```

---

### 7.2 `perf/time_per_step` - 每步耗时

**含义**：完整训练步的总时间

**等价于**：`timing_s/step`

**示例**：
```
perf/time_per_step: 67.8  # 秒
```

---

### 7.3 `perf/throughput` - 吞吐量

**含义**：每秒每 GPU 处理的 token 数

**计算公式**：
```python
throughput = total_num_tokens / (time_per_step * n_gpus)
```

**示例**：
```
perf/total_num_tokens: 98304
perf/time_per_step: 67.8
n_gpus: 8
perf/throughput: 145.0  # tokens/s/GPU
```

**性能基准**：

| 模型大小 | 目标吞吐量 (tokens/s/GPU) |
|---------|--------------------------|
| 7B      | 100 - 300 |
| 13B     | 50 - 150 |
| 30B     | 20 - 60 |
| 70B     | 10 - 30 |

---

### 7.4 `perf/mfu/*` - 模型 FLOPs 利用率

**含义**：实际 FLOPs 占理论峰值的比例

**计算公式**：
```python
estimated_flops = model_flops(num_tokens, model_params)
promised_flops = hardware_peak_flops(gpu_type)
mfu = estimated_flops / (promised_flops * n_gpus)
```

**指标**：
- `perf/mfu/actor`：Actor 训练的 MFU
- `perf/mfu/critic`：Critic 训练的 MFU

**目标值**：
- 优秀：> 40%
- 良好：30% - 40%
- 一般：20% - 30%
- 较差：< 20%

**示例**：
```
perf/mfu/actor: 0.35  # 35% MFU ✅
perf/mfu/critic: 0.28  # 28% MFU
```

---

### 7.5 内存指标

**指标**：
- `perf/max_memory_allocated_gb`：GPU 分配的最大内存
- `perf/max_memory_reserved_gb`：GPU 保留的最大内存
- `perf/cpu_memory_used_gb`：CPU 内存使用

**示例**：
```
perf/max_memory_allocated_gb: 45.2
perf/max_memory_reserved_gb: 48.0
perf/cpu_memory_used_gb: 128.5
```

---

## 8. 训练控制指标

### 8.1 `training/global_step` - 全局步数

**含义**：当前训练步数（从 0 开始）

**用途**：
- X 轴坐标
- 学习率调度
- 保存检查点

**示例**：
```
training/global_step: 123
```

---

### 8.2 `training/epoch` - 训练轮次

**含义**：当前训练 epoch

**示例**：
```
training/epoch: 5
```

---

## 9. World Model 指标

### 9.1 `world_model/loss` - World Model 损失

**含义**：World Model 的预测损失（通常是交叉熵）

**应用场景**：
- 模型推理树搜索
- 状态预测

**示例**：
```
world_model/loss: 2.345
world_model/grad_norm: 0.823
```

---

## 10. 验证指标

### 10.1 验证集指标

**格式**：`val/{metric_name}`

**常见指标**：
- `val/score`：验证集平均分数
- `val/accuracy`：任务准确率
- `val/pass@k`：编程任务的 pass@k

**示例**：
```
val/score: 0.856
val/accuracy: 85.6%
```

---

## 11. 指标监控最佳实践

### 11.1 核心指标组合

**训练健康度**：
```python
# 策略改进
actor/pg_loss: 下降
critic/score/mean: 上升
episode/reward/mean: 上升  # 如果有

# 训练稳定性
actor/pg_clipfrac: 0.05 - 0.30
actor/ppo_kl: 0.001 - 0.01
actor/grad_norm: < grad_clip

# Critic 质量
critic/vf_explained_var: > 0.7
critic/vf_loss: 下降
```

**性能监控**：
```python
perf/throughput: 越高越好
perf/mfu/actor: > 0.3
timing_s/step: 越低越好
```

---

### 11.2 异常诊断流程

#### 问题 1：训练不收敛

**症状**：
```
actor/pg_loss: 波动大
critic/score/mean: 不上升
```

**检查清单**：
1. 学习率是否过高？→ 查看 `actor/grad_norm`
2. 奖励信号是否合理？→ 查看 `critic/rewards/mean`
3. Critic 是否学习？→ 查看 `critic/vf_explained_var`
4. 批次大小是否太小？→ 查看 batch_size 配置

**诊断步骤**：
```python
# Step 1: 检查梯度
if actor/grad_norm == grad_clip:
    print("梯度被裁剪，考虑降低学习率")

# Step 2: 检查奖励
if critic/rewards/std > 10 * critic/rewards/mean:
    print("奖励方差过大，考虑归一化")

# Step 3: 检查 Critic
if critic/vf_explained_var < 0.3:
    print("Critic 学习不足，增大 critic 学习率")
```

---

#### 问题 2：策略过早收敛

**症状**：
```
actor/entropy_loss: 快速下降到接近 0
critic/score/mean: 停止上升
actor/ppo_kl: 接近 0
```

**原因**：
- 陷入局部最优
- 探索不足

**解决方案**：
```yaml
# 增加熵正则化
actor_rollout_ref.actor.entropy_coeff: 0.01  # 从 0.001 增大

# 或增加温度
actor_rollout_ref.rollout.temperature: 1.2  # 从 1.0 增大
```

---

#### 问题 3：内存溢出

**症状**：
```
CUDA out of memory
```

**检查**：
```
perf/max_memory_allocated_gb: 接近 GPU 总内存
```

**解决方案**：
```yaml
# 减小 batch size
actor_rollout_ref.actor.ppo_mini_batch_size: 128  # 从 256 减半

# 或启用梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing: True

# 或使用动态批次大小
actor_rollout_ref.actor.use_dynamic_bsz: True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 8192  # 从 16384 减半
```

---

#### 问题 4：训练速度慢

**症状**：
```
perf/throughput: < 50 tokens/s/GPU (7B 模型)
perf/time_per_step: > 120s
```

**诊断**：
```python
# 找到瓶颈
if timing_s/gen > 0.5 * timing_s/step:
    print("生成是瓶颈")
    # 优化：增大 vllm max_num_seqs，调整 tensor_parallel_size

if timing_s/update_actor > 0.4 * timing_s/step:
    print("Actor 更新是瓶颈")
    # 优化：调整 micro_batch_size，启用 torch.compile
```

---

### 11.3 Wandb/TensorBoard 可视化

**推荐面板**：

**面板 1：训练进度**
```python
# Y 轴
- critic/score/mean
- episode/reward/mean
# X 轴
- training/global_step
```

**面板 2：策略更新**
```python
- actor/pg_loss
- actor/pg_clipfrac
- actor/ppo_kl
- actor/entropy_loss
```

**面板 3：Critic 质量**
```python
- critic/vf_loss
- critic/vf_explained_var
- critic/vpred_mean vs critic/returns/mean
```

**面板 4：性能**
```python
- perf/throughput
- perf/mfu/actor
- timing_s/step
```

---

### 11.4 日志示例解读

#### 示例 1：健康训练

```
step:0 - timing/gen:21.470 - timing/ref:4.360 - timing/values:5.800 
       - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty_coeff:0.001 
       - timing/adv:0.109 - timing/update_critic:15.664 
       - critic/vf_loss:14.947 - critic/vf_clipfrac:0.000 
       - critic/vpred_mean:-2.056 - critic/grad_norm:1023.278 
       - critic/lr(1e-4):0.100 
       - timing/update_actor:20.314 
       - actor/entropy_loss:0.433 - actor/pg_loss:-0.005 
       - actor/pg_clipfrac:0.000 - actor/ppo_kl:0.000 
       - actor/grad_norm:1.992 - actor/lr(1e-6):0.010 
       - critic/score/mean:0.004 - critic/score/max:1.000 - critic/score/min:0.000 
       - response_length/mean:239.133 - response_length/max:256.000
```

**解读**：
- ✅ `actor/pg_loss:-0.005`：策略在改进
- ✅ `actor/pg_clipfrac:0.000`：初期无裁剪
- ✅ `critic/score/max:1.000`：有成功样本
- ⚠️ `critic/vf_loss:14.947`：Critic 初期损失高，正常
- ✅ 各项指标在合理范围

---

#### 示例 2：学习率过高

```
step:5 - actor/pg_loss:0.523 - actor/pg_clipfrac:0.850 
       - actor/ppo_kl:0.125 - actor/grad_norm:1.000
       - critic/score/mean:0.003
```

**解读**：
- ⚠️ `actor/pg_clipfrac:0.850`：85% 被裁剪，过高！
- ⚠️ `actor/ppo_kl:0.125`：KL 过大（>0.05）
- ⚠️ `actor/grad_norm:1.000`：频繁触发梯度裁剪
- ❌ `critic/score/mean`：不上升

**建议**：
```yaml
# 降低学习率
actor_rollout_ref.actor.optim.lr: 5e-7  # 从 1e-6 减半
```

---

#### 示例 3：Critic 学习不足

```
step:50 - critic/vf_loss:45.23 - critic/vf_explained_var:-0.234
        - critic/vpred_mean:0.123 - critic/returns/mean:5.678
        - actor/pg_loss:-0.002 - critic/score/mean:0.256
```

**解读**：
- ❌ `critic/vf_explained_var:-0.234`：负值！
- ❌ `critic/vf_loss:45.23`：50 步后仍然很高
- ❌ `vpred_mean` vs `returns/mean`：相差太大

**建议**：
```yaml
# 增大 Critic 学习率
critic.optim.lr: 5e-5  # 从 1e-5 增大 5 倍

# 或增加训练 epochs
critic.ppo_epochs: 2  # 从 1 增加
```

---

## 12. 快速参考表

### 核心指标目标值

| 指标 | 目标范围 | 理想趋势 |
|------|---------|---------|
| `actor/pg_loss` | -0.01 ~ 0.01 | → 0 |
| `actor/pg_clipfrac` | 0.05 ~ 0.30 | 稳定 |
| `actor/ppo_kl` | 0.001 ~ 0.01 | 稳定或↓ |
| `actor/entropy_loss` | 0.1 ~ 0.5 | 缓慢↓ |
| `critic/vf_loss` | < 1.0 | ↓ |
| `critic/vf_explained_var` | > 0.7 | ↑ |
| `critic/score/mean` | 任务相关 | ↑ |
| `perf/throughput` | 见模型基准 | 越高越好 |

---

### 异常模式速查

| 异常模式 | 可能原因 | 检查指标 |
|---------|---------|---------|
| 不收敛 | 学习率/奖励/Critic | `grad_norm`, `vf_explained_var` |
| 过早收敛 | 探索不足 | `entropy_loss`, `ppo_kl` |
| 训练崩溃 | 梯度爆炸 | `grad_norm`, `pg_loss` |
| 性能差 | 瓶颈 | `timing_s/*`, `throughput` |
| 内存溢出 | batch 过大 | `max_memory_*` |

---

## 附录：指标计算源码位置

| 指标类别 | 源文件 |
|---------|--------|
| Actor 指标 | `verl/workers/actor/dp_actor.py` |
| Critic 指标 | `verl/workers/critic/dp_critic.py` |
| 数据指标 | `verl/trainer/ppo/metric_utils.py` |
| 时间指标 | `verl/trainer/ppo/metric_utils.py` |
| 核心算法 | `verl/trainer/ppo/core_algos.py` |

---

**文档版本**：v1.0  
**最后更新**：2025-01-XX  
**适用于**：verl 0.3.x+


