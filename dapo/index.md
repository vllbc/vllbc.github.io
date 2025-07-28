# dapo

DAPO 是对 GRPO 的改进。DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，即解耦裁剪和动态采样策略优化）的优化点有四个（其中前 2 个是主要亮点，是命名的来源）

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717203821.png)

## 更高裁剪
clip的上边界可以放宽一些，即 clip_high 从 0.2 提高到了 0.28

## 动态采样（Dynamic Sampling）

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717203911.png)

## Token 级策略梯度损失（Token-Level Policy Gradient Loss）

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717203932.png)


## 超长奖励塑造（Overlong Reward Shaping）

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250717203956.png)

## 参考

[# DAPO全是已有的小trick，为什么这么火?](https://www.zhihu.com/question/1895273986537014226/answer/1899582779408245950)
