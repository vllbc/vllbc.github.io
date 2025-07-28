# ulysses_sequence_parallel

一句话：在sequence维度上进行切分





- 将输入序列 X (长度 N) 沿序列维度切分为 SP 块，每个 GPU 分配到 N/SP 长度的子序列。
    - 对于非注意力层 (如 MLP)，计算是完全局部的，每个 GPU 处理自己的子序列即可。
        - token 之间独立，token-level projection
        - Ulysses SP的核心复杂性在于Attention层。为了让每个token在计算注意力时能够考虑到全局序列信息（或者说，让每个head在计算时能看到完整的序列，即使这个head只在当前rank计算），Attention模块前后需要进行两次精密的all-to-all数据重排。MLP层则没有这样的需求，数据在进入MLP时已经是按序列分片好的，可以直接进行本地计算。
    - 对于注意力层:
        - 步骤 1 (计算 Q, K, V): 每个 GPU 基于其本地子序列计算出本地的 Q_local, K_local, V_local (维度约为 N/SP x d，d 是隐藏维度)。
        - 步骤 2 (全局 K, V 收集 - 关键): 使用 **All-to-All** 通信操作（All-Gather??）。每个 GPU 将自己的 K_local, V_local 发送给所有其他 GPU，并接收来自所有其他 GPU 的 K, V 块。执行后，**每个 GPU 拥有完整的全局 K 和 V 矩阵 (维度 N x d)**，但仍然只拥有本地的 Q_local (维度 N/SP x d)。
            - [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
        - 步骤 3 (本地注意力计算): 每个 GPU 使用其 Q_local 和完整的全局 K, V 计算其负责的那部分注意力输出 O_local (维度 N/SP x d)。计算公式为 Attention(Q_local, K_global, V_global)。这一步的计算量是 (N/SP) * N * d，内存瓶颈在于存储临时的注意力分数矩阵，大小约为 **(N/SP) * N**。相比原始的 **N*N**，内存显著降低。
        - 步骤 4 (可选的输出重组): 如果后续层需要按序列拼接的完整输出，可能需要另一次通信（如 All-Gather 或另一次 All-to-All 的变种）来组合 O_local。但在 DeepSpeed 实现中，通常保持分布式状态，直接输入到下一个同样按序列并行的层。



## verl中的序列并行

在verl中，一般与remove_padding一起使用，即
```python

if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
            ):
                assert config.actor_rollout_ref.model.use_remove_padding, (
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )
```
 
 1. 先进行remove padding操作
2. 然后进行序列并行的pad和slice操作
## 参考

[pytorch\_distribute\_tutorials/tutorials/3D-parallel/SP-序列并行.ipynb at main · chunhuizhang/pytorch\_distribute\_tutorials · GitHub](https://github.com/chunhuizhang/pytorch_distribute_tutorials/blob/main/tutorials/3D-parallel/SP-%E5%BA%8F%E5%88%97%E5%B9%B6%E8%A1%8C.ipynb)
