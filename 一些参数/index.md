# 一些参数

# max_num_batched_tokens、max_num_seqs 以及 gpu_memory_utilization 的协作机制

这三个参数在 vLLM 中共同控制着请求调度和内存分配。让我详细解释它们之间的关系。

## 参数定义

### 1. max_num_batched_tokens
这是**单次迭代中可以处理的最大 token 数量**。它控制着每个批次的计算预算。 [1](#0-0) 

### 2. max_num_seqs
这是**单次迭代中可以并发处理的最大请求（序列）数量**。 [2](#0-1) 

### 3. gpu_memory_utilization
这是**GPU 内存的使用比例**（默认 0.9，即 90%），用于确定分配给 KV cache 的内存大小。 [3](#0-2) 

## 它们如何协作

### 调度器的约束机制

调度器在调度请求时会同时检查这两个约束： [4](#0-3) 

**关键验证逻辑**：
- `max_num_batched_tokens` 必须 >= `max_num_seqs`
- `max_num_batched_tokens` 不能超过 `max_num_seqs * max_model_len` [5](#0-4) 

### KV Cache 内存分配流程

`gpu_memory_utilization` 通过以下流程影响 KV cache：

1. **计算请求内存**： [6](#0-5) 

2. **内存分析**：执行一次前向传播来测量激活内存使用 [7](#0-6) 

3. **计算可用 KV cache 内存**：
   ```
   可用 KV cache 内存 = 请求内存 - 模型权重 - 峰值激活内存 - 非 torch 内存 - CUDA graph 内存
   ```

4. **计算 block 数量**： [8](#0-7) 

## 实际例子说明

### 例子 1：基本配置场景

假设配置：
- `max_num_batched_tokens = 2048`
- `max_num_seqs = 128`
- `gpu_memory_utilization = 0.9`
- `max_model_len = 4096`

**调度行为**：
- 调度器最多可以同时处理 128 个请求
- 但单次迭代的总 token 数不能超过 2048
- 例如：可以调度 128 个请求，每个生成 16 个 token（128 × 16 = 2048）
- 或者：调度 64 个请求，每个处理 32 个 token

**KV Cache 分配**：
- 假设 GPU 有 80GB 内存，`gpu_memory_utilization=0.9` 意味着可用 72GB
- 扣除模型权重（假设 40GB）和激活内存（假设 8GB）后，剩余 24GB 用于 KV cache
- 这 24GB 需要支持最多 128 个并发请求的 KV cache 存储

### 例子 2：Chunked Prefill 场景 [9](#0-8) 

配置：
- `max_num_batched_tokens = 8192`
- `max_num_seqs = 256`
- `gpu_memory_utilization = 0.9`

**场景**：一个长提示词请求（1000 tokens）+ 多个解码请求

调度器会：
1. 优先调度所有解码请求（假设 200 个请求，各 1 token = 200 tokens）
2. 剩余预算：8192 - 200 = 7992 tokens
3. 可以调度长提示词的部分（如 7992 tokens），或完整调度多个较短的新请求

### 例子 3：内存不足导致的抢占 [10](#0-9) 

当 KV cache 空间不足时：
- 调度器会抢占最低优先级的请求
- 释放其 KV cache blocks
- 被抢占的请求重新加入等待队列
- `num_computed_tokens` 重置为 0，需要重新计算

## 优化建议

### 提高吞吐量
- **增加** `max_num_batched_tokens`（如 16384）：允许更大的批次
- **增加** `gpu_memory_utilization`（如 0.95）：提供更多 KV cache 空间
- **增加** `max_num_seqs`（如 512）：支持更多并发请求

### 改善延迟
- **减小** `max_num_batched_tokens`（如 2048）：减少 prefill 对 decode 的影响
- **减小** `max_num_seqs`：减少竞争，降低抢占频率

### 避免 OOM
- **降低** `gpu_memory_utilization`：为系统预留更多内存
- **减小** `max_num_seqs`：减少 KV cache 需求
- **使用** tensor parallelism：在多个 GPU 间分片模型

## Notes

这三个参数形成了一个相互制约的系统：
- `max_num_batched_tokens` 和 `max_num_seqs` 是**调度约束**，限制每次迭代能处理多少工作
- `gpu_memory_utilization` 是**内存约束**，通过 KV cache 分配间接限制能同时支持多少请求
- 如果 KV cache 内存不足以支持 `max_num_seqs` 个满长度序列，会发生频繁的抢占，严重影响性能

合理配置需要根据：
1. 模型大小和 GPU 内存容量
2. 预期的请求模式（长提示词 vs 短对话）
3. 性能目标（吞吐量 vs 延迟）

进行平衡调整。

### Citations

**File:** vllm/config/scheduler.py (L50-55)
```python
    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """Maximum number of tokens to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """
```

**File:** vllm/config/scheduler.py (L57-62)
```python
    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """
```

**File:** vllm/config/scheduler.py (L265-270)
```python
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )
```

**File:** vllm/config/scheduler.py (L272-278)
```python
        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )
```

**File:** vllm/config/cache.py (L49-56)
```python
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
```

**File:** vllm/v1/worker/utils.py (L253-272)
```python
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> float:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = init_snapshot.total_memory * cache_config.gpu_memory_utilization

    if init_snapshot.free_memory < requested_memory:
        GiB = lambda b: round(b / GiB_bytes, 2)
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({GiB(init_snapshot.free_memory)}/"
            f"{GiB(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{GiB(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory
```

**File:** vllm/v1/worker/gpu_worker.py (L343-345)
```python
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )
```

**File:** vllm/v1/core/kv_cache_utils.py (L829-844)
```python
def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks
```

**File:** docs/configuration/optimization.md (L15-24)
```markdown
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

While this mechanism ensures system robustness, preemption and recomputation can adversely affect end-to-end latency.
If you frequently encounter preemptions, consider the following actions:

- Increase `gpu_memory_utilization`. vLLM pre-allocates GPU cache using this percentage of memory. By increasing utilization, you can provide more KV cache space.
- Decrease `max_num_seqs` or `max_num_batched_tokens`. This reduces the number of concurrent requests in a batch, thereby requiring less KV cache space.
- Increase `tensor_parallel_size`. This shards model weights across GPUs, allowing each GPU to have more memory available for KV cache. However, increasing this value may cause excessive synchronization overhead.
- Increase `pipeline_parallel_size`. This distributes model layers across GPUs, reducing the memory needed for model weights on each GPU, indirectly leaving more memory available for KV cache. However, increasing this value may cause latency penalties.
```

**File:** docs/configuration/optimization.md (L43-48)
```markdown
You can tune the performance by adjusting `max_num_batched_tokens`:

- Smaller values (e.g., 2048) achieve better inter-token latency (ITL) because there are fewer prefills slowing down decodes.
- Higher values achieve better time to first token (TTFT) as you can process more prefill tokens in a batch.
- For optimal throughput, we recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs.
- If `max_num_batched_tokens` is the same as `max_model_len`, that's almost the equivalent to the V0 default scheduling policy (except that it still prioritizes decodes).
```

