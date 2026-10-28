# nano-vllm 全代码精读笔记:用 1200 行 Python 理解 vLLM 推理引擎


# nano-vllm 全代码精读笔记

> 配套参考:博客 [从零构建 nano-vllm:深入理解 vLLM 推理引擎](https://zhuanlan.zhihu.com/p/1925484783229698084)(原博客正文需自行访问,本文以源码为主)
>
> 适用对象:想**从零理解 vLLM 底层结构**的初学者。建议先读完本文,再去看 vLLM 主仓库。

`nano-vllm` 是一个**约 1200 行**的精简 vLLM 实现,覆盖了 vLLM 最核心的工程思想:连续批处理、PagedAttention、Prefix Caching、FlashAttention、Tensor Parallelism、CUDA Graph。本文**逐文件、逐函数**地把代码讲透,目标是让一个不熟悉 vLLM 的人读完本文后,能直接去看 vLLM 主仓库并快速定位到对应模块。

## 0. 怎么读本文

| 你想了解的内容 | 跳转章节 |
|---|---|
| 整体架构、模块关系 | §1 §2 |
| 一个请求从进入到返回的完整流程 | §3 |
| Sequence(请求对象)怎么工作 | §4 |
| Scheduler 怎么调度、怎么抢占 | §5 |
| PagedAttention 和 Prefix Caching 怎么实现 | §6 |
| ModelRunner 怎么准备输入、怎么跑模型、CUDA Graph | §7 |
| Attention 怎么读 KV cache,FlashAttention 怎么用 | §8 |
| Qwen3 模型结构、张量并行怎么切 | §9 §10 |
| 采样、tokenize、权重加载等周边 | §11 §12 §13 |
| 这些组件和官方 vLLM 的对应关系 | §14 |
| 推荐阅读顺序 | §15 |

## 1. 项目定位和文件地图

```text
nano-vllm/
├── README.md
├── example.py            # 最小使用样例
├── bench.py              # 性能基准
├── nano_vllm_learning_notes.md  # 仓库自带的精简笔记
├── SUMMARY.md
└── nanovllm/
    ├── __init__.py       # 暴露 LLM, SamplingParams
    ├── llm.py            # LLM = LLMEngine(空壳)
    ├── config.py         # Config 数据类
    ├── sampling_params.py
    ├── engine/
    │   ├── llm_engine.py     # 主循环
    │   ├── scheduler.py      # 调度器(prefill/decode)
    │   ├── block_manager.py  # Paged KV cache + Prefix cache
    │   ├── sequence.py       # Sequence 请求对象
    │   └── model_runner.py   # GPU 执行 + TP 进程 + CUDA Graph
    ├── layers/
    │   ├── attention.py      # Attention + Triton store_kvcache_kernel
    │   ├── linear.py         # 张量并行的 4 种线性层
    │   ├── embed_head.py     # VocabParallelEmbedding / ParallelLMHead
    │   ├── layernorm.py      # RMSNorm(支持 fused residual)
    │   ├── rotary_embedding.py
    │   ├── activation.py     # SiluAndMul
    │   └── sampler.py        # 采样
    ├── models/
    │   └── qwen3.py          # 完整 Qwen3 模型
    └── utils/
        ├── context.py        # 全局 Context(用模块级单例)
        └── loader.py         # safetensors 权重加载
```

代码体量(粗略):

| 模块 | 行数 | 主要职责 |
|---|---|---|
| `llm_engine.py` | ~90 | 推理主循环 |
| `scheduler.py` | ~70 | 调度 + 抢占 |
| `block_manager.py` | ~115 | 分页 + 前缀缓存 |
| `sequence.py` | ~85 | 序列状态 |
| `model_runner.py` | ~250 | TP 进程、KV cache 分配、CUDA Graph |
| `attention.py` | ~80 | KV cache 写入 + FlashAttention 调用 |
| `linear.py` | ~175 | 4 种 TP 线性层 |
| `qwen3.py` | ~215 | Qwen3 模型 |
| 其他 layers/utils | ~250 | 周边 |
| **合计** | **~1200** | |

## 2. 整体架构:四层金字塔

```text
                    ┌───────────────────────────────────┐
                    │         LLM / LLMEngine           │  用户 API
                    │   (llm.generate → add_request)    │
                    └─────────────┬─────────────────────┘
                                  │ step()
                    ┌─────────────▼─────────────────────┐
                    │             Scheduler             │  调度层
                    │   waiting/running 双队列           │
                    │   优先 prefill,否则 decode         │
                    └─────────────┬─────────────────────┘
                                  │ 调 block_manager
                    ┌─────────────▼─────────────────────┐
                    │          BlockManager             │  内存层
                    │   物理 Block + block_table         │
                    │   引用计数 + xxhash 前缀缓存        │
                    └─────────────┬─────────────────────┘
                                  │ 调 run
                    ┌─────────────▼─────────────────────┐
                    │           ModelRunner             │  执行层
                    │  prepare_prefill / prepare_decode │
                    │  run_model(CUDA Graph)            │
                    │  Qwen3ForCausalLM → Sampler       │
                    └───────────────────────────────────┘
```

**核心数据流**:

```text
用户 prompts
  → LLM.generate(...)
  → LLMEngine.add_request(prompt, sp)   # tokenize + 建 Sequence
  → Scheduler.add(seq)                   # 进 waiting 队列
  → loop step():
      Scheduler.schedule()               # 选 batch,返回 (seqs, is_prefill)
      ModelRunner.run(seqs, is_prefill)  # 准备输入、跑模型、采样
      Scheduler.postprocess()            # append token,完成则 deallocate
  → outputs = [seq.completion_token_ids for finished seqs]
  → tokenizer.decode → 文本
```

## 3. 端到端流程:从 `llm.generate` 到出 token

下面用 `example.py` 里的最小例子走一遍整个调用链,所有 `<<` 标注的行都是 nano-vllm 中的实际代码位置。

```python
# example.py
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
sp = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sp)
```

### 3.1 初始化阶段(`LLMEngine.__init__`)

`<<nanovllm/engine/llm_engine.py:17-34>>`

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # 1. 构造 Config
        config = Config(model, **kwargs)
        # 2. 如果 TP > 1, 启动子进程 (rank 1..tp_size-1)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, i, event))
            p.start()
        # 3. 主进程上初始化 rank 0 的 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        # 4. tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 5. scheduler
        self.scheduler = Scheduler(config)
```

`<<nanovllm/engine/llm_engine.py:4>>` 中的 `class LLM(LLMEngine): pass` 说明 LLM 没有任何附加逻辑,纯粹是为了对齐 vLLM 的 API 命名。

**`ModelRunner.__init__` 内部**会按顺序完成 5 件事(详见 §7):

1. 初始化分布式环境(`dist.init_process_group("nccl", ...)`)。
2. 加载模型(`Qwen3ForCausalLM(hf_config)`,然后 `load_model`)。
3. **warmup** 一次,统计显存峰值。
4. **动态分配 KV cache**(`allocate_kv_cache`)。
5. 捕获 **CUDA Graph**(除非 `enforce_eager=True`)。

### 3.2 添加请求(`LLMEngine.add_request`)

`<<nanovllm/engine/llm_engine.py:42-46>>`

```python
def add_request(self, prompt, sampling_params):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)   # 构造 Sequence
    self.scheduler.add(seq)                   # 入 waiting 队列
```

### 3.3 主循环(`LLMEngine.generate` + `step`)

`<<nanovllm/engine/llm_engine.py:48-93>>`

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()                # ① 调度
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 跑模型
    self.scheduler.postprocess(seqs, token_ids)                 # ③ 收尾
    outputs = [(seq.seq_id, seq.completion_token_ids)
              for seq in seqs if seq.is_finished]
    return outputs, num_tokens

def generate(self, prompts, sampling_params, use_tqdm=True):
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    while not self.is_finished():
        output, num_tokens = self.step()
        ...
```

主循环的关键点:

- **每个 step 只调度一次**,但调度的内容可能全是 prefill、全是 decode、或混合。
- `is_prefill` 是这一批 step 的标志,prefill 批和 decode 批**不会在同一步混合**(这是 vLLM v0 的设计,nano-vllm 保留了这个限制)。
- 完成的序列在 `step` 内就被 `outputs` 抓出,`postprocess` 会负责释放它的 KV blocks。
- `num_tokens` 在 prefill 时返回正数,在 decode 时返回负的序列数,用于在 tqdm 中显示吞吐。

## 4. Sequence:一个请求的"全生命周期"

文件 `<<nanovllm/engine/sequence.py>>`(85 行)定义了一个请求的所有状态。

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()  # 进程内全局递增,用于恢复输出顺序

    def __init__(self, token_ids, sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)            # prompt + 已生成
        self.last_token = token_ids[-1]             # decode 时只用这个
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0                  # prefix cache 命中数
        self.block_table = []                       # 逻辑块 → 物理块 ID
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

### 4.1 状态机

```text
       add_request
WAITING ─────────────► RUNNING ────────────► FINISHED
   ▲                      │                     ▲
   │   preempt(seq)        │  EOS / max_tokens  │
   └──────────────────────┘                     │
   │                                            │
   │   deallocate + appendleft                  │
   └────────────────────────────────────────────┘
```

状态变迁的代码位置:
- `WAITING → RUNNING`:`<<nanovllm/engine/scheduler.py:36>>` `seq.status = SequenceStatus.RUNNING`
- `RUNNING → WAITING`(抢占):`<<nanovllm/engine/scheduler.py:60-63>>` `preempt()`
- `RUNNING → FINISHED`:`<<nanovllm/engine/scheduler.py:69>>` `seq.status = SequenceStatus.FINISHED`

### 4.2 关键属性

| 属性 | 含义 |
|---|---|
| `seq_id` | 进程内单调递增 ID,用于最终按提交顺序输出 |
| `token_ids` | 完整 token 列表(prompt + 已生成) |
| `last_token` | decode 阶段只需要"上一步的输出 token" |
| `num_cached_tokens` | prefix cache 命中的 token 数,prefill 时**不**重算 |
| `block_table` | 该序列逻辑块 → 物理 KV block 的映射 |
| `num_blocks` | `ceil(num_tokens / block_size)` |
| `last_block_num_tokens` | 最后一个块已占用的 token 数 |
| `num_cached_blocks` | `num_cached_tokens // block_size` |
| `is_finished` | `status == FINISHED` |
| `prompt_token_ids` / `completion_token_ids` | 切片访问 |

### 4.3 `block(i)` 和 `append_token`

```python
def block(self, i):
    assert 0 <= i < self.num_blocks
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]

def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

`block(i)` 是 BlockManager 计算 hash 时调用的:它从 token 列表中切出第 i 个逻辑块。**Sequence 自己持有完整的 token 列表**,BlockManager 不需要单独存 token 副本。

### 4.4 `__getstate__` / `__setstate__`:TP 跨进程传输

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)

def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
    if self.num_completion_tokens == 0:
        self.token_ids = state[-1]
    else:
        self.last_token = state[-1]
```

**这一段非常关键**,是 vLLM TP 设计的精华:当 `tensor_parallel_size > 1` 时,主进程通过 `pickle` 把 Sequence 传到子进程(`<<nanovllm/engine/model_runner.py:76-83>>`)。**已经完成 prefill 的序列**不再需要完整 token 列表,只需要 `last_token` 一个值,这样可以**极大减少 IPC 数据量**。`num_completion_tokens == 0` 表示还在 prefill 阶段,要传完整 token_ids;否则只传 last_token。

## 5. Scheduler:连续批处理的"调度官"

文件 `<<nanovllm/engine/scheduler.py>>`(70 行)。

```python
class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

### 5.1 调度策略

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # 阶段 1: 优先 prefill
    scheduled_seqs = []
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
           or not self.block_manager.can_allocate(seq):
            break
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill=True

    # 阶段 2: decode
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())   # 抢占别人
            else:
                self.preempt(seq)                  # 抢占自己
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    self.running.extendleft(reversed(scheduled_seqs))  # 保持原顺序
    return scheduled_seqs, False  # is_prefill=False
```

**关键设计点**:

1. **prefill 优先**:有 waiting 请求就优先做 prefill,不让新请求无限等待。
2. **三重闸门**:`max_num_seqs`(并发数)、`max_num_batched_tokens`(单步 token 预算)、`can_allocate`(显存闸门),任何一个不满足就停。
3. **prefill token 预算**:`num_batched_tokens += len(seq) - seq.num_cached_tokens`。**prefix cache 命中的部分不计入预算**,这是 vLLM 节省计算的关键。
4. **decode 阶段抢占**:从 running 队列尾部开始 pop 抢占(默认 LIFO 策略,后进先抢占,防止队头长序列一直被抢)。
5. **抢占 = 退回 waiting + 释放所有 KV block**:`<<nanovllm/engine/scheduler.py:60-63>>`

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)   # 释放全部 KV block
    self.waiting.appendleft(seq)         # 放回 waiting 队头
```

**为什么 `appendleft`?** 抢占后该序列回到 waiting 队头,下次 schedule 会被优先考虑,降低首 token 时延。

### 5.2 Postprocess

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) \
           or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

**两个停止条件**(OR):
- 生成了 EOS token(且 `ignore_eos=False`)
- 已生成 token 数 == `max_tokens`

完成时**立即释放** KV block(给其他 decode 序列腾位置)。

## 6. BlockManager:PagedAttention + Prefix Caching

文件 `<<nanovllm/engine/block_manager.py>>`(115 行)。这是理解 vLLM 的"心脏"。

### 6.1 Block 和 BlockManager 数据结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0     # 共享计数
        self.hash = -1         # -1 表示不在 prefix cache 中
        self.token_ids = []    # 冗余存储,用于 hash 命中后校验

    def update(self, hash, token_ids): ...
    def reset(self):           # 被 _allocate_block 调用
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()   # prefix cache 索引
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

- `free_block_ids`:空闲 block ID 队列(O(1) 分配)。
- `hash_to_block_id`:**前缀缓存的核心**,key 是 xxhash,value 是 block_id。
- `blocks[i]`:第 i 个 block 的元数据,实际 KV 数据存在 Attention 模块的 `k_cache`/`v_cache` 里(见 §8)。

### 6.2 哈希链:Prefix Caching 的精髓

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 关键: 链入前一块的 hash
    h.update(np.array(token_ids).tobytes())      # 再 hash 当前块 token
    return h.intdigest()
```

**为什么必须链入前缀 hash?** 看这个反例:

```text
序列 1: [A][B][C]
序列 2: [X][B][C]
```

如果不链前缀,两个 `[B][C]` 的 hash 相同,会被错误复用。但有了前缀 hash,序列 1 的第二个块 hash = `hash([B], hash([A]))`,序列 2 的第二个块 hash = `hash([B], hash([X]))`,**完全不同**。

**只有完整块才计算 hash**:`len(token_ids) == self.block_size` 时才计算,否则 hash 设为 -1。这意味着**最后一个未满块永远不在 prefix cache 中**。

### 6.3 `allocate`:分配 + 复用 prefix cache

```python
def allocate(self, seq):
    assert not seq.block_table
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:    # 已经被别人在用
                block = self.blocks[block_id]
                block.ref_count += 1
            else:                                  # 在 prefix cache 但没人用
                block = self._allocate_block(block_id)
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

**算法精髓**:从前往后扫,**一旦 cache miss,后续所有块都必须重新分配**(因为哈希链断了)。cache hit 的块有两种情况:

1. block 已经在 `used_block_ids`(被别的序列占用):`ref_count += 1`,**共享 KV cache**。
2. block 不在 `used_block_ids`(在 prefix cache 中但没人用):`_allocate_block` 把它从 `free_block_ids` 移到 `used_block_ids`。

**校验 token_ids**:即使 hash 命中,也要对比 `self.blocks[block_id].token_ids != token_ids` 确认内容一致。这是为了防止 hash 冲突(虽然 64-bit hash 几乎不会冲突,但工程上要做)。

### 6.4 `deallocate` 和 `can_append` / `may_append`

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):     # 逆序
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)       # 真正放回 free 队列
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**引用计数归零才真正释放**。共享 block 在最后一个引用者释放时才回收。

```python
def can_append(self, seq):
    # 当且仅当下一个 token 会"换块"时才需要新 block
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    if len(seq) % self.block_size == 1:                       # 进入新块
        assert last_block.hash != -1                          # 旧块必须已 sealed
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
    elif len(seq) % self.block_size == 0:                     # 刚填满一个块
        assert last_block.hash == -1                          # 必须还没 hash
        token_ids = seq.block(seq.num_blocks-1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
    else:                                                     # 普通追加
        assert last_block.hash == -1
```

**三种情况**(`% block_size`):
- == 1:刚跨过 256 的整数倍 → 分配新块。
- == 0:刚填满一块 → 算 hash,加入 prefix cache。
- 其他:块未满 → 啥也不做。

> 📌 `can_append` 只在需要新块时才检查,这是个小优化。

### 6.5 一个图示

```text
假设 block_size=4,prefix cache 里已有 hash("AB") → block 7,hash("CD") → block 3

新请求:prompt = "ABCDEF"
  → num_blocks = 2
  → 第 0 块:token_ids = "ABCD",算 hash("ABCD", prefix=hash("AB"))? 
    实际算法:prefix 是上一块的 hash,初始为 -1,所以 h = hash("ABCD")
  → 第 1 块:token_ids = "EF"(不满 4),h = -1,不进 cache

  decode 1 步:token 变成 "ABCDEFG"
  → num_blocks = 2
  → may_append:len = 7, 7 % 4 = 3,普通情况

  decode 1 步:token 变成 "ABCDEFGH"
  → num_blocks = 2
  → may_append:len = 8, 8 % 4 = 0,刚好填满第 1 块
  → 算 hash("EFGH", prefix=hash("ABCD"))
  → 注册到 hash_to_block_id
```

## 7. ModelRunner:真正"跑模型"的地方

文件 `<<nanovllm/engine/model_runner.py>>`(250 行)。这是 nano-vllm 最复杂的一个文件,涵盖分布式、KV cache 分配、CUDA Graph 捕获、prefill/decode 输入准备。

### 7.1 初始化(`__init__`)

```python
def __init__(self, config, rank, event):
    self.config = config
    self.block_size = config.kvcache_block_size
    self.enforce_eager = config.enforce_eager
    self.world_size = config.tensor_parallel_size
    self.rank = rank
    self.event = event

    # 1. 初始化分布式
    dist.init_process_group("nccl", "tcp://localhost:2333",
                            world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)

    # 2. 切默认 dtype / device
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.torch_dtype)  # 通常是 bfloat16
    torch.set_default_device("cuda")

    # 3. 建模型 + 加载权重
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    self.sampler = Sampler()

    # 4. warmup → 5. 分配 KV cache → 6. 捕获 CUDA Graph
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.enforce_eager:
        self.capture_cudagraph()
```

### 7.2 分布式:简单粗暴的共享内存

```python
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()           # 非 rank 0 的子进程进入 loop
```

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

**主进程调用 `self.model_runner.call("run", seqs, is_prefill)` 时**(`<<llm_engine.py:50>>`):

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)  # rank 0 写 SHM,通知所有子进程
    method = getattr(self, method_name, None)
    return method(*args)                   # rank 0 自己同步执行
```

**这是很取巧的设计**:rank 0 通过 `pickle` 把方法名 + 参数序列化到 1MB 共享内存,然后用 Event 通知子进程,子进程反序列化后调用同名方法。子进程不直接接收 NCCL 集合通信,而是**镜像执行** rank 0 的 Python 逻辑,NCCL 只在 `ColumnParallel`/`RowParallel` 内部的 `dist.all_reduce` 中起作用。

### 7.3 `warmup_model`:为显存分配做准备

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)               # 跑一次 prefill
    torch.cuda.empty_cache()
```

**关键目的**:
- 让 PyTorch 完成 lazy kernel 初始化、CUDA context 建立。
- 让 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` 有真实数据可读。

`reset_peak_memory_stats` 在 warmup 前调用,所以 `peak` 反映的是**模型本身 + 一次最大 prefill 的显存**。后续 `allocate_kv_cache` 用这个数字反算 KV cache 容量。

### 7.4 `allocate_kv_cache`:动态算 block 数

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size   # TP 切分
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads \
                  * hf_config.head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization
                                     - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0

    # 形状: [2 (K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                self.block_size, num_kv_heads, hf_config.head_dim)
    # 把每层 slice 挂到对应 attention 模块
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**计算公式**:

```text
可用显存 = total * gpu_memory_utilization - used - peak + current
block 字节数 = 2 (K+V) × num_layers × block_size × num_kv_heads_per_rank × head_dim × sizeof(dtype)
num_blocks = 可用显存 // block 字节数
```

注意 `+ current` 是因为 warmup 时分配的临时 tensor 已经被 `empty_cache` 释放,但 `current` 记录了"现在还有多少是模型权重本身",`peak` 是"模型 + 临时 tensor"的最大值。两者相减再加回去,等价于"从分配比例里扣掉模型权重,留给 KV cache"。

**KV cache 是 1 个大张量**:`[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`,**所有层、所有 block、所有 head 共享**。每层通过 `self.kv_cache[0, layer_id]` 拿到属于自己的 slice(共享存储,无复制)。这是 PagedAttention 的"页表"层面,而 Attention 模块的 `k_cache`/`v_cache` 是"实际数据"层面,二者通过 `slot_mapping` 关联。

### 7.5 `prepare_prefill`:prefill 输入准备

```python
def prepare_prefill(self, seqs):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:
        seqlen = len(seq)
        input_ids.extend(seq[seq.num_cached_tokens:])        # 只算未缓存部分
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if not seq.block_table:
            continue
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            end = start + self.block_size if i != seq.num_blocks-1 \
                  else start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))

    # prefix cache 命中时, K 长度 > Q 长度,需要 block_table
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    # 全部转 tensor + pin_memory + non_blocking
    input_ids = torch.tensor(input_ids, dtype=torch.int64,
                             pin_memory=True).cuda(non_blocking=True)
    ...
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)
    return input_ids, positions
```

**关键概念**:

| 数据 | 含义 |
|---|---|
| `input_ids` | 所有序列未缓存 token 拼成的扁平数组 |
| `positions` | 每个 token 的位置 ID |
| `cu_seqlens_q` | query 累计序列长度(变长 attention 用) |
| `cu_seqlens_k` | key 累计长度(= prompt 长度,无 prefix 截断) |
| `max_seqlen_q` / `max_seqlen_k` | 最大长度,FlashAttention 用于分块 |
| `slot_mapping` | 每个新 token 在 KV cache 中的物理 slot |
| `block_tables` | 已有 KV 块的逻辑表(只有 prefix cache 命中时才有) |

**prefill + prefix cache 命中**:`cu_seqlens_k[-1] > cu_seqlens_q[-1]`,因为 K 看到完整上下文(包括已缓存的 prefix),Q 只看到未缓存部分。这时 Attention 层会用 `flash_attn_varlen_func` + `block_table` 从 KV cache 读已缓存部分。

### 7.6 `prepare_decode`:decode 输入准备

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)                 # 每条序列只一个 token
        positions.append(len(seq))                       # 当前位置
        context_lens.append(len(seq))                    # 上下文长度(给 FlashAttention)
        slot_mapping.append(seq.block_table[-1] * self.block_size
                            + seq.last_block_num_tokens - 1)
    ...
    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

**decode 的关键差异**:
- 每条序列**只输入 1 个 token**(`last_token`)。
- **没有 cu_seqlens**(因为每条序列长度都是 1)。
- 需要 `context_lens` 告诉 FlashAttention 每条序列当前的上下文长度。
- `slot_mapping` 计算:`最后一个 block 的起始 offset + 该 block 已有 token 数 - 1`(指向新 token 写入位置)。

### 7.7 `run_model`:决定走 eager 还是 CUDA Graph

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        for k, v in graph_vars.items():
            if k != "outputs":
                v.zero_()
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**走 CUDA Graph 的条件**(全部满足):
- 不是 prefill
- 没有 `enforce_eager`
- `batch_size ≤ 512`

**为什么 decode 才走 CUDA Graph?** decode 阶段每 step 处理 1 个 token,GPU 算得快但 CPU launch kernel 慢,launch overhead 占总时间的比例很高。CUDA Graph 一次性捕获整段计算,后续 `replay()` 直接交给 GPU 调度,极大降低 CPU 开销。prefill 阶段算力密集,launch overhead 不成问题,不需要 Graph。

### 7.8 `capture_cudagraph`:批量大小网格

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
self.graphs = {}
self.graph_pool = None
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])   # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = graph
```

- 预设了 `1, 2, 4, 8, 16, 32, 48, ...` 这一系列 batch size。
- 每个 batch size 单独 capture 一张图,运行时**选第一个 ≥ 当前 batch size 的图**(`next(x for x in self.graph_bs if x >= bs)`)。
- 使用 `graph_pool` **共享内存池**:不同 batch size 的图共享底层显存,避免每张图重复分配。

> ⚠️ **CUDA Graph 限制**:同一张图的所有输入张量必须**地址不变**(用预分配的 `graph_vars`)。`run_model` 里把当前 batch 的数据 `[:bs]` 复制到 `graph_vars` 中,然后 `replay()`,这正是用空间换时间。

## 8. Attention:FlashAttention + KV Cache 写入

文件 `<<nanovllm/layers/attention.py>>`(80 行)。包含两个关键部分:Triton 写的 `store_kvcache_kernel`,以及 `Attention` 前向。

### 8.1 `store_kvcache_kernel`:用 Triton 把 K/V 写到 cache

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**输入参数**:
- `key` / `value`:`[N, num_kv_heads, head_dim]`,N 是当前 step 的 token 数。
- `k_cache` / `v_cache`:`[num_blocks, block_size, num_kv_heads, head_dim]`。
- `slot_mapping`:`[N]`,每个 token 写入 cache 的物理 slot 索引。
- `D = num_kv_heads * head_dim`:把多个 head 合并成一个 1D 写。

**逻辑**:对每个 token(用一个 program 处理),从 `slot_mapping[idx]` 读出它在 cache 中的物理位置,然后用 `tl.store` 把整个 D 维一次性写进去。

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                               k_cache, v_cache, slot_mapping, D)
```

`grid=(N,)` 表示启动 N 个 program,每个 program 处理一个 token。**这个 kernel 性能很关键**——它会在每层 attention、每个 step 都被调用,直接决定 decode 延迟。

### 8.2 `Attention.forward`

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        ...
        self.k_cache = self.v_cache = torch.tensor([])  # 会被 ModelRunner 替换

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache 命中
                k, v = k_cache, v_cache             # ← 关键: K/V 来自 cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True,
                block_table=context.block_tables,
            )
        else:    # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale, causal=True,
            )
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
```

**两种路径,两种 FlashAttention API**:

| 阶段 | 函数 | 输入 | 特点 |
|---|---|---|---|
| Prefill 无 prefix cache | `flash_attn_varlen_func` | q, k, v 都是当前 step 算出 | 变长序列支持,内部按 cu_seqlens 切分 |
| Prefill 有 prefix cache | `flash_attn_varlen_func` | q 当前, k/v 来自 cache | K/V 用 `block_table` 从分页 cache 读 |
| Decode | `flash_attn_with_kvcache` | q 单 token, k/v 来自 cache | 直接基于 KV cache 算当前 token attention |

**关于 `block_table`**:FlashAttention 的 `block_table` 参数接受 `[batch, max_blocks]` 的 int32 tensor,告诉 kernel 怎么从分页 cache 中 gather K/V。这是把 PagedAttention 集成到 FlashAttention 的关键。

**关于 `q.unsqueeze(1)`**:`flash_attn_with_kvcache` 要求 query 的形状是 `[batch, 1, num_heads, head_dim]`(每条序列 query 长度 1),所以要把 `[N, num_heads, head_dim]` 加一个维度。

## 9. Qwen3 模型结构

文件 `<<nanovllm/models/qwen3.py>>`(215 行)。

```text
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding
│   ├── layers: nn.ModuleList[Qwen3DecoderLayer × N]
│   │   ├── input_layernorm: RMSNorm
│   │   ├── self_attn: Qwen3Attention
│   │   │   ├── qkv_proj: QKVParallelLinear
│   │   │   ├── q_norm: RMSNorm(head_dim)         ← Qwen3 特有
│   │   │   ├── k_norm: RMSNorm(head_dim)         ← Qwen3 特有
│   │   │   ├── rotary_emb: RotaryEmbedding
│   │   │   ├── attn: Attention                    ← §8
│   │   │   └── o_proj: RowParallelLinear
│   │   ├── post_attention_layernorm: RMSNorm
│   │   └── mlp: Qwen3MLP
│   │       ├── gate_up_proj: MergedColumnParallelLinear
│   │       ├── act_fn: SiluAndMul
│   │       └── down_proj: RowParallelLinear
│   └── norm: RMSNorm
└── lm_head: ParallelLMHead
```

### 9.1 DecoderLayer:fused residual + RMSNorm

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        # 第一层: 没有 residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        # 后续层: 把残差加和 norm 融合
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**`add_rms_forward`**(在 `layernorm.py`)把 `x = x + residual` 和 `x = norm(x)` 融合成一个 kernel:

```python
@torch.compile
def add_rms_forward(self, x, residual):
    orig_dtype = x.dtype
    x = x.to(torch.float32).add_(residual.to(torch.float32))  # 融合 1
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

这样**每层只多 1 个 residual tensor 占用显存**(而不是 2 个)。`torch.compile` 进一步把这个函数编译成高效 kernel。

### 9.2 Qwen3Attention:per-head QK Norm

```python
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_by_head = q.view(-1, self.num_heads, self.head_dim)
    q_by_head = self.q_norm(q_by_head)               # Qwen3 特有
    q = q_by_head.view(q.shape)
    k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
    k_by_head = self.k_norm(k_by_head)               # Qwen3 特有
    k = k_by_head.view(k.shape)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)
    output = self.o_proj(o)
    return output
```

**Qwen3 关键差异**:和 LLaMA 不同,Qwen3 对 Q 和 K 的**每个 head 单独做 RMSNorm**(`head_dim` 大小),这能提升训练稳定性。

### 9.3 SwiGLU MLP

```python
class Qwen3MLP(nn.Module):
    def forward(self, x):
        gate_up = self.gate_up_proj(x)   # 一次性输出 [gate, up]
        x = self.act_fn(gate_up)         # silu(gate) * up
        x = self.down_proj(x)
        return x
```

`SiluAndMul`(`<<layers/activation.py>>`):

```python
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)
    return F.silu(x) * y
```

把一个 `2 * intermediate` 维度的 tensor 在最后一维切成两半,前一半过 SiLU,和后一半逐元素相乘。这就是 SwiGLU 激活。

### 9.4 Qwen3ForCausalLM

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**`packed_modules_mapping`** 是权重加载的关键:

- HF 格式:`model.layers.0.self_attn.q_proj.weight`、`k_proj.weight`、`v_proj.weight` 三个独立文件。
- nano-vllm 格式:合并成 `qkv_proj.weight`,每条通过 `QKVParallelLinear.weight_loader` 切到正确的 slice。

`compute_logits` 单独抽出来,是因为**只有最后一个 token 的 hidden state 需要算 logits**,ModelRunner 会利用这一点减少 LM head 计算量(见 §10.2)。

## 10. 张量并行(Tensor Parallelism)

文件 `<<nanovllm/layers/linear.py>>` 和 `<<nanovllm/layers/embed_head.py>>`。

### 10.1 4 种线性层

| 类 | 切分方式 | 通信 | 用途 |
|---|---|---|---|
| `ReplicatedLinear` | 不切 | 无 | 单卡等价 |
| `ColumnParallelLinear` | 输出维度切 | 后续是否 all-reduce 看下游 | QKV projection、MLP gate/up |
| `RowParallelLinear` | 输入维度切 | forward 末尾 all-reduce | attention output proj、MLP down proj |
| `QKVParallelLinear` | 继承 Column | 后续是否 all-reduce 看下游 | 合并 Q+K+V 三个投影 |
| `MergedColumnParallelLinear` | 继承 Column | 后续是否 all-reduce 看下游 | 合并 gate+up |

### 10.2 `ColumnParallelLinear`(输出切分)

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        super().__init__(input_size, output_size, 0)   # tp_dim=0 表示按行切
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        ...

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)   # 不需要 all-reduce
```

每个 rank 只算自己分到的输出维度。`weight_loader` 把 HF 完整权重按行切,只 copy 当前 rank 的那一片。

### 10.3 `RowParallelLinear`(输入切分)

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        super().__init__(input_size, output_size, 1)   # tp_dim=1 表示按列切
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        ...

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

每个 rank 输入维度只有 `input_size / tp_size`,因此计算出的 `y` 是不完整的(只是部分和),需要 `all_reduce` 汇总。**只 rank 0 算 bias**,因为所有 rank 加完 bias 之后是 N 份相同的 bias 相加。

### 10.4 `QKVParallelLinear`:QKV 三合一

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None, bias=False):
        ...
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias=bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # loaded_shard_id ∈ {"q", "k", "v"}
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:  # "v"
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

**Qwen3 用 GQA**(Grouped Query Attention):`num_kv_heads` 通常比 `num_heads` 少(例如 16 query heads 对 8 kv heads),通过 GQA 减少 KV cache 内存。`QKVParallelLinear` 内部按 `[Q heads, K heads, V heads]` 顺序拼接,每个 rank 只 load 自己负责的 heads。

### 10.5 `VocabParallelEmbedding` 和 `ParallelLMHead`

```python
class VocabParallelEmbedding(nn.Module):
    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

**Embedding 切分思路**:
1. 每个 rank 只存自己那段 vocab 的 embedding。
2. forward 时,token ID 是完整 vocab 的全局 ID,通过 mask 标记"是否在本 rank"。
3. 不在的 token 减成本地 ID 后**乘 mask = 0**,在的 token 正常查表。
4. 最后 `all_reduce` 求和,得到每条 token 的完整 embedding(不在自己 rank 的 token 贡献 0,在的贡献真实值)。

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            # 只取每条序列最后一个 token 的 hidden state
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] \
                         if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

**prefill 优化**:`cu_seqlens_q[1:] - 1` 取出每条序列最后一个 token 的下标(因为变长 attention 中,每条序列的最后一个位置就是 `cu_seqlens_q` 中的结束边界减 1)。这样 **LM head 只需要算 N 个 token(seq 数)的 logits,而不是整个 prompt**。

**TP 下**:rank 0 gather 所有 rank 的部分 logits(`dist.gather` 而非 `all_reduce`),然后 concat 拼成完整 vocab 维度。

## 11. 采样 Sampler

文件 `<<nanovllm/layers/sampler.py>>`(20 行)。

```python
class Sampler(nn.Module):
    def forward(self, logits, temperatures):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
```

**Gumbel-max 技巧**:

```text
E ~ Exponential(1)
argmax(p_i / E_i) ~ Categorical(p)
```

为什么? 取 `g_i = -log(-log(u_i))` (Gumbel),等价于 `argmax(log p_i + g_i)`。而 `-log(E_i) = g_i`,所以 `argmax(p_i / E_i) = argmax(log p_i - log E_i) = argmax(log p_i + (-log E_i))`,正好等价。

**两种模式**:
- `temperature == 0`:走 `argmax`,greedy decoding。
- `temperature > 0`:走 Gumbel-max sampling。
- `torch.where` 一行完成两种模式的选择。

## 12. 权重加载

文件 `<<nanovllm/utils/loader.py>>`(30 行)。

```python
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)

def load_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

**两种加载路径**:

1. **packed 权重**(在 `packed_modules_mapping` 中):比如 `q_proj.weight` → 改名 → `qkv_proj.weight`,然后调用**该参数的 weight_loader**(`QKVParallelLinear.weight_loader`)做 TP 切片。
2. **普通权重**:直接用参数的 `weight_loader`(由 `ColumnParallelLinear` 等定义)做 TP 切片,或 fallback 到 `default_weight_loader`(纯拷贝)。

**关键点**:每个 `nn.Parameter` 上挂 `weight_loader` 属性,而不是用全局字典。这让每个并行层可以**自定义自己的切片方式**(QKV 和 gate_up 都需要传入 `shard_id`)。

## 13. 周边模块

### 13.1 `utils/context.py`:全局 Context

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, ...):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, ...)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**模块级单例**。Attention、LMHead 等模块都通过 `get_context()` 拿到当前 step 的 metadata。**这是一种隐式的全局状态**——优点是避免到处传参,缺点是不易扩展。vLLM 主仓库也保留了这种模式。

### 13.2 `rotary_embedding.py`:RoPE

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        ...
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_sin_cache", torch.cat((cos, sin), dim=-1), persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query.view(num_tokens, -1, head_size), cos, sin).view(query_shape)
        key = apply_rotary_emb(key.view(num_tokens, -1, head_size), cos, sin).view(key_shape)
        return query, key
```

- 预计算 `cos_sin_cache` 存到 buffer,**避免每 step 重算**三角函数。
- forward 用 `positions` 索引需要的 cos/sin,大幅减少计算量。
- `cos_sin_cache` 不持久化(`persistent=False`),节省权重文件大小。
- `get_rope` 用 `lru_cache(1)` 缓存一个 RotaryEmbedding 实例(所有层共享同一份 cos_sin_cache)。

### 13.3 `sampling_params.py`

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

**vLLM 完整版**还支持 `top_p`、`top_k`、`presence_penalty`、`frequency_penalty`、`stop`、`logprobs` 等,nano-vllm 只保留最简版。

### 13.4 `config.py`

```python
@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0     # 必须能被 256 整除
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

**`kvcache_block_size % 256 == 0`** 的硬约束:FlashAttention 的某些优化路径要求对齐;`<= 256` 的因子也行,但 nano-vllm 选择只允许 ≥ 256。

## 14. 关键优化原理(vLLM 核心思想)

### 14.1 Continuous Batching(连续批处理)

**传统静态批处理**:一个 batch 内 N 条请求必须**同时开始、同时结束**。短请求要等长请求,GPU 利用率低。

**nano-vllm 的实现**:
- `waiting`:还没进 batch 的新请求。
- `running`:已经 prefill 完、正在 decode 的请求。
- 每次 `schedule()` **优先 prefill 新请求,否则 decode running 请求**。完成的请求立即释放资源,新请求能立刻补位。

```text
Static batching:  [req1 ████][req2 ████][req3 ████] → 等待最慢的
Continuous:       [req1 ████]
                  [req2 ████][req2 ████][req2 ██]
                  [req3 ████][req3 ██]              ← 新请求随时补位
```

### 14.2 Prefill/Decode 分离

| 阶段 | 输入 | 计算模式 | 瓶颈 |
|---|---|---|---|
| Prefill | 整个 prompt(N 个 token) | GEMM 主导,计算密集 | GPU 算力 |
| Decode | 每条 1 token | Memory-bound,小 kernel 频繁 | 显存带宽 + CPU launch |

**nano-vllm 怎么做**:
- 准备输入时区分(`prepare_prefill` vs `prepare_decode`)。
- 选不同 FlashAttention API。
- 只有 decode 走 CUDA Graph。

### 14.3 PagedAttention

KV Cache 被切分为固定大小 block,通过 `block_table` 间接寻址。**核心收益**:

1. **避免预分配浪费**:不需要为每条请求预留 `max_model_len` 长度的连续 KV cache。
2. **降低内存碎片**:操作系统分页思想。
3. **支持变长 batch**:不同长度的序列可以共存于一个 batch。
4. **支持 prefix 共享**:多个序列可以共享同一个物理 block(引用计数)。

### 14.4 Prefix Caching

**链式 hash** + `hash_to_block_id` 索引 = 命中相同 prefix 直接复用 KV cache。

**典型场景**:
- 相同 system prompt 的多轮对话。
- 共享 few-shot 示例的批量推理。
- 相同文档的多次 QA。

**触发条件**:
- 块必须**完整填满**(`len == block_size`)。
- 序列一旦 cache miss,后续所有块必须重新分配(哈希链断了)。

### 14.5 FlashAttention

- `flash_attn_varlen_func`:变长序列支持,通过 cu_seqlens 把多个不同长度序列打包成一个 batch。
- `flash_attn_with_kvcache`:专门为 decode 阶段设计,直接基于 paged KV cache 计算。
- **核心优势**:不显式 materialize 巨大的 attention matrix,大幅减少 HBM 读写,适合长序列。

### 14.6 Tensor Parallelism

**列切分**(ColumnParallel):输出维度分到多卡,各自算各自的输出,无需 all-reduce。
**行切分**(RowParallel):输入维度分到多卡,各算部分和,末尾 all-reduce 汇总。

**Qwen3 上的切分**:

```text
QKV proj:   Column 切 (无 all-reduce,直接进入 attention)
O proj:     Row    切 (all-reduce 之后接下一层)
Gate/Up:    Column 切 (无 all-reduce,直接进入 SiluAndMul)
Down:       Row    切 (all-reduce 之后接下一层)
Embedding:  Vocab  切 (mask + all-reduce)
LM Head:    Vocab  切 (gather 拼接)
```

### 14.7 CUDA Graph

**为什么 decode 特别需要**?decode 阶段每 step 只算 1 token,但 launch 几十个 CUDA kernel(attention、norm、linear...),CPU 调度开销占总时间比例可能高达 30-50%。

**怎么解决**:把"形状固定"的一串 CUDA 操作捕获成一张图,后续 `replay()` 由 GPU 自己调度,CPU 不再介入。

**限制**:
- 同一张图的所有输入必须**地址不变**。所以 nano-vllm 预分配了 `graph_vars` tensor,运行时把当前数据 `[:bs]` 复制过去。
- 不同 batch size 需要不同图。nano-vllm 预 capture `1, 2, 4, 8, 16, 32, ...`,运行时选第一个 ≥ 当前 bs 的图。

### 14.8 Torch Compile

`RMSNorm`、`SiluAndMul`、`RotaryEmbedding.forward` 都加了 `@torch.compile`:

- 减少 Python overhead。
- 自动融合小算子(比如把 `chunk` + `silu` + `*` 融成一个 kernel)。
- 自动选择最优实现(为不同 shape 编译专用 kernel)。

## 15. 与官方 vLLM 的对应关系

| nano-vllm 组件 | 官方 vLLM 中的位置 |
|---|---|
| `LLMEngine` | `vllm/v1/engine/` (v1) 或 `vllm/engine/llm_engine.py` (v0) |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` (v1) |
| `Sequence` + `SequenceStatus` | `vllm/v1/sequence.py` |
| `BlockManager` + `Block` | `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/block_pool.py` |
| `block_table` | `block_table` (相同概念) |
| `hash_to_block_id` + `xxhash` | `vllm/v1/core/kv_cache_utils.py` 的 `BlockHash` |
| `ModelRunner` | `vllm/v1/worker/gpu_model_runner.py` |
| `prepare_prefill` / `prepare_decode` | `prepare_inputs` 内部 |
| `capture_cudagraph` | `vllm/compilation/cuda_graph.py` |
| `Attention` | `vllm/attention/backends/flash_attn.py` |
| `store_kvcache_kernel` | 在 vLLM 中由 C++/CUDA 实现 |
| `Sampler` | `vllm/model_executor/layers/sampler.py` |
| `ColumnParallelLinear` | `vllm/model_executor/layers/linear.py` |
| `RowParallelLinear` | 同上 |
| `QKVParallelLinear` | 同上 |
| `VocabParallelEmbedding` / `ParallelLMHead` | 同上 |
| `load_model` + `packed_modules_mapping` | `vllm/model_executor/model_loader/` |

**官方 vLLM 比 nano-vllm 多的东西**:

1. **Async engine**:`AsyncLLMEngine` + asyncio,支持 streaming 输出。
2. **OpenAI API server**:HTTP/gRPC server,支持 `/v1/chat/completions` 等。
3. **多种 model**:LLaMA、Mistral、Mixtral、Qwen、GPT-OSS、DeepSeek 等。
4. **Chunked prefill**:把长 prompt 切成多段,和 decode 混合调度。
5. **Speculative decoding**:draft model + verify,加速 2-3x。
6. **Quantization**:GPTQ、AWQ、FP8、INT4 等。
7. **LoRA**:动态加载 adapter。
8. **Pipeline parallel**:跨节点分阶段。
9. **多种 attention backend**:FlashInfer、XFormers、FlashAttention2/3、Triton。
10. **更复杂的采样**:top_p, top_k, min_p, penalties, stop, logprobs。
11. **prefix cache 策略**:LRU eviction、eviction policy。

**但核心抽象一致**:理解 nano-vllm 后,看官方 vLLM 就是把这些概念"工程化"。

## 16. 阅读建议路线

| 步骤 | 文件 | 重点 |
|---|---|---|
| 1 | `README.md`, `example.py` | 知道怎么用 |
| 2 | `nanovllm/__init__.py`, `nanovllm/llm.py` | 入口 |
| 3 | `nanovllm/config.py`, `nanovllm/sampling_params.py` | 知道有哪些配置 |
| 4 | `nanovllm/engine/llm_engine.py` | 主循环,理解 `step()` |
| 5 | `nanovllm/engine/sequence.py` | 理解请求对象 |
| 6 | `nanovllm/engine/scheduler.py` | 理解调度,理解 `preempt` |
| 7 | `nanovllm/engine/block_manager.py` | 重点:PagedAttention + Prefix Cache |
| 8 | `nanovllm/engine/model_runner.py` | 重点:warmup、KV cache 分配、CUDA Graph |
| 9 | `nanovllm/utils/context.py` | 知道 Attention 怎么拿到 metadata |
| 10 | `nanovllm/layers/attention.py` | 重点:Triton kernel + FlashAttention |
| 11 | `nanovllm/layers/linear.py` | 张量并行 |
| 12 | `nanovllm/layers/embed_head.py` | 词表并行 |
| 13 | `nanovllm/layers/layernorm.py`, `activation.py`, `rotary_embedding.py` | 周边算子 |
| 14 | `nanovllm/models/qwen3.py` | 模型结构 |
| 15 | `nanovllm/layers/sampler.py` | 采样 |
| 16 | `nanovllm/utils/loader.py` | 权重加载 |
| 17 | `bench.py` | 看怎么 benchmark |

**对照阅读**:每读一个文件,回到本文的对应章节;再去看官方 vLLM 的对应位置。

## 17. 常见疑问(基于代码反推)

### Q1:为什么 `kvcache_block_size` 必须能被 256 整除?

为了和某些硬件/CUDA 库的对齐要求兼容。FlashAttention 在某些路径上对 block size 有 256 字节的对齐要求,nano-vllm 把这个限制直接暴露成 API 约束。

### Q2:`num_kvcache_blocks` 是怎么算出来的?

`int(total * gpu_memory_utilization - used - peak + current) // block_bytes`。详见 §7.4。

### Q3:decode 阶段 batch 包含 prefill 的序列吗?

**不包含**。nano-vllm 的 `schedule()` 设计是"非 prefill 即 decode"——一次 step 要么全是 prefill,要么全是 decode。这是 vLLM v0 的设计,v1 才引入了 chunked prefill 允许混合。

### Q4:为什么 prefill 时也要传 `block_table`?

只有 prefix cache 命中时才需要(代码中 `if cu_seqlens_k[-1] > cu_seqlens_q[-1]`)。FlashAttention 用 `block_table` 从分页 KV cache gather 已缓存的 K/V,避免重算。

### Q5:preempt 后,被抢占的序列 prefix cache 还有效吗?

**完全有效**。`deallocate` 只把 block 引用计数 -1,只有 `ref_count == 0` 时才真正释放。即使在 prefix cache 中,只要 `ref_count > 0`,`hash_to_block_id` 映射就保留。被抢占的序列下次 prefill 还能命中之前的 prefix cache。

### Q6:`enforce_eager=True` 时性能会差多少?

decode 阶段慢 30-50%(具体看模型)。CUDA Graph 是 nano-vllm 性能的关键优化之一。开发调试时用 `enforce_eager=True` 比较方便,正式推理时建议关掉。

### Q7:nano-vllm 和 vLLM v0/v1 哪个对应?

nano-vllm 是 vLLM **v0 思想**的精简实现:Scheduler-Worker 分离、prefill/decode 分离、TP 简单 SHM 通信。它**没有** vLLM v1 的 multi-step scheduling、chunked prefill 等高级特性。

### Q8:为什么 `samper.py` 里同时算了 `greedy_tokens` 和 `sample_tokens`?

为了用一个 kernel 覆盖两种模式,最后用 `torch.where` 选择。如果想省一点,可以在 Python 层分流,只跑对应分支。

### Q9:`Sequence.__getstate__` 里的优化真的有效吗?

非常有效。decode 阶段一个序列可能已经 1000+ tokens,完整传 token 列表要 ~4KB(int32);只传 `last_token` 只要 4 字节。100 条序列差 400KB,跨进程 IPC 时这差距很明显。

## 18. 一句话总结

> **nano-vllm 用 1200 行 Python 实现了 vLLM 的"骨架":**
> - **Scheduler** 做连续批处理和抢占,
> - **BlockManager** 用分页 + 哈希链做 KV cache 和 prefix cache,
> - **ModelRunner** 用 warmup 算显存、分配 KV cache、捕获 CUDA Graph,跨 TP rank 用共享内存 + NCCL,
> - **Attention** 用 Triton kernel 写 KV cache,用 FlashAttention 读 KV cache,
> - **各种 parallel linear** 把模型切到多卡,
> - **Sampler** 用 Gumbel-max 完成采样。
>
> **这套抽象就是 vLLM 的"魂"**。读懂它,再去看 vLLM 主仓库(几万行 C++/Python 混合),就不会迷失。

## 附:文件 → 章节速查

| 文件 | 本文章节 |
|---|---|
| `nanovllm/__init__.py` | §1 |
| `nanovllm/llm.py` | §1, §3.1 |
| `nanovllm/config.py` | §13.4 |
| `nanovllm/sampling_params.py` | §13.3 |
| `nanovllm/engine/llm_engine.py` | §3 |
| `nanovllm/engine/sequence.py` | §4 |
| `nanovllm/engine/scheduler.py` | §5 |
| `nanovllm/engine/block_manager.py` | §6 |
| `nanovllm/engine/model_runner.py` | §7 |
| `nanovllm/utils/context.py` | §13.1 |
| `nanovllm/utils/loader.py` | §12 |
| `nanovllm/models/qwen3.py` | §9 |
| `nanovllm/layers/attention.py` | §8 |
| `nanovllm/layers/linear.py` | §10.1 - §10.4 |
| `nanovllm/layers/embed_head.py` | §10.5 |
| `nanovllm/layers/layernorm.py` | §9.1 |
| `nanovllm/layers/rotary_embedding.py` | §13.2 |
| `nanovllm/layers/activation.py` | §9.3 |
| `nanovllm/layers/sampler.py` | §11 |

