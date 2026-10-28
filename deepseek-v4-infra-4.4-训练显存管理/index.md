# DeepSeek V4 Infra Deep Dive 4.4: PyTorch 训练显存管理


> 来源：[DeepSeekV4 Infra Deep Dive · 第四节 4.4: PyTorch 训练显存管理：从 Caching Allocator 到 Stream 同步](https://zhuanlan.zhihu.com/p/2052200237373118275)
>
> 系列：可灵 AI infra 训练团队出品，全文系列见原文末链接。本文是该系列的 4.4 一篇，专门讲训练显存怎么分、怎么 OOM、怎么省。

笔记以"按生命周期分四类"为主线展开。**比按"哪一项占得多"分类更能看清 OOM 卡在哪一环**：从初始化到 step 结束，四类显存分别在什么时刻驻留、用什么手段能挪走，挪到哪里去。

---

## 01 `allocated` 与 `reserved` 的语义

这一段是后面所有判断的前提。

- `allocated` = caching allocator 实际切给存活 tensor 的字节数
- `reserved` = allocator 从驱动一次性申请到的 segment 总量；驱动一次分一大块连续显存，叫一个 segment

两者永远满足 `reserved ≥ allocated`，差出来的是 allocator 持有的、暂时既不还驱动也没分出去的空闲 block，也就是**碎片**。

```text
reserved = allocated + 碎片 + 瞬态峰值被抬高后留下的「台阶」
```

「台阶」是 OOM 判断里最容易踩的坑：caching allocator 的 `free` 只把 block 退回池里复用、并不向驱动退还，整段 segment 全部空闲才有资格归还。瞬态峰值那几块大 block 常常和长寿命 block（激活缓存、通信预留）位于同一 segment，始终无法同时空出来，于是 `reserved` 只升难降——它记录的**不是此刻用了多少**，而是**训练至今某一瞬间最多需要过多少连续显存**。

> 「`allocated` 不高却 OOM」通常就是这道 `reserved` 台阶造成的。

---

## 02 四类训练显存：按生命周期划分

训练显存的增量主要来自反向传播。推理时单卡 HBM 上基本只有 weights，加一份随生成长度增长的 KV cache；训练多出的十几倍几乎都来自反向——chain rule 要求前向中间结果和梯度副本一直驻留到反向用完，于是 activation、梯度、optimizer state、重算中间值才同时占用卡上 HBM。下面四类是**同一个反向需求在不同生命周期上的分布**，去掉反向（即推理）它们全部消失。

按生命周期划分，关注每一类显存什么时候分配、什么时候释放、有没有可能跨 step 还被保留。四类按占用从高到低排列：当前最高的那类降下来后，下一类就成为主导项。

| 类别 | 生命周期 | 单卡量级（V3） |
|---|---|---|
| 恒定占用（weights / grads / opt state） | 跨 step 不变 | 68.5 GB |
| Activation（forward saved） | 单 step 单峰 | 131 GB（未压缩） |
| 计算峰值（kernel 内瞬态） | μs 尖峰 | 单层 GiB 级 |
| 通信 buffer（NCCL 预留） | 跨 step 常驻 | 数百 MiB ~ 1 GB（EP all-to-all + DP all-reduce） |

下面逐类拆开。

### 02.1 恒定占用：weights / gradients / optimizer states

生命周期跨 step 不变：从初始化那一刻分配出来，持续保留到训练结束。每一步前向 / 反向 / optimizer 都要读它们，没有任何释放窗口。

体现在 `reserved` 曲线上：训练启动后几个 step 内，`reserved` 分几次跳到一个稳定值，此后整段训练再不变。

必须提前一次性分配。DDP 下模型刚 build 出来就占满。FSDP / ZeRO 是特例，按 rank 分片到多卡。除了沿 DP 副本维度分片的 FSDP / ZeRO，TP（按 head / hidden 划分）、PP（按层划分）、EP（按 expert 划分）也都在削减这条恒定基线，只是各占一个不同的维度。

### 02.2 Activation：前向累积、反向释放

生命周期由 chain rule 决定：前向算到第 i 层时产生的中间结果，要保留到反向传播回第 i 层、读取完成后才释放。中间隔着「forward 跑完剩下层 + backward 倒回这一层」几十毫秒到秒级的窗口——**跨 kernel 却不跨 step**。

`allocated` 曲线上是随 forward 单调累积、随 backward 单调消费的单峰曲线——前向到达峰值时全部 activation 都保留在 HBM 上，backward 每过一层释放一份。这段累积-释放曲线是 Activation Checkpointing 的优化目标。

每个 micro-batch 临时申请：每一步开始前 activation 都不存在；前向算到哪层，那层中间结果就被 allocator 划出 block；分配路径仍留在 caching allocator 内部。

上一个 micro-batch 释放的同档 block 留在池里复用。warmup 之后几乎所有 activation 分配都在 allocator 内部命中已有 block，驱动那条较慢的路径只在分桶不匹配（请求大小不落进任一现成 block 的尺寸区间）或 segment 已用尽时才走。

### 02.3 单个 kernel 内的瞬态峰值

进入 kernel 时申请、退出 kernel 立刻释放，存在时间只有毫秒甚至微秒级。MoE 的 expert FFN intermediate、FlashAttention backward 的 work tensor、cuBLAS GEMM 的 workspace 都属于这一类。

在 MemTrace 上表现为一道 kernel 刻度宽的尖峰，按时间平均下来几乎看不出来，瞬时却常常达到单层 GiB 级。因为存在时间太短、不进入前向保存的激活，手算 saved activation 时容易漏掉。

**常驻还是临时**：看上去是临时的，kernel 跑完就标记为空闲。但 caching allocator 的 `free` 只把 block 退回池里复用、并不向驱动退还，整段 segment 全部空闲才有资格归还；瞬态峰值那几块大 block 常常和长寿命 block 位于同一 segment，始终无法同时空出来，于是 `reserved` 只升难降。

> `allocated` 曲线规律地起落，`reserved` 却是台阶式抬升。

这类瞬时峰值的根本解决办法是 fused kernel，把多个 kernel 的中间结果保留在 SRAM、不写回 HBM，单次数据量不会产生几 GiB 的瞬态 buffer。

### 02.4 通信 buffer：集合通信的中转缓冲

训练一开始就预分配，此后一直占用。NCCL 按上限一次性预留每条通道的发送 / 接收缓冲、ring 上的中转 buffer、reduce 算法的工作区。

尺寸由四个因素共同决定：message size、rank 数、NCCL 选定的算法（ring / tree / NVLS）、bucket 聚合。

若是等训练主循环跑起来之后才临时申请，`cudaMalloc` 会阻塞发起它的 CPU 线程，并触发 device-wide 同步：整卡所有 stream 已入队的工作要先排空，卡在集合通信的关键路径上。中后期 `reserved` 已接近上限，驱动这一刻可能分配不出连续的几百 MiB，等到通信发不出去才发现显存不够，训练 hang 在集合通信上。预分配在启动时一次性解决这些不确定性，跑起来之后通信 buffer 占用恒定不变，不再与主计算争用驱动。

### 02.5 通信 stream：overlap 与显存池分裂的取舍

同一条 stream 上的 kernel 严格按提交顺序串行执行，通信和计算共用同一条 stream 时，allreduce 一发就阻塞后面的 GEMM，重叠执行无从谈起。

为了 overlap，NCCL 的集合通信通常跑在独立的通信 stream 上：集合通信 kernel 独立排队执行，主计算 stream 同时还能发 GEMM。这不是强制要求——集合通信留在 default stream 一样能跑通，只是退化成串行，但 buffer 进入主池反倒不分裂、碎片更少。只有真正存在数据依赖的关键点才用 `wait_event` 显式同步，比如 backward 算到 dW 那一刻等待上一步 allreduce 的 event。

副作用是**显存池跟着分裂**。caching allocator 按「一个池只服务一条 stream」的规则过滤空闲 block（隔离用于保证正确性：A stream 的 kernel 可能还没读完这块 block，B stream 拿去写会发生数据竞争）。主计算 stream 的池占满之后，通信 stream 名下整段 segment 即便全部空闲也无法被主计算 stream 复用，OOM 由此产生。

反过来，通信 stream 上要是没多少计算 / 通信可重叠，或显存已经紧张到池分裂的额外开销划不来，就不要分出去——让通信留在主 stream，用一点串行换回不分裂的显存池更划算。

---

## 03 四类合计：DeepSeek V3 建模

四类合计对 V4 同样成立。下表 per-GPU 数字取自 NVIDIA Megatron Core MoE 报告对 V3 配置的建模（报告设 256 GPU、PP4×VPP4×EP64，对标 H100 80 GB 容量）：

| 类别 | per-GPU 占用 | 生命周期 |
|---|---|---|
| 恒定占用（weights + gradients + optimizer states） | 68.5 GB | 全程恒定 |
| Activation（forward saved，未压缩） | 131 GB | 单峰 |
| 计算峰值（MoE 专家 FFN 中间值等单 kernel 瞬态） | 单层 GiB 级 | μs 尖峰 |
| 通信 buffer（NCCL EP all-to-all + DP allreduce 预留） | 数百 MiB ~ 1 GB 量级 | 全程恒定 |

表中通信 buffer 只计 EP all-to-all + DP allreduce 预留；再加上 PP send/recv（点对点，非集合通信）预留，整体 ≈ 3 GB。

第一行给出恒定基线 68.5 GB，只剩 11.5 GB 供动态显存使用。第二行 Activation 131 GB 是这点余量的十倍有余——recompute、FP8 storage、offload 都是要把它压缩到 11.5 GB 以内；压缩不下去，就只能继续划分恒定基线，腾出更多余量。

未优化前合计 199.5 GB ≈ 2.5× H100。目标是把 per-GPU 总占用压回 80 GB；131 GB 的 activation 是首要压缩目标。

### 03.1 真实峰值不是各类最大值的简单相加

四类峰并不在同一时刻。判断某套配置会不会超出容量，要逐时刻看占用：

- 恒定占用与通信 buffer 是两条**水平线**
- activation 是一个**单峰**
- 计算峰值是一个**尖峰**

某一刻的占用 = 那一刻各类的瞬时值相加；真峰值 = 这个逐时刻之和在整个 step 里取到的最大值。峰值时刻 $t^*$ 会随配置变化：

- **不开 recompute** 时，activation 单峰停在 forward 末端，$t^*$ 位于那里。
- **开 full recompute** 后，forward 末端那段保存的激活被削平，但 backward 早期出现一个新窗口：层间保存的激活还没卸掉几层、当前层 recompute 重算出的 forward intermediate、当前层新分配的 grad buffer 三类在同一窗口叠加，$t^*$ 从 forward 末端移到 backward 早期。

这只是把它推迟到反向、并非删除。峰值是否下降取决于移入窗口的高度：低于被削平那段就降低（V3 full recompute 即把峰值从 199.5 GB 压到 ~85 GB），要是比削平那段还高，反而会更早 OOM。

---

## 04 MemTrace 与 OOM 诊断

### 04.1 怎么打 snapshot

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# 跑几个 step
torch.cuda.memory._dump_snapshot(path)
```

得到的 pickle 文件里，每一次 alloc / free 都带着时间戳、字节数、所在 stream 和 Python stack trace。

把 `allocated` 和 `reserved` 按时间画出来，是 MemTrace 时间线；取 `allocated` 最高那一刻各类的瞬时值、按 tensor 归类排列，是峰值占用分解。

### 04.2 案例：peak ≈ 93.5 GB（部分 recompute 后）

下面这张 MemTrace 时间线，浅色区域是 `allocated`，深色线是 `reserved`；峰值落在 backward 早期 recompute 窗口，OPT 段另有约 10 GB 的临时 m / v 副本尖峰。

![MemTrace](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260707110818-1.jpg)

下面这张是同一时刻峰值截面按 tensor 来源分解，合计约 93.5 GB，高出 H100 80 GB 上限约 13.5 GB。

![峰值占用分解](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260707110818-2.jpg)

同一时刻的占用分解里，六项是：

| 分量 | 大小 | 说明 |
|---|---|---|
| 恒定占用 | 68.5 GB | Megatron 报告 Table 3 拆为 BF16 weights+gradients 36.4 GB + FP32 main weights+optimizer states 32.1 GB |
| 通信 buffer | 3 GB | PP send/recv（P2P）+ 各集合通信预留池 |
| Activation 保存的激活 | 12 GB | 部分 recompute 后还没被反向消费完 |
| recompute forward intermediate | 2 GB | 当前层重算出的中间量 |
| grad buffer | 3 GB | 当前层 backward 临时分配的 dY/dX/dW |
| allocator overhead | 5 GB | `reserved − allocated` 之间的碎片余量 |

这六类与开头四类并非一一对应：Activation 在峰值处拆成层间保存的激活和 recompute intermediate（计算峰值瞬态归入后者的重算窗口），另起 grad buffer 与 allocator overhead；恒定占用与通信 buffer 在占用分解里保持不变。

### 04.3 OOM 的两种典型情况

**情况 A：`reserved` 与 `allocated` 都接近上限。** 空间已经用尽，瓶颈在峰值时刻各类的同时占用量。看峰值时刻哪一类（恒定基线、层间激活缓存、瞬态、还是 allocator 空隙）的量级最大，针对最大的那类优化。开 recompute 才有意义，它压缩的层间激活缓存正是触及上限的主因。

**情况 B：`reserved` 已达上限，`allocated` 却低出好几 GB。** 可用字节并没有真的被填满，实际是 `reserved` 被虚高抬起。抬高 `reserved` 的三种成因都不靠多用显存：

- 瞬态峰值被抬进 `reserved` 后不再回落形成的那道台阶
- shape 变化造成的碎片
- 多 stream 分池后一条 stream 的空闲填不进另一条的请求

再开 recompute 去压缩 `allocated` 没有收益——压缩的那一维根本不是瓶颈。

> OOM 判断要同时看峰值占用分解（**时间维**：某一刻同时被持有多少）和 allocator 碎片状态（**空间维**：池能不能凑出这一刻要的连续大块）。低于上限的峰值遇到碎片化的池照样 OOM；等于上限的峰值若 shape 固定、池无碎片，反而能分配成功。

### 04.4 失效条件

一次 snapshot 只是某个 step、某张 rank 的快照。MoE 路由逐 step 波动，负载最重的那张 rank 也不固定在同一张卡上——这一回没碰到峰值，不能推导出下一个 step、另一张卡也安全。要做出判断，得在几个高负载 step、几张可疑 rank 上各采样分析一次，看差值是**稳定接近上限**还是**偶发接近上限**。

### 04.5 各分量能否静态估计

各分量字节量可以静态估到 ± 数 GB；真正无法静态预测的只有两样：

- 峰值落点 $t^*$（取决于运行时 stream 交错、各 kernel 时长、MoE 路由逐 step 波动）
- allocator 碎片余量

都只能事后从 snapshot 里分析出来。

---

## 05 优化手段：四类各自的取舍

### 05.1 恒定占用：分片、降精度、offload

V3 配置下，恒定占用这三块不分片的话，单 rank 要装：

$$
m_{const} = \underbrace{2N}_{\text{BF16 weights}} + \underbrace{2N}_{\text{BF16 gradients}} + \underbrace{(4+4)N}_{\text{Adam FP32 m/v}} \approx 218 \text{ GB}
$$

若加 FP32 master weight（4 bytes/param），完整是 16 bytes/param ≈ 291 GB。前面表里的 68.5 GB 是 Megatron 报告口径下按 V3 并行度切分后的 per-GPU 值。

#### 分片

| 工具 | 划分对象 | 关键代价 |
|---|---|---|
| FSDP / ZeRO-3 | 三类全分片（沿 DP 维） | 每层 all-gather + reduce-scatter；sharding 组越大通信越靠近关键路径 |
| ZeRO-1 / ZeRO-2 | 仅分片 opt state / +grad | 通信量低于 ZeRO-3，省得少 |
| Megatron TP | 按 head / hidden 维切 attention 和 MLP 权重 | 同时划分该层 activation；每层 4 次 all-reduce，**节点内 NVLink 限 8-way** |
| PP | 按层切分（每 rank 持 $L/p$ 层） | 跨 stage send/recv 通信；bubble 代价；**不动 activation 峰值** |
| EP | 按 expert 切分（每 rank 持 $1/e$ 份 expert） | 主要削减 MoE 参数大头；dense 部分不动 |

四种切法各对应一个维度，互不争用同一维度，可以相乘：

$$
\text{per-GPU 恒定占用} = \frac{\text{全量}}{\prod \text{各相关并行维}}
$$

#### 降精度

- **FP8 weight**：把 forward / backward 实际计算、并参与 DP all-gather 的那份工作权重从 BF16 降到 FP8，单 weights 一项再省 2×；FP32 master weight 仍保持高精度。改变的是 forward / backward 的计算精度与 all-gather 的通信精度，optimizer 更新仍在 FP32 计算。
- **FP8 optimizer state**：把 Adam 的 m / v 也降到 FP8。

**失效条件**：master weight、m / v 这类「极小增量反复累加进一个大基数」的量，仅靠 scaling 还不够。每步要加的更新值比基数小几个数量级，量化后落在基数尾数能分辨的台阶之下，加法被整个舍去，即 **swamping**（大基数把小增量淹没）。表现为参数实际不更新、loss 曲线无报错却不再下降。所以这类累加量必须保留一份高精度累加器。

#### 外移

- **Optimizer state offload → CPU**：通过 PCIe Gen5（≈ 64 GB/s 单向）移到 CPU 内存
- **Weights → NVMe**：再往后退一档到 NVMe（PCIe + filesystem，≈ 7 GB/s 单向）

从 NVMe 把 weights 拉回 GPU 计算一层比 HBM 直接读取慢两个数量级，MFU 会大幅下降，所以只在「单卡资源严重受限、必须容纳大模型」的场景才用。

> 降精度只在划分完之后单项仍占几 GB 以上时才划算。FSDP 划分到 256 rank 后 per-GPU optimizer state 已经只有 0.57 GB，再降到 FP8 也只节省约 0.4 GB，不值得为这点字节数增加复杂度。

### 05.2 Activation 缓存的压缩手段

Activation 缓存是峰值的大头，V3 配置下未压缩前每卡就有 131 GB（约 1.6× 一张 H100）。

#### Recompute（时间换空间）

| 策略 | 做法 | 代价 |
|---|---|---|
| **Full layer recompute** | 每层只 save 一个 layer-input `2BSD`，反向把整层 forward + backward 一并跑 | forward:backward ≈ 1:2 原本是 1+2=3 份算力，full recompute 变成 1+(1+2)=4 份，wall-time ≈ +33% |
| **Selective recompute** | 只对 attention / MLP / LayerNorm 某几种 op 重算 | 节省的缓存少、wall-time 也少；Megatron 最初做的是只重算 attention 的 `5BHS²` 项，但 FA 之后 `5BHS²` 已不在 HBM，意义变小 |
| **模块级混合** | 按 nn.Module 边界标记 checkpoint，不同 module 独立决定 | V3/V4 这种 MoE 主导的架构里常比 full layer 更省时间 |

#### Offload（搬到 CPU / NVMe）

- **Activation → CPU offload**：走 PCIe Gen5 ≈ 64 GB/s 单向
- **Activation → NVMe offload**：再慢 10× 以上，预训练中几乎不用

**关键**：必须把 PCIe DMA 和 forward / backward 计算 overlap：移下去要赶在下一层 forward 还没需要缓存之前完成，移回来要在反向使用缓存之前预取完。两个方向都得对准时序，任一边没与计算重叠 MFU 就下降——recompute 只在反向单点插入一段重算，offload 要双向与计算重叠，调度余量小得多。

#### 序列并行（长上下文）

按 sequence 维切到多 rank 上。序列越长、`S` 相关的那几项越主导，把 `S` 划分到多 rank 是唯一随长度同步缩小数据量的方式。

| 工具 | 划分对象 | 代价 |
|---|---|---|
| **Megatron SP** | 补充 TP，划分 TP 没覆盖的 `10sbh`（LayerNorm、dropout、block 输入激活） | 仅动 TP 没划到的部分 |
| **CP / Ring Attention** | 每卡 `S/N`，attention kernel 内数据量随 S/N 缩小 | 跨 rank 通信 |
| **DeepSpeed-Ulysses** | 两次 all-to-all 在 sequence ↔ head 间交换维度 | P 受 attention head 整除限制；GQA/MQA 下常受 KV head 数限制 |
| **USP（Ulysses + Ring-Attention）** | 同时叠加 | long-context 训练目前的常见组合 |

#### FP8 storage

把前向保存下来的中间值每元素的字节数从 16 bit 降到 8 bit。activation 是一次性写、一次性读的量，配 scaling 和反量化即可。

#### Fused kernel（中间量留 SRAM）

SRAM 和 HBM 的容量、带宽都差一个数量级，中间量每在 HBM 上往返一趟就消耗一份带宽。fuse 把数据量沿某个维度划分成小到能整个放进 SRAM 的 tile，逐 tile 算完只把最终结果写回 HBM，中间 tensor 始终留在片上、既不占 HBM 也不付往返带宽。**融合的边界由 SRAM 预算限定**。

判据不在中间量是否瞬时，而看 consumer 要的每个输出 tile，是不是只依赖产出它的那一个 CTA（一个 thread block）：

- 依赖都在本 CTA 就留片上：逐元素 pointwise，或归约轴整条留在 CTA 内 streaming
- 要跨 CTA 读取就必须物化——在 producer 已切散到多 CTA 的那根轴上做 reduction / transpose / shuffle

归约本身不必然导致物化：只要归约轴留在 CTA 内 streaming（FA 的 KV、norm 的 $D$，running state 小到进寄存器），归约也能片上；物化只在归约轴被 tile 到多个 CTA 时才出现。

代表实例：

- **FlashAttention 系**：把 `QK^T`、softmax 输出 `P`、`dS`、`dP` 全部留在 SRAM tile 内。HBM 上只剩 `O ∈ [B,S,D]` 和 LSE `∈ [B,H,S]`。参考参数 ($B=4, H=32, S=8192, D=4096$, BF16) 下，原本几块约 40 GiB，FA 后只剩 0.254 GiB
- **fused LayerNorm / RMSNorm + residual**：归约轴 $D$ 整条留在 CTA 内
- **fused MoE token permute**：路由后的 token 重排融进 grouped-GEMM 的数据准备路径（Megatron 开关 `moe_permute_fusion`）
- **fused cross-entropy**（长 vocab）：$V \ge 128k$ 时把 logits 和 NLL loss 合并到一个 kernel 内，按 row 流式计算

### 05.3 计算峰值瞬态的压缩

瞬态是 kernel 执行中临时占用的 HBM，规模 = f(B, 每卡 token 数, dtype 位宽)。recompute 和 offload 对瞬态均无效：瞬态是 kernel 内 μs 级的瞬时量，recompute 针对的是跨 kernel 存下来给反向用的 saved activation，削减它反而新开一个重算窗口、又产生一份瞬态；offload 走 PCIe，搬运延迟比 μs 慢几个数量级，根本来不及在 kernel 算完前把数据量移出。

走 fused kernel / tiling 这条路：不去缩小数据量的尺寸，而是让中间量根本不写进 HBM。

**fused MoE expert FFN** 是个典型反例：把 expert 内部的 up_proj + gate_proj + 激活（SwiGLU）+ down_proj 几步在 expert 内 fuse，能不能留片上取决于 kernel 怎么 tiling，而不取决于「融没融」。标准实现（Megatron-Core grouped GEMM、V3 同理）下这份 `[C_local, 2n]` 实际会落 HBM：FC1 把输出按 N 维切散到多个 CTA，FC2 又要沿同一维（FC1 的输出维 F）跨 CTA 归约。**这正是「跨 CTA 读取就得物化」那种情形，只能过 HBM/L2**。即便把 FC1 + 激活 + FC2 强融进一个 megakernel 也没用——一个 kernel 不等于一个 CTA，各 stage 间仍要在 global memory 上会合。

要真留片上，得换成每个 CTA 独占 token 行块、自己做完整 $F$ 归约的 FA 式 tiling，代价是降低 GEMM 并行度 / 吞吐。V3 训练路径同理走 grouped GEMM、这份激活照样落 HBM：按 V3 配置 ($B=1, S=4096$) 静态估算，$[C_local, 2n]$ 单层约 0.3 GiB / 卡，不是单层峰值主导项；主峰是随路由数 K 放大的进出口张量（$[C_local, D]$ 量级、随 K 累加）。

cuBLAS workspace shrink：`CUBLAS_WORKSPACE_CONFIG=:4096:8` 固定 workspace 上限（多 stream 共享 handle 时换取确定性），可省下几十个 handle 累积的几 GiB；代价是某些 SplitK / 多阶段 GEMM 算法用不了，吞吐可能略降。

### 05.4 通信 buffer：减少 staging 份数与尺寸

NCCL 在每个 GPU 上为每条通信 stream 注册一段 staging buffer，尺寸按最坏情况一次预分配。V3 配置下每卡每方向 EP all-to-all staging 约百 MiB 级（某配置下估算约 230 MiB）。

#### 调度（bucket 聚合）

把同一层多个参数的 gradient 合并成一次 collective。bucket 小则小消息多、staging 总量大、带宽利用率低；bucket 大则单次 staging 大、还要等更多 grad 准备好才发，反向关键路径等待时间长。

| 工具 | 默认 bucket | 说明 |
|---|---|---|
| DDP `bucket_cap_mb` | 25 MiB | 经典 PyTorch 默认 |
| FSDP | 无 MiB 级阈值 | 合并粒度由 `fully_shard` 包裹的 module 单元决定 |
| DeepSpeed ZeRO `reduce_bucket_size` | 5e8 元素 ≈ 1 GB | DeepSpeed 风格 |

#### 配置（NCCL 环境变量）

- `NCCL_BUFFSIZE` × channel 数 = 每个 rank 预留的总 staging（Simple 默认 4 MiB/channel）
- `NCCL_NTHREADS`：每 channel 线程数，决定 staging 复用速度
- `NCCL_ALGO` / `NCCL_PROTO`：选 ring / tree / collnet / nvls 算法与协议（LL 有效带宽 50% / LL128 95% / Simple 最高延迟也最高）

#### 算法 / 拓扑

| 算法 | 特点 | 硬件前提 |
|---|---|---|
| Ring | 每跳一份 staging，N 卡 ring 一共 N 份；带宽最优，latency 随 N 线性涨 | 通用 |
| Tree | 二叉树聚合，latency log(N) | 通用 |
| Collnet | 借交换机 / fabric 做 in-network aggregation | SHARP 硬件 |
| NVLS | 走 NVLink SHARP，把 reduction offload 到 NVSwitch 域；几乎不在 GPU 显存里留 staging | NVSwitch + NVLink SHARP |

#### 复用（对称内存直接写远端）

- **NVSHMEM**：rank A 可直接 `nvshmem_put` 把数据写到 rank B 显存上，省去本地 + 远端 staging
- **DeepEP**：早期 / legacy 版本用 NVSHMEM 省去 EP all-to-all 两侧的中转拷贝（某配置下每卡每方向 ≈ 230 MiB）

### 05.5 对照表

「优先级」一列表示当前配置下的收益/代价排序：先动富余资源，后动紧缺资源。它会随配置变：通信已是反向关键路径瓶颈时再做恒定占用的拆分反而不划算。

| 显存类型 | 典型手段顺序 | 取舍逻辑 |
|---|---|---|
| 恒定占用 | FSDP / ZeRO > FP8 master + opt state > offload | 先按 rank 摊薄常驻状态；分片收益耗尽后，再考虑降精度或 offload |
| Activation 激活缓存 | recompute > FP8 storage / offload（视代价）> 序列并行类（长上下文）> fused kernel | 通常先用 recompute；FP8 storage / offload 取决于数值风险和带宽；CP / Ulysses 主要改善负载均衡 |
| 计算峰值 瞬态 | fused kernel（FA / MoE FFN）> kernel 内 tiling | 中间量留 SRAM，省 HBM 往返 |
| 通信 buffer | bucket 调度 > algorithm 选择 > NVSHMEM | bucket 调度通常先改；算法/拓扑依赖硬件；NVSHMEM 用有界对称内存替换随跳数增长的 staging |

![四类显存与工具的映射](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020260707110818-3.jpg)

### 05.6 各手段的失效条件

- **FSDP / ZeRO**：sharding 组扩大到 all-gather 慢于反向计算时就不再划算——通信从「藏在计算后面」变成关键路径瓶颈
- **FP8 storage**：失效在反复累加那一类量（swamping），只有一次性读写的激活才换得安全
- **Offload**：搬运时间藏不进计算就反过来拖慢，PCIe 来回的延迟进入关键路径
- **fused kernel**：标准 grouped-GEMM 实现下反向会失效——反向要把它整张 fan-out 给多个梯度 GEMM，forward 那种「逐 tile 流过、从不整张落地」的办法用不上
- **NVSHMEM**：用单边写替换掉随跳数增长的 staging，但同步和完成语义要自己管

---

## 06 CP/SP/PP/TP 各分片哪一段显存

### 06.1 想降低恒定 68.5 GB：用 FSDP / ZeRO

每加一份 DP 维就 $/N$。分到极致的代价是每步都要 all-gather + reduce-scatter，再往下分省下的显存会被新增的通信开销抵消——这时要退回到 HSDP（hybrid sharding，组内分片、组间用 DDP）或限制 sharding 组大小。

PP（按层划分）、EP（按 expert 划分）同样把恒定占用降到这条基线，但它们**对激活缓存帮助有限**。

### 06.2 层间激活缓存靠什么分片

设每 step 的 Global Token Size = $G$，切 sequence 后每个 rank 持有 $G/N$ 个 token，attention 数据量总和没变。

固定 Global Batch Size 时，划分序列会让 Local Batch Size 上涨，省下的又被原样涨回：总卡数固定，新拿去划分 sequence 的卡只能从 DP 维分出，DP 度一降，$global\_batch = DP \times local\_batch$ 里每个 DP rank 摊到的 local batch 就成倍上涨。这种抵消对所有 activation 项都成立：

$$
\text{CP 沿 S 切每项} \times \frac{1}{N}, \quad \text{batch 涨回} \times N \;\Rightarrow\; \text{单卡峰值净变化趋近零}
$$

抵消假设涨上来的 local batch 由并发 micro-batch 承担——峰值激活只由同时在飞的 micro-batch 数决定。若改用梯度累积让多出来的 batch 串行跑，峰值激活并不回弹，CP 切出的 $S/N$ 不被抵消、仍净省。

例外条件：当训练目标是训更长 context、要容纳单卡放不下的 sequence 时，CP 加卡用来容纳更长 $S$、不从 DP 分卡，local batch 不回弹——此时相对「不切 sequence 直接训练同样长度」，划分 sequence 才真正压低单卡 activation。

### 06.3 反向当前层的临时量与 kernel 数据量：TP 或 fused kernel

峰值窗口里的动态量分两种：

- **Chain rule 决定的反向临时量**：重算出的 forward intermediate、与它同形状的中间量梯度（MLP/MoE 这档宽 $2n$），以及 dY、dX、dQ、dK、dV 这类宽 D 的边界梯度——这类不能删，只能用 TP 减少每卡持有的 head / hidden 切片（attention / MLP 数据量同样划分 $/t$），代价是每层 4 次 all-reduce、对 NVLink 带宽敏感
- **真正 μs 级的 kernel workspace**：靠 fused kernel 或 workspace 配置缩小

### 06.4 PP 对激活缓存帮助有限

PP 大小 $p$ 在激活缓存公式里被直接约掉：1F1B 第一阶段同时在飞 $m_{inflight} = p$ 个 micro-batch（$m \ge p$ 时）。原因：1F1B 启动时先逐级跑完 $p$ 个 micro-batch 的前向、把 $p$ 级流水线填满，才轮到第一个 micro-batch 的反向，同时驻留 $p$ 份激活缓存，正好与 PP 度 $p$ 约掉。

| Schedule | multiplier | 说明 |
|---|---|---|
| 1F1B | $1\times$ | 经典做法 |
| VPP（virtual pipeline, $m_v$ chunks） | $1 + \frac{p-1}{p \cdot m_v}$ | $p=16, m_v=4$ 时 ≈ 1.23× |
| DualPipe | $1 + \frac{1}{PP}$ | DeepSeek V3 双向流水线；另需双份模型参数副本（独立代价，V3 靠大 EP 摊薄） |

Korthikanti et al. 2022 (arXiv:2205.05198 §4.2.3) 给出的最优 schedule，bubble 最小化的代价是**内存压力集中在 pipeline 第一个 stage**。

> Schedule multiplier 讨论只在不开 recompute 时才显著：那时激活缓存是整层中间结果，V3 未压缩前每卡 131 GB，$1\times$ 与 $1.25\times$ 能差出三十多 GB。开启 Full Recompute 后，每层只 save 一个 layer-input $2BSD$，激活缓存缩到 3.5 GiB（V3 量级），multiplier 差异不到 1 GiB。

### 06.5 分片维度对照

| 工具 | 划分对象 | 量级效果 |
|---|---|---|
| Activation Checkpointing | 层间激活缓存 | 缩减到只剩 layer-input $2BSD$ |
| Activation Offload | 层间激活缓存（移到 CPU） | 受 PCIe 带宽限制，覆盖不了 μs 级 kernel workspace |
| FSDP / ZeRO | 恒定占用 | 按 DP 度数 N 划分，每卡 /N |
| PP（pipeline） | 恒定占用（按层 $L/p$ 摊） | 每 stage 参数 /p；不划分激活缓存（一卡放 $L/p$ 层、激活缓存 $\propto L/p \times m_{inflight}$，$p$ 在两头约掉） |
| EP（expert parallel） | 恒定占用里的 expert 权重 | 每卡 expert /e；划分的是 MoE 参数大头，dense 部分不动 |
| TP（Megatron） | 恒定 + 激活缓存 + kernel 数据量（按 head / hidden 维划分） | attention 数据量 /t、MLP 中间 /t；受 NVLink 带宽限制，最多 8-way |
| Megatron SP | TP 未覆盖的 sequence 维 | $10sbh$（每 rank 各存一份）/t；不动 attention |
| CP / Ring Attention | 全局 sequence 维 | 每卡 $S/N$；Global Token 固定时作用是负载均衡 |
| DeepSpeed-Ulysses | sequence + head 两次 all-to-all | 同 CP，但 P 受 attention head 整除限制（GQA/MQA 下常受限于 KV head 数，可经 KV replication 放宽） |
| 小 micro-batch $B$ | $B$ | 瞬态量随 B 降低；代价是单 kernel 数据量变小、算力利用率下降，且 PP 下 m 会变大（激活缓存总量不变） |
| FlashAttention | attention 的 $S^2$ 项 | 移到 SRAM，HBM 只剩 $2BSD$（O）+ $4BHS$（LSE，FP32） |

---

## 07 FA 之后 HBM 上的残留

今天主流训练框架默认用 FlashAttention，标准 attention 那个 $[B, H, S, S]$ 不再 materialize 到 HBM。

峰值主要来自 FA 之后还留在 HBM 上的：MoE 层的 dispatch / permute / expert FFN 中间值，以及 Full Recompute 重算窗口里的中间值。

FA 那段的参考参数 ($B=4, H=32, S=8192, D=4096$, BF16) 是单 EP rank、单层、单 micro-batch 的尺度——真实训练里还要再乘上 PP、EP 和 micro-batch 数。

未启用 fused expert kernel 的配置下，MoE FFN intermediate $[C_local, 2n]$ 会成为主导项之一。启用 fused kernel（把 SwiGLU 融进 FC1 epilogue）后，$2n$ 宽的 pre-activation $H$ 在 forward 不再单独落 HBM——逐 tile 在 epilogue 里算完即被消费。但这条收益对反向容易失效：反向算 expert FFN 梯度时，forward intermediate $[C_local, 2n]$ **必须在场**——无论是 save-for-backward 存下来的，还是 full recompute 现算出来的——它本来就得写出去。此刻还要同时生成它同形状的梯度 $dH$（同样 $[C_local, 2n]$）。forward 不必把它整张落地（随算随被下一段消费），backward 却要整张物化在 HBM 上 fan-out 给多个梯度 GEMM——失效的根因是这道刚性物化边界，不是「tile 超过 SRAM 容量」的问题。

---

## 08 Full Recompute 之后峰值仍达 85 GB：差值来自哪几项

上面 MemTrace 占用分解那 93.5 GB，是部分 recompute（层间激活缓存还残留 12 GB）时截下来的快照；这里改为分析 full recompute 把层间激活缓存降到下界后，峰值还剩多少、能不能回到 80 GB 以内。三个配置档的峰值依次递减：

- 未优化 199.5 GB
- 部分 recompute 削到 93.5 GB
- full recompute 继续降低

按 per-GPU 视角逐项算（EP 路由波动下最重 rank 只会更吃紧），V3 配置（$PP=16, L=61, S=4096, B=1, D=7168$, BF16）下六类分量：

| 分量 | 大小 | 说明 |
|---|---|---|
| 恒定占用 | 68.5 GB | 已切分后的 per-GPU 数 |
| 通信 buffer | 3 GB | EP all-to-all + DP allreduce + PP send/recv |
| 层间激活缓存 | 3.5 GiB | full recompute 后只留 layer-input $2BSD$；1F1B 下总激活缓存与 $p$ 无关 |
| 当前层重算 forward intermediate | ≈ 2 GB | recompute 换来的是时间、不是让这部分消失——重算的那一瞬间该层所有中间产物都保留在 HBM 上；MoE 是大头（进出口张量 $[C_local, D]$ 随 K 累加） |
| 当前层 grad buffer | ≈ 3 GB | dY + dX + 当前层 dW（用完即并入梯度存储） |
| allocator overhead | ≈ 5 GB | `reserved − allocated`，V3 预训练 shape 稳定，碎片率低 |

合计：

$$
68.5 + 3 + 3.76 + 2 + 3 + 5 \approx 85 \text{ GB}
$$

这一组分量可代入估算：

$$
\text{恒定基线} \approx \frac{218 \text{ GB}}{\prod \text{FSDP} \times \text{TP} \times \text{PP} \times \text{EP}}
$$

$$
\text{层间激活缓存} \approx 2BSD \cdot \frac{L}{p} \cdot \text{multiplier} \cdot \text{recompute\_factor}
$$

其中 `multiplier` 是 PP 调度系数（1F1B ≈ 1×、VPP ≈ 1.23×），`recompute_factor` 是该粒度下每层的保留比例（full recompute 取 1，不开则是整层中间结果的倍数）。

若峰值超出容量 5 GB，同一时刻占用最大的是恒定那 68.5 GB，该动的就是它：进一步分片（FSDP / TP / PP / EP）或压精度（FP8 master weight / optimizer state）。层间激活缓存只剩 3.5 GiB，再加大重算收益已经很小。

---

## 09 Hopper → Blackwell：PP 依赖随硬件扩容下降

之前讨论的前提是单卡 HBM 容量与 NVLink 域大小固定不变；这两项参数一旦变了，最先受影响的就是 PP。

PP 主要解决两件事：

1. 单卡 HBM 容纳不下整个模型，只能按层把恒定占用分摊到多卡
2. 模型必须跨节点划分时，PP 的跨 stage 通信量比 TP/EP 小得多，能用较低带宽的连接承载——代价是流水线 bubble

单卡 HBM 从 H100 80 GB → H200 141 GB → B200（HGX B200 常见可用 180 GB HBM3e，物理口径 192 GB），一张卡能容纳的层数随之上升，容量约束首先放宽。更关键的是 NVLink 域的扩张：Hopper 一域 8 卡，到 NVL72 一域 72 卡。原来必须跨节点、只能交给 PP 走低带宽连接的那部分 TP/EP，现在能整块放进同一个高带宽 NVLink 域里，低带宽连接的约束也随之消失。这两条约束都放宽后，PP 那份 bubble 代价就不再划算，对 PP 的依赖会逐步下降。

---

## 10 一些个人化整理

把全文拆成三句话总结：

1. **OOM 看的是「峰值时刻各类同时占用」**，不是某一类的最大值；reserved 容易被瞬态峰值抬成台阶只升难降。
2. **降显存最划算的顺序**：先用 FSDP / ZeRO 把恒定占用摊薄；再用 recompute / fused kernel 砍 activation 峰值；最后才考虑 FP8 / offload 这些有数值或带宽代价的手段。
3. **2026 年的硬件趋势**让 PP 的必要性下降：HBM 增大 + NVLink 域扩张，把跨节点划分的硬约束松开了；未来大概率是 TP / EP / CP 在 NVLink 域内整块摊开、PP 留给极少数真正跨域的场景。

跟 [[UltraScale Playbook 笔记]] 互为补充：那本讲「怎么选并行度」，这篇讲「选完之后峰值落在哪、会不会 OOM」。两份笔记里 FSDP / ZeRO / TP 的描述口径一致，差异主要在本文把 MoE + Recompute 的反向窗口单独拎出来画了峰值时刻分解。

---

## 参考

- 原文：[DeepSeekV4 Infra Deep Dive · 第四节 4.4: PyTorch 训练显存管理](https://zhuanlan.zhihu.com/p/2052200237373118275)
- 同系列：
  - [第一节: infra 的设计原则](https://zhuanlan.zhihu.com/p/2031523642467689507)
  - [第二节: Hybrid Attention - 问题演进视角](https://zhuanlan.zhihu.com/p/2032298882625976192)
  - [第三节：在缓存命中率就是一切的世界里，PagedAttention 失效后会怎样？](https://zhuanlan.zhihu.com/p/2034345100680114900)
  - [4.1: Deepseek V4 为什么不能再按 module 做 checkpoint？](https://zhuanlan.zhihu.com/p/2036897911921964012)
  - [4.2: Activation Checkpointing 的四档粒度](https://zhuanlan.zhihu.com/p/2036898381826675516)
  - [4.3: Tensor-level AC 的三个工程问题：joint graph、最小子图、storage reuse](https://zhuanlan.zhihu.com/p/2037700892774888414)
- 配套阅读：
  - [[UltraScale Playbook 笔记]]
  - Korthikanti et al. 2022 (arXiv:2205.05198) — Reducing Activation Recomputation in Large Transformer Models
  - Liu et al. 2023 (arXiv:2310.01889) — Ring Attention with Blockwise Transformers
  - Jacobs et al. 2023 (arXiv:2309.14509) — DeepSpeed-Ulysses
  - Fang et al. 2024 (arXiv:2405.07719) — USP: A Unified Sequence Parallelism Approach
