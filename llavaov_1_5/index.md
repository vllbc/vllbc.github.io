# llavaov_1_5

# LLaVA-OneVision-1.5 模型与Megatron训练框架详细文档

## 目录
1. [项目概述](#1-项目概述)
2. [架构设计](#2-架构设计)
3. [模型实现](#3-模型实现)
4. [训练框架](#4-训练框架)
5. [数据处理](#5-数据处理)
6. [训练流程](#6-训练流程)
7. [模型转换](#7-模型转换)
8. [配置与优化](#8-配置与优化)

---

## 1. 项目概述

### 1.1 项目介绍
LLaVA-OneVision-1.5 是一个完全开源的大规模多模态模型(LMM)训练框架，具有以下特点：
- **原生分辨率图像处理**：支持高分辨率图像输入
- **SOTA性能**：在多个多模态基准测试上达到业界领先水平
- **高效训练**：基于Megatron-LM优化，全流程训练成本约$16,000
- **完全开源**：提供完整的训练框架、配置、数据集和训练日志

### 1.2 核心特性
- 支持MoE（Mixture of Experts）
- FP8精度训练
- 长序列并行
- 原生分辨率图像输入（最大1344x1344）
- 端到端优化的训练流水线

### 1.3 技术栈
- **基础框架**：Megatron-LM (NVIDIA)
- **语言模型**：Qwen系列（Qwen2.5/Qwen3）
- **视觉模型**：Rice-ViT（基于ViT优化）
- **训练框架**：AIAK-Training-LLM（百度AIAK团队）

---

## 2. 架构设计

### 2.1 整体架构

项目采用模块化设计，主要包含以下几个核心模块：

```
LLaVA-OneVision-1.5/
├── aiak_megatron/              # Megatron-LM核心代码
│   └── megatron/
│       ├── core/               # 核心功能模块
│       │   ├── models/         # 模型定义
│       │   ├── transformer/    # Transformer实现
│       │   ├── optimizer/      # 优化器
│       │   └── pipeline_parallel/  # 流水线并行
│       └── training/           # 训练工具
├── aiak_training_llm/          # AIAK训练框架
│   ├── models/                 # 模型实现
│   │   ├── llavaov_1_5/       # LLaVA-OV-1.5模型
│   │   ├── qwen/              # Qwen语言模型
│   │   └── qwen_vl/           # Qwen-VL模型
│   ├── data/                   # 数据处理
│   ├── train/                  # 训练逻辑
│   │   ├── pretrain/          # 预训练
│   │   └── sft/               # 监督微调
│   └── tokenizer/             # 分词器
└── tools/                      # 工具集
    └── convert_checkpoint/     # 模型转换工具
```

### 2.2 模型工厂模式

项目使用工厂模式管理多个模型，核心文件：`aiak_training_llm/models/factory.py`

#### 2.2.1 注册机制

```python
# 全局注册表
MODEL_ARCH_CONFIGS = {}          # 模型架构配置
MODEL_ARCH_TO_FAMILY = {}        # 架构到模型族映射
MODEL_FAMILY_TO_ARCHS = {}       # 模型族到架构映射
MODEL_FAMILY_TO_PROVIDER = {}    # 模型族到提供者映射

# 注册模型配置
@register_model_config(model_family="llava-ov-1.5", model_arch="llava-ov-1.5-4b")
def llava_one_vision_1_5_4b():
    return LlavaOnevision1_5Config(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        # ... 更多配置
    )

# 注册模型提供者
@register_model_provider(model_family=[VisionLanguageModelFamilies.LLAVA_OV_1_5])
def rice_vl_model_provider(pre_process=True, post_process=True):
    # 模型构建逻辑
    pass
```

### 2.3 训练器工厂模式

训练器也采用工厂模式注册：`aiak_training_llm/train/trainer_builder.py`

```python
@register_model_trainer(
    model_family=[constants.VisionLanguageModelFamilies.LLAVA_OV_1_5],
    training_phase=constants.TrainingPhase.PRETRAIN
)
def default_pretrain_trainer(train_args):
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        model_type=model_type,
        forward_step_func=forward_step,
    )
    return trainer
```

---

## 3. 模型实现

### 3.1 LlavaOnevision1_5 模型

核心文件：`aiak_training_llm/models/llavaov_1_5/llavaov_1_5_model.py`

#### 3.1.1 模型组成

LlavaOnevision1_5由三个主要组件构成：

```python
class LlavaOnevision1_5(MegatronModule):
    def __init__(self, ...):
        # 1. 视觉编码器（Vision Encoder）
        if self.add_encoder:
            self.vision_model = RiceViTModel(
                vision_config,
                vision_layer_spec,
            )
            
        # 2. 适配器（Adapter）- 连接视觉和语言模态
            self.adapter = Adapter(
                adapter_config,
                adapter_layer_spec,
                input_size=vision_config.hidden_size,
                output_size=language_config.hidden_size,
            )
        
        # 3. 语言模型（Language Model）
        if self.add_decoder:
            self.language_model = QwenModel(
                config=language_config,
                transformer_layer_spec=language_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                # ... 更多参数
            )
```

#### 3.1.2 前向传播流程

```python
def forward(
    self,
    images: torch.Tensor,              # 图像输入
    image_grid_thw: torch.Tensor,      # 图像网格维度（时间、高、宽）
    input_ids: torch.Tensor,           # 文本token ID
    position_ids: torch.Tensor,        # 位置ID
    attention_mask: torch.Tensor,      # 注意力掩码
    labels: torch.Tensor = None,       # 标签（训练时使用）
    packed_seq_params: PackedSeqParams = None,  # 序列打包参数
    inference_params: InferenceParams = None,    # 推理参数
):
    # 步骤1：视觉编码
    if self.add_encoder and images is not None:
        # 通过视觉模型提取图像特征
        image_embeddings, window_index = self.vision_model(
            images, 
            grid_thw=image_grid_thw
        )
        # 通过适配器映射到语言模型维度
        image_embeddings = self.adapter(image_embeddings, window_index)
    
    # 步骤2：文本编码
    if self.pre_process:
        language_embeddings = self.language_model.embedding(
            input_ids=input_ids, 
            position_ids=None
        )
        
        # 步骤3：融合视觉和语言嵌入
        if images is not None:
            # 找到图像token位置并替换为视觉嵌入
            images_mask = (input_ids == self.config.image_token_id)
            combined_embeddings = language_embeddings.masked_scatter(
                images_mask, 
                image_embeddings
            )
    
    # 步骤4：通过语言模型生成输出
    output = self.language_model(
        input_ids=None,
        decoder_input=combined_embeddings,
        attention_mask=attention_mask,
        labels=labels,
        # ... 其他参数
    )
    
    return output
```

### 3.2 视觉模型：RiceViTModel

核心文件：`aiak_training_llm/models/llavaov_1_5/rice_vision_model.py`

#### 3.2.1 关键组件

```python
class RiceViTModel(VisionModel):
    def __init__(self, config, transformer_layer_spec, spatial_merge_size=2):
        # 1. Patch嵌入层
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,  # 14
            in_channels=config.in_channels,  # 3
            embed_dim=config.hidden_size,    # 1024
        )
        
        # 2. 旋转位置编码
        self.rotary_pos_emb = VisionRotaryEmbedding(
            dim=config.kv_channels // 2
        )
        
        # 3. Transformer块
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
        )
        
        # 4. 类别token（用于全局特征）
        self.class_embedding = torch.randn(config.hidden_size)
        self.class_pos_emb = torch.randn(1, config.kv_channels // 2)
        
        # 5. 预层归一化
        self.pre_layernorm = torch.nn.LayerNorm(
            config.hidden_size, 
            eps=1e-4
        )
```

#### 3.2.2 位置编码策略

Rice-ViT使用特殊的2D旋转位置编码：

```python
def rot_pos_emb(self, grid_thw):
    """
    为每个图像patch生成2D旋转位置编码
    grid_thw: [num_images, 3] - (时间, 高度, 宽度)
    """
    pos_ids = []
    for t, h, w in grid_thw:
        # 生成高度位置ID
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
        
        # 生成宽度位置ID
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        # ... 类似处理 ...
        
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))
    
    # 应用旋转位置编码
    rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    
    return rotary_pos_emb
```

### 3.3 语言模型：QwenModel

基于Qwen系列架构，支持：
- **GQA (Grouped Query Attention)**：减少KV缓存内存
- **RoPE位置编码**：更好的长度外推能力
- **SwiGLU激活**：更好的性能
- **RMSNorm**：更稳定的训练

### 3.4 适配器（Adapter）

核心文件：`aiak_training_llm/models/qwen_vl/adapter.py`

适配器负责将视觉特征映射到语言模型空间：

```python
class Adapter(MegatronModule):
    """
    2层MLP适配器
    Input: [seq_len, hidden_size_vision]
    Output: [seq_len, hidden_size_language]
    """
    def __init__(self, config, layer_spec, input_size, output_size):
        # 第一层：投影到中间维度
        self.linear_fc1 = build_module(
            layer_spec.linear_fc1,
            input_size,
            config.ffn_hidden_size,
        )
        
        # 激活函数（通常是GELU）
        self.activation_func = config.activation_func
        
        # 第二层：投影到输出维度
        self.linear_fc2 = build_module(
            layer_spec.linear_fc2,
            config.ffn_hidden_size,
            output_size,
        )
```

### 3.5 模型配置

#### 3.5.1 支持的模型规格

| 模型 | 层数 | 隐藏层维度 | FFN维度 | 注意力头数 | 参数量 |
|------|------|-----------|---------|-----------|--------|
| llava-ov-1.5-3b | 36 | 2048 | 11008 | 16 | ~3B |
| llava-ov-1.5-4b | 36 | 2560 | 9728 | 32 | ~4B |
| llava-ov-1.5-8b | 36 | 4096 | 12288 | 32 | ~8B |
| llava-ov-1.5-14b | 40 | 5120 | 17408 | 40 | ~14B |
| llava-ov-1.5-32b | 64 | 5120 | 25600 | 64 | ~32B |
| llava-ov-1.5-30b-a3b | 48 | 2048 | 6144 | 32 | ~30B (MoE) |

#### 3.5.2 视觉模型配置

```python
VisionConfig(
    num_layers=24,                    # Transformer层数
    hidden_size=1024,                 # 隐藏层维度
    ffn_hidden_size=4096,            # FFN维度
    num_attention_heads=16,          # 注意力头数
    patch_size=14,                   # Patch大小
    image_size=(1344, 1344),         # 最大图像尺寸
    kv_channels=64,                  # KV通道数
    normalization="LayerNorm",       # 归一化类型
    position_embedding_type="rope"   # 位置编码类型
)
```

---

## 4. 训练框架

### 4.1 训练器架构

#### 4.1.1 MegatronTrainer

核心文件：`aiak_training_llm/train/megatron_trainer.py`

```python
class MegatronTrainer(object):
    """Megatron训练器基类"""
    def __init__(
        self,
        train_args,                              # 训练参数
        train_valid_test_dataset_provider,       # 数据集提供者
        model_provider,                          # 模型提供者
        model_type,                              # 模型类型
        forward_step_func,                       # 前向传播函数
        process_non_loss_data_func=None,         # 非损失数据处理函数
        get_embedding_ranks=None,                # 获取嵌入层所在rank
        get_position_embedding_ranks=None,       # 获取位置嵌入所在rank
    ):
        self.train_args = train_args
        self.train_valid_test_datasets_provider = train_valid_test_dataset_provider
        self.model_provider = model_provider
        self.model_type = model_type
        self.forward_step_func = forward_step_func
        # ...
    
    def train(self):
        """启动训练"""
        pretrain(
            train_args=self.train_args,
            train_valid_test_dataset_provider=self.train_valid_test_datasets_provider,
            model_provider=self.model_provider,
            model_type=self.model_type,
            forward_step_func=self.forward_step_func,
            # ...
        )
```

### 4.2 训练主流程

核心文件：`aiak_training_llm/train/training_utils.py`

#### 4.2.1 预训练主函数

```python
def pretrain(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
):
    """
    主训练流程：
    1. 初始化Megatron
    2. 构建模型、优化器和学习率调度器
    3. 获取数据迭代器
    4. 执行训练循环
    """
    
    # 步骤1：初始化
    initialize_aiak_megatron(
        args=train_args,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )
    
    # 步骤2：构建模型和优化器
    model, ema, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, 
        model_type, 
        checkpointing_context=checkpointing_context
    )
    
    # 步骤3：构建数据迭代器
    train_data_iterator, valid_data_iterator, test_data_iterator = \
        build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider
        )
    
    # 步骤4：训练循环
    if args.do_train and args.train_iters > 0:
        iteration, num_floating_point_operations_so_far = train(
            forward_step_func=forward_step_func,
            model=model,
            ema=ema,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
            # ...
        )
```

#### 4.2.2 训练循环

```python
def train(forward_step_func, model, ema, optimizer, ...):
    """主训练循环"""
    
    while iteration < args.train_iters:
        # 1. 执行训练步骤
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, \
        grad_norm, num_zeros_in_grad = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config
        )
        
        # 2. 更新EMA模型（如果启用）
        if args.enable_ema and ema is not None:
            for e, m in zip(ema, model):
                update_ema(e, m, rate=args.ema_decay)
        
        # 3. 日志记录
        report_memory_flag = training_log(
            loss_dict, 
            total_loss_dict,
            learning_rate,
            iteration, 
            loss_scale,
            # ...
        )
        
        # 4. 验证评估（按间隔）
        if args.eval_interval and iteration % args.eval_interval == 0:
            evaluate_and_print_results(
                prefix, 
                forward_step_func,
                valid_data_iterator, 
                model,
                iteration, 
                # ...
            )
        
        # 5. 保存检查点
        should_exit = checkpoint_and_decide_exit(
            model, 
            optimizer, 
            opt_param_scheduler, 
            iteration,
            num_floating_point_operations_so_far,
            # ...
        )
        
        iteration += 1
```

### 4.3 训练步骤详解

```python
def train_step(forward_step_func, data_iterator, model, optimizer, ...):
    """单个训练步骤"""
    
    # 1. 清空梯度
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    
    # 2. 前向和反向传播
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False
    )
    
    # 3. 更新参数
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    
    # 4. 更新学习率
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1
    
    # 5. 计算平均损失
    loss_reduced = {}
    for key in losses_reduced[0].keys():
        numerator = sum(x[key][0] for x in losses_reduced)
        denominator = sum(x[key][1] for x in losses_reduced)
        loss_reduced[key] = numerator / denominator
    
    return loss_reduced, skipped_iter, ...
```

### 4.4 前向传播函数

#### 4.4.1 预训练阶段

核心文件：`aiak_training_llm/train/pretrain/pretrain_llavaov_1_5.py`

```python
def forward_step(data_iterator, model):
    """预训练前向传播步骤"""
    
    # 1. 获取批次数据
    images, image_grid_thw, pixel_values_videos, video_grid_thw, \
    input_ids, position_ids, attention_mask, \
    labels, loss_mask, attn_mask_type, packed_seq_params = get_batch(data_iterator)
    
    # 2. 模型前向传播
    output_tensor = model(
        images,                    # 图像输入
        image_grid_thw,           # 图像网格
        input_ids,                # 文本token
        position_ids,             # 位置ID
        attention_mask,           # 注意力掩码
        attn_mask_type,           # 掩码类型
        labels,                   # 标签
        packed_seq_params,        # 打包序列参数
        pixel_values_videos=pixel_values_videos,  # 视频输入
        video_grid_thw=video_grid_thw             # 视频网格
    )
    
    # 3. 返回输出和损失函数
    return output_tensor, partial(loss_func, loss_mask)
```

#### 4.4.2 损失函数

```python
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """
    损失函数计算
    
    Args:
        loss_mask: 损失掩码，标记哪些位置需要计算损失
        output_tensor: 模型输出的损失张量
    
    Returns:
        loss: 标量损失值
        local_num_tokens: 本地token数量
        loss_reduced_dict: 包含各种统计信息的字典
    """
    
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    
    # 计算加权损失
    total_tokens = loss_mask.sum()
    loss = torch.cat([
        torch.sum(losses.view(-1) * loss_mask).view(1), 
        total_tokens.view(1)
    ])
    
    # Context Parallel归约
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
    
    # NaN检查
    if args.check_for_nan_in_loss_and_grad:
        assert not loss[0].isnan(), f'Found NaN in loss on rank {rank}'
    
    # 跨数据并行组归约
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    
    loss_reduced_dict = {
        'lm loss': (reporting_loss[0], reporting_loss[1])
    }
    
    # 可变序列长度处理
    if args.variable_seq_lengths:
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int)
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens.item()
    
    return (
        loss[0] * args.context_parallel_size,
        loss[1].to(torch.int),
        loss_reduced_dict
    )
```

### 4.5 模型和优化器初始化

```python
def setup_model_and_optimizer(model_provider_func, model_type):
    """设置模型和优化器"""
    
    # 1. 构建模型
    model = get_model(model_provider_func, model_type)
    
    # 2. 构建优化器配置
    config = OptimizerConfig(
        # 从args中提取配置
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        # ...
    )
    
    # 3. 创建优化器
    optimizer = get_megatron_optimizer(
        config, 
        model,
        use_gloo_process_groups=args.enable_gloo_process_groups
    )
    
    # 4. 创建学习率调度器
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    
    # 5. 加载检查点（如果存在）
    if args.load is not None:
        args.iteration, args.num_floating_point_operations_so_far = \
            load_checkpoint(model, optimizer, opt_param_scheduler)
    else:
        args.iteration = 0
    
    # 6. 创建EMA模型（如果启用）
    if args.enable_ema:
        ema = get_model(model_provider_func, model_type)
        if args.iteration == 0:
            # 初始化EMA为模型参数
            for e, m in zip(ema, model):
                update_ema(e, m, rate=0)
        else:
            load_checkpoint(ema, None, None, load_arg="load_ema")
    
    return model, ema, optimizer, opt_param_scheduler
```

---

## 5. 数据处理

### 5.1 数据集架构

项目支持多种数据格式和加载方式：

#### 5.1.1 数据集类型

1. **预训练数据集**
   - WebDataset格式（推荐）
   - 支持大规模数据流式加载
   - 支持离线样本打包（Sample Packing）

2. **SFT数据集**
   - HuggingFace Datasets格式
   - 支持多种对话格式（Alpaca, ShareGPT等）
   - 支持多模态输入（图像+文本）

#### 5.1.2 数据配置

核心文件：`aiak_training_llm/data/blended_hf_dataset_config.py`

```python
@dataclass
class SFTDatasetConfig(BlendedHuggingFaceDatasetConfig):
    """SFT数据集配置"""
    
    # 数据路径和混合
    dataset_config_file: str          # 数据集配置文件路径
    blend: Optional[List]              # 数据混合比例
    blend_per_split: Optional[List]    # 各分割的混合比例
    
    # 模板和处理器
    chat_template: Optional[ChatTemplate]    # 对话模板
    processor: Optional[ProcessorMixin]      # 数据处理器
    
    # 训练配置
    sequence_length: int                     # 最大序列长度
    ignore_index: int = -100                 # 忽略索引
    train_on_prompt: bool = False           # 是否在提示上训练
    
    # 数据增强
    packing: bool = False                   # 是否打包数据
    sort_batch: bool = False                # 是否排序批次
    packing_batch_size: int = 10000        # 打包批次大小
    
    # 流式加载
    streaming: bool = False                 # 是否流式加载
    streaming_buffer_size: int = 16384     # 流式缓冲区大小
    
    # 并行配置
    context_parallel_size: Optional[int] = None  # 上下文并行大小
```

### 5.2 SFT数据集

核心文件：`aiak_training_llm/data/sft_dataset.py`

#### 5.2.1 支持的数据格式

```python
class SFTDataFormats:
    """支持的SFT数据格式"""
    ALPACA = "alpaca"           # Alpaca格式
    SHAREGPT = "sharegpt"       # ShareGPT格式
    OPENAI = "openai"           # OpenAI格式
```

#### 5.2.2 数据加载流程

```python
class SFTDataset(HuggingFaceDataset):
    """SFT数据集类"""
    
    def __init__(self, config: SFTDatasetConfig):
        super().__init__(config)
        
        # 1. 加载原始数据集
        raw_datasets = self._load_datasets()
        
        # 2. 格式转换
        processed_datasets = self._convert_format(raw_datasets)
        
        # 3. 分词处理
        tokenized_datasets = self._tokenize(processed_datasets)
        
        # 4. 数据打包（可选）
        if config.packing:
            tokenized_datasets = self._pack_samples(tokenized_datasets)
        
        # 5. 排序（可选）
        if config.sort_batch:
            tokenized_datasets = self._sort_by_length(tokenized_datasets)
        
        self.dataset = tokenized_datasets
```

### 5.3 多模态数据处理

核心文件：`aiak_training_llm/data/multimodal/qwen2vl_task_encoder.py`

#### 5.3.1 任务编码器

```python
class Qwen2VLTaskEncoder:
    """Qwen2-VL任务编码器"""
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = get_tokenizer()
        self.processor = AutoProcessor.from_pretrained(
            args.hf_tokenizer_path,
            trust_remote_code=True
        )
    
    def encode_sample(self, sample):
        """
        编码单个样本
        
        Args:
            sample: 包含文本和图像的字典
                {
                    "messages": [...],        # 对话消息
                    "images": [...],          # 图像路径列表
                    "videos": [...],          # 视频路径列表（可选）
                }
        
        Returns:
            编码后的样本，包含：
                - tokens: token ID
                - labels: 标签
                - imgs: 图像张量
                - image_grid_thw: 图像网格维度
                - attn_mask: 注意力掩码
                - cu_lengths: 累积长度（用于打包）
        """
        
        # 1. 应用对话模板
        text = self.processor.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 2. 处理图像
        image_inputs = None
        if sample.get("images"):
            images = [Image.open(img_path) for img_path in sample["images"]]
            image_inputs = self.processor.image_processor(
                images,
                return_tensors="pt"
            )
        
        # 3. 分词
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.args.seq_length
        )
        
        # 4. 合并输入
        encoded = {
            "tokens": text_inputs["input_ids"],
            "labels": self._build_labels(text_inputs, sample),
            "attn_mask": text_inputs["attention_mask"],
        }
        
        if image_inputs:
            encoded.update({
                "imgs": image_inputs["pixel_values"],
                "image_grid_thw": image_inputs["image_grid_thw"],
            })
        
        return encoded
```

### 5.4 数据打包（Sample Packing）

数据打包技术将多个短样本组合成一个批次，提高训练效率。

核心文件：`aiak_training_llm/data/sft_supervised_utils.py`

```python
def pack_samples(samples, max_length, context_parallel_size=1):
    """
    将多个样本打包到固定长度
    
    Args:
        samples: 样本列表
        max_length: 最大序列长度
        context_parallel_size: 上下文并行大小
    
    Returns:
        打包后的样本
    """
    
    packed_samples = []
    current_pack = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "cu_seqlens": [0],  # 累积序列长度
        "images": [],
        "image_grid_thw": [],
    }
    current_length = 0
    
    for sample in samples:
        sample_length = len(sample["input_ids"])
        
        # 检查是否可以添加到当前pack
        if current_length + sample_length <= max_length:
            # 添加到当前pack
            current_pack["input_ids"].extend(sample["input_ids"])
            current_pack["labels"].extend(sample["labels"])
            current_pack["attention_mask"].extend(sample["attention_mask"])
            current_pack["cu_seqlens"].append(current_length + sample_length)
            
            if "images" in sample:
                current_pack["images"].append(sample["images"])
                current_pack["image_grid_thw"].append(sample["image_grid_thw"])
            
            current_length += sample_length
        else:
            # 当前pack已满，开始新pack
            if current_length > 0:
                # 填充到max_length（考虑CP）
                pad_length = max_length - current_length
                if context_parallel_size > 1:
                    # 确保长度可被CP整除
                    pad_length = (max_length // context_parallel_size) * context_parallel_size - current_length
                
                current_pack["input_ids"].extend([0] * pad_length)
                current_pack["labels"].extend([-100] * pad_length)
                current_pack["attention_mask"].extend([0] * pad_length)
                
                packed_samples.append(current_pack)
            
            # 开始新pack
            current_pack = initialize_new_pack()
            current_length = 0
    
    return packed_samples
```

### 5.5 数据加载器

核心文件：`aiak_training_llm/data/multimodal/dataloader_provider.py`

```python
def get_train_loader(dataset, collator):
    """
    创建训练数据加载器
    
    Args:
        dataset: 数据集对象
        collator: 数据整理函数
    
    Returns:
        数据加载器迭代器
    """
    args = get_args()
    
    # 1. 创建数据采样器
    if args.dataloader_type == 'cyclic':
        # 循环数据加载器（用于WebDataset）
        dataloader = build_pretraining_data_loader(dataset)
    else:
        # 标准数据加载器
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
        )
    
    # 2. 创建循环迭代器
    def cyclic_iter(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    return cyclic_iter(dataloader)
```

### 5.6 数据整理器（Data Collator）

```python
class MultiModalDataCollatorForSupervisedDataset:
    """多模态监督数据集整理器"""
    
    def __init__(self, processor, plugin):
        self.processor = processor
        self.plugin = plugin
    
    def __call__(self, samples):
        """
        整理一个批次的样本
        
        Args:
            samples: 样本列表
        
        Returns:
            批次字典
        """
        
        # 1. 提取各个字段
        input_ids = [sample["input_ids"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        attention_masks = [sample["attention_mask"] for sample in samples]
        
        # 2. 填充到相同长度
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []
        
        for ids, labs, mask in zip(input_ids, labels, attention_masks):
            pad_length = max_length - len(ids)
            
            padded_input_ids.append(ids + [0] * pad_length)
            padded_labels.append(labs + [-100] * pad_length)
            padded_attention_masks.append(mask + [0] * pad_length)
        
        # 3. 转换为张量
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }
        
        # 4. 处理图像（如果存在）
        if "images" in samples[0]:
            images = []
            image_grid_thw = []
            
            for sample in samples:
                if sample["images"] is not None:
                    images.append(sample["images"])
                    image_grid_thw.append(sample["image_grid_thw"])
            
            if images:
                batch["images"] = torch.cat(images, dim=0)
                batch["image_grid_thw"] = torch.tensor(image_grid_thw, dtype=torch.long)
        
        return batch
```

---

## 6. 训练流程

### 6.1 三阶段训练策略

LLaVA-OneVision-1.5采用三阶段训练策略：

#### 6.1.1 Stage 1: Alignment Training（对齐训练）

**目标**：学习视觉-语言对齐

**数据**：LLaVA-558K（图像描述数据）

**配置**：
- 只训练Adapter（冻结LLM和ViT）
- 学习率：1e-4
- 训练步数：2500
- 序列长度：32768

**脚本示例**：
```bash
# examples/llava_ov_1_5/quick_start/stage_1_alignment_llava_ov_4b.sh

MODEL_ARGS=(
    --model-name llava-ov-1.5-4b
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules adapter        # 只训练adapter
    --seq-length 32768
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1.0e-4
    --train-iters 2500
    --bf16
)
```

#### 6.1.2 Stage 1.5: Mid-Training（中期训练）

**目标**：增强视觉理解能力

**数据**：LLaVA-OneVision-1.5-Mid-Training-85M（大规模高质量描述数据）

**配置**：
- 训练所有组件（LLM + ViT + Adapter）
- 学习率：1e-5
- 训练步数：20000
- 支持样本打包

**特点**：
- 概念平衡的数据分布
- 原生分辨率图像（最大1344x1344）
- 严格的质量控制

#### 6.1.3 Stage 2: Instruction Tuning（指令微调）

**目标**：学习指令跟随和多任务能力

**数据**：LLaVA-OneVision-1.5-Instruct-Data（综合指令数据）

**配置**：
- 训练所有组件
- 学习率：1e-5
- 训练步数：3500
- 支持多种任务类型

### 6.2 训练启动流程

#### 6.2.1 入口脚本

核心文件：`aiak_training_llm/train.py`

```python
def main():
    """训练主入口"""
    
    # 1. 解析参数
    args = parse_train_args()
    
    # 2. 构建训练器
    trainer = build_model_trainer(args)
    
    # 3. 启动训练
    trainer.train()
```

#### 6.2.2 参数解析

核心文件：`aiak_training_llm/train/arguments.py`

```python
def parse_train_args():
    """解析训练参数"""
    
    args = parse_arguments(
        extra_args_provider=aiak_extra_train_args_provider,
        validate_extra_args_provider=validate_aiak_extra_args,
    )
    
    return args

def aiak_extra_train_args_provider(parser):
    """添加AIAK自定义参数"""
    
    # 1. 模型参数
    parser = _add_extra_model_args(parser)
    
    # 2. 分词器参数
    parser = _add_extra_tokenizer_args(parser)
    
    # 3. SFT参数
    parser = _add_extra_sft_args(parser)
    
    # 4. 多模态参数
    parser = _add_extra_multimodal_args(parser)
    
    # 5. 并行参数
    parser = _add_extra_parallel_args(parser)
    
    return parser
```

#### 6.2.3 关键参数说明

**模型参数**：
```python
--model-name llava-ov-1.5-4b          # 模型架构
--trainable-modules adapter           # 可训练模块（all/adapter/language_model/vision_model）
```

**数据参数**：
```python
--data-path /path/to/dataset          # 数据路径
--dataloader-type external            # 数据加载器类型
--split 100,0,0                       # 训练/验证/测试分割
--chat-template qwen2-vl              # 对话模板
--num-workers 16                      # 数据加载工作线程
```

**训练参数**：
```python
--training-phase sft                  # 训练阶段（pretrain/sft）
--seq-length 32768                    # 序列长度
--micro-batch-size 1                  # 微批次大小
--global-batch-size 8                 # 全局批次大小
--train-iters 2500                    # 训练迭代次数

--lr 1.0e-4                           # 学习率
--min-lr 1.0e-6                       # 最小学习率
--lr-decay-style cosine               # 学习率衰减策略
--lr-warmup-fraction 0.002            # 学习率预热比例
--clip-grad 1.0                       # 梯度裁剪

--optimizer adam                      # 优化器
--adam-beta1 0.9                      # Adam beta1
--adam-beta2 0.99                     # Adam beta2
--weight-decay 0                      # 权重衰减
```

**并行参数**：
```python
--tensor-model-parallel-size 1        # 张量并行度
--pipeline-model-parallel-size 1      # 流水线并行度
--use-distributed-optimizer           # 使用分布式优化器
--attention-backend flash             # 注意力后端（flash/native）
```

**优化参数**：
```python
--bf16                                # 使用BF16精度
--recompute-granularity full          # 重计算粒度
--recompute-method uniform            # 重计算方法
--recompute-num-layers 4              # 重计算层数
```

### 6.3 分布式训练

#### 6.3.1 并行策略

LLaVA-OneVision-1.5支持多种并行策略：

1. **数据并行（Data Parallelism, DP）**
   - 每个GPU维护完整模型副本
   - 梯度在所有GPU间同步

2. **张量并行（Tensor Parallelism, TP）**
   - 模型权重按张量维度切分
   - 适合大型模型
   
3. **流水线并行（Pipeline Parallelism, PP）**
   - 模型按层切分到不同GPU
   - 使用微批次流水线减少气泡

4. **上下文并行（Context Parallelism, CP）**
   - 长序列按序列维度切分
   - 支持超长上下文训练

5. **序列并行（Sequence Parallelism, SP）**
   - 激活值按序列维度切分
   - 减少内存占用

#### 6.3.2 并行配置示例

```bash
# 8GPU单节点，使用TP=2, PP=1
GPUS_PER_NODE=8
TP=2
PP=1

# 实际数据并行度 = 8 / (2 * 1) = 4
# 全局批次大小 = 微批次大小 * 数据并行度 * 梯度累积步数
```

### 6.4 内存优化技术

#### 6.4.1 激活检查点（Activation Checkpointing）

```python
TRAINING_ARGS=(
    --recompute-granularity full      # 完全重计算
    --recompute-method uniform        # 均匀选择层
    --recompute-num-layers 4          # 重计算4层
)
```

#### 6.4.2 分布式优化器

```python
MODEL_PARALLEL_ARGS=(
    --use-distributed-optimizer       # 启用分布式优化器
)
```

特点：
- 优化器状态跨DP组分片
- 减少优化器内存占用
- 支持Adam/AdamW

#### 6.4.3 Flash Attention

```python
MODEL_PARALLEL_ARGS=(
    --attention-backend flash         # 使用Flash Attention
)
```

优势：
- 更快的注意力计算
- 更低的内存占用
- 支持长序列

### 6.5 检查点管理

#### 6.5.1 保存检查点

```python
TRAINING_ARGS=(
    --save /path/to/checkpoint        # 检查点保存路径
    --save-interval 2000              # 保存间隔
    --ckpt-format torch               # 检查点格式（torch/torch_dist/zarr）
)
```

#### 6.5.2 加载检查点

```python
TRAINING_ARGS=(
    --load /path/to/checkpoint        # 检查点加载路径
    --ckpt-fully-parallel-load        # 完全并行加载
)
```

#### 6.5.3 异步保存

支持异步检查点保存以减少训练中断：

```python
TRAINING_ARGS=(
    --async-save                      # 启用异步保存
)
```

---

## 7. 模型转换

### 7.1 转换流程

LLaVA-OneVision-1.5提供完整的模型转换工具，支持HuggingFace格式与Megatron格式互转。

#### 7.1.1 HuggingFace → Megatron

转换脚本：`examples/llava_ov_1_5/convert/convert_4b_hf_to_mcore.sh`

```bash
#!/bin/bash
LOAD=$1    # HuggingFace模型路径
SAVE=$2    # Megatron输出路径
TP=$3      # 张量并行度
PP=$4      # 流水线并行度

# 步骤1：转换语言模型
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL

# 步骤2：转换视觉模型
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL

# 步骤3：转换适配器
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# 步骤4：转换视觉补丁嵌入
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# 步骤5：合并所有组件
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP
```

#### 7.1.2 Megatron → HuggingFace

转换脚本：`examples/llava_ov_1_5/convert/convert_4b_mcore_to_hf.sh`

```bash
#!/bin/bash
LOAD=$1    # Megatron模型路径
SAVE=$2    # HuggingFace输出路径
TP=$3      # 源张量并行度
PP=$4      # 源流水线并行度

# 步骤1：转换语言模型
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llava-ov-1.5-4b/qwen3.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL

# 步骤2-4：转换其他组件（类似）
# ...

# 步骤5：合并到HuggingFace格式
python $CONVERT_CHECKPOINT_PATH/custom/llavaov_1_5/merge_huggingface.py \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --adapter_path $SAVE_ADAPTER \
    --vision_patch $SAVE_PATCH \
    --save_ckpt_path $SAVE

# 复制非模型文件（tokenizer等）
find $ORIGINAL_HF_PATH/ -type f -not -iname '*safetensors*' -exec cp {} $SAVE/ ';'
```

### 7.2 转换配置

转换配置文件定义了权重映射规则：

#### 7.2.1 语言模型配置示例

文件：`tools/convert_checkpoint/config/llava-ov-1.5-4b/qwen3.json`

```json
{
  "model_type": "GPT",
  "num_layers": 36,
  "hidden_size": 2560,
  "ffn_hidden_size": 9728,
  "num_attention_heads": 32,
  "num_query_groups": 8,
  
  "weight_mapping": {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.layers.{}.self_attention.linear_qkv.weight": [
      "model.layers.{}.self_attn.q_proj.weight",
      "model.layers.{}.self_attn.k_proj.weight", 
      "model.layers.{}.self_attn.v_proj.weight"
    ],
    "decoder.layers.{}.self_attention.linear_proj.weight": 
      "model.layers.{}.self_attn.o_proj.weight",
    "decoder.layers.{}.mlp.linear_fc1.weight": [
      "model.layers.{}.mlp.gate_proj.weight",
      "model.layers.{}.mlp.up_proj.weight"
    ],
    "decoder.layers.{}.mlp.linear_fc2.weight": 
      "model.layers.{}.mlp.down_proj.weight",
    "output_layer.weight": "lm_head.weight"
  }
}
```

#### 7.2.2 视觉模型配置示例

文件：`tools/convert_checkpoint/config/llava-ov-1.5-4b/vision-model.json`

```json
{
  "model_type": "VisionModel",
  "num_layers": 24,
  "hidden_size": 1024,
  "num_attention_heads": 16,
  
  "weight_mapping": {
    "decoder.layers.{}.self_attention.linear_qkv.weight": [
      "visual.blocks.{}.attn.qkv.weight"
    ],
    "decoder.layers.{}.self_attention.linear_proj.weight":
      "visual.blocks.{}.attn.proj.weight",
    "decoder.layers.{}.pre_mlp_layernorm.weight":
      "visual.blocks.{}.norm2.weight",
    "decoder.layers.{}.mlp.linear_fc1.weight":
      "visual.blocks.{}.mlp.fc1.weight",
    "decoder.layers.{}.mlp.linear_fc2.weight":
      "visual.blocks.{}.mlp.fc2.weight"
  }
}
```

### 7.3 权重映射逻辑

核心文件：`tools/convert_checkpoint/model.py`

```python
def convert_checkpoint(load_platform, save_platform, config, load_path, save_path):
    """执行检查点转换"""
    
    # 1. 加载源检查点
    if load_platform == "huggingface":
        src_checkpoint = load_huggingface_checkpoint(load_path)
    elif load_platform == "mcore":
        src_checkpoint = load_mcore_checkpoint(load_path, config)
    
    # 2. 创建目标检查点结构
    dst_checkpoint = {}
    
    # 3. 根据映射规则转换权重
    weight_mapping = config["weight_mapping"]
    
    for dst_key_template, src_key_template in weight_mapping.items():
        # 处理层索引
        if "{}" in dst_key_template:
            for layer_idx in range(config["num_layers"]):
                dst_key = dst_key_template.format(layer_idx)
                
                if isinstance(src_key_template, list):
                    # 需要拼接多个源权重
                    src_keys = [k.format(layer_idx) for k in src_key_template]
                    weights = [src_checkpoint[k] for k in src_keys]
                    dst_checkpoint[dst_key] = torch.cat(weights, dim=0)
                else:
                    src_key = src_key_template.format(layer_idx)
                    dst_checkpoint[dst_key] = src_checkpoint[src_key]
        else:
            # 直接映射
            dst_checkpoint[dst_key_template] = src_checkpoint[src_key_template]
    
    # 4. 处理张量并行切分（如果需要）
    if save_platform == "mcore" and config.get("tensor_model_parallel_size", 1) > 1:
        dst_checkpoint = split_for_tensor_parallel(
            dst_checkpoint,
            config["tensor_model_parallel_size"]
        )
    
    # 5. 保存目标检查点
    if save_platform == "huggingface":
        save_huggingface_checkpoint(dst_checkpoint, save_path)
    elif save_platform == "mcore":
        save_mcore_checkpoint(dst_checkpoint, save_path, config)
```

---

## 8. 配置与优化

### 8.1 推荐配置

#### 8.1.1 不同规模模型的配置

**4B模型（单节点8卡A100-80GB）**:
```bash
TP=1
PP=1
MBS=1
GBS=8
SEQ_LEN=32768
RECOMPUTE_LAYERS=4

# 内存占用：约60GB/GPU
# 训练速度：约1.2 samples/s
```

**8B模型（单节点8卡A100-80GB）**:
```bash
TP=2
PP=1
MBS=1
GBS=8
SEQ_LEN=32768
RECOMPUTE_LAYERS=8

# 内存占用：约70GB/GPU
# 训练速度：约0.8 samples/s
```

**32B模型（2节点16卡A100-80GB）**:
```bash
TP=4
PP=2
MBS=1
GBS=16
SEQ_LEN=32768
RECOMPUTE_LAYERS=16

# 内存占用：约75GB/GPU
# 训练速度：约0.3 samples/s
```

#### 8.1.2 性能优化建议

1. **Flash Attention**: 始终启用以获得最佳性能
```bash
--attention-backend flash
```

2. **BF16训练**: 推荐使用BF16而非FP16
```bash
--bf16
```

3. **激活检查点**: 根据模型大小调整重计算层数
```bash
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers <根据模型调整>
```

4. **分布式优化器**: 大模型必备
```bash
--use-distributed-optimizer
```

5. **数据加载**: 增加工作线程数
```bash
--num-workers 16
```

### 8.2 常见问题与解决

#### 8.2.1 OOM（内存溢出）

**症状**: CUDA out of memory

**解决方案**:
1. 减小微批次大小
```bash
--micro-batch-size 1  # 最小值
```

2. 增加重计算层数
```bash
--recompute-num-layers 8  # 或更多
```

3. 增加张量并行度
```bash
--tensor-model-parallel-size 2  # 或更大
```

4. 启用序列并行
```bash
--sequence-parallel
```

#### 8.2.2 训练速度慢

**症状**: 训练吞吐量低

**解决方案**:
1. 检查Flash Attention是否启用
```bash
--attention-backend flash
```

2. 优化数据加载
```bash
--num-workers 16          # 增加工作线程
--dataloader-type cyclic  # 使用循环数据加载器
```

3. 启用混合精度
```bash
--bf16  # 或 --fp16
```

4. 调整并行配置
```bash
# 平衡TP和DP
# TP太大会增加通信开销
# DP太大可能导致批次大小过小
```

#### 8.2.3 损失为NaN

**症状**: Loss突然变成NaN

**解决方案**:
1. 降低学习率
```bash
--lr 5.0e-5  # 从更小的值开始
```

2. 启用梯度裁剪
```bash
--clip-grad 1.0
```

3. 检查数据质量
```bash
# 确保数据中没有异常值
```

4. 使用更稳定的优化器配置
```bash
--optimizer adam
--adam-beta1 0.9
--adam-beta2 0.999
--adam-eps 1e-08
```

### 8.3 监控与日志

#### 8.3.1 TensorBoard监控

```bash
LOGGING_ARGS=(
    --tensorboard-dir ./tensorboard
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
)
```

监控指标：
- 训练损失
- 学习率
- 梯度范数
- 内存使用
- 训练吞吐量

#### 8.3.2 WandB集成

```bash
export WANDB_API_KEY=<your-api-key>

LOGGING_ARGS=(
    --wandb-project llava-onevision-1.5
    --wandb-exp-name stage1-alignment
)
```

#### 8.3.3 日志级别

```bash
LOGGING_ARGS=(
    --log-interval 1              # 日志记录间隔
    --detail-log-interval 20      # 详细日志间隔
    --timing-log-level 0          # 计时日志级别
)
```

### 8.4 最佳实践总结

1. **渐进式训练**
   - Stage 1: 只训练Adapter（快速对齐）
   - Stage 1.5: 全模型训练（增强能力）
   - Stage 2: 指令微调（任务适配）

2. **数据质量**
   - 使用高质量、多样化的数据
   - 进行概念平衡
   - 严格的质量控制

3. **内存管理**
   - 合理配置并行度
   - 使用激活检查点
   - 启用分布式优化器

4. **训练稳定性**
   - 适当的学习率和warmup
   - 梯度裁剪
   - 定期保存检查点

5. **效率优化**
   - Flash Attention
   - 混合精度训练
   - 样本打包
   - 高效的数据加载

---

## 总结

LLaVA-OneVision-1.5是一个完整、高效、可扩展的多模态大模型训练框架，具有以下核心优势：

1. **完全开源**: 提供完整的代码、数据、配置和训练日志
2. **高效训练**: 基于Megatron-LM深度优化，支持各种并行和优化策略
3. **模块化设计**: 易于扩展和定制
4. **生产级质量**: 经过大规模验证，训练成本仅$16K

通过本文档，您应该能够：
- 理解项目的整体架构和设计思路
- 掌握模型的实现细节
- 配置和启动训练任务
- 进行模型转换和部署
- 优化训练性能和解决常见问题

建议在实际使用时：
1. 从小规模实验开始，验证配置
2. 逐步增加模型规模和数据量
3. 密切监控训练指标和系统资源
4. 根据具体硬件和任务需求调整配置

如需更多帮助，请参考：
- [项目GitHub](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)
- [技术报告](https://arxiv.org/abs/2509.23661)
- [HuggingFace模型](https://huggingface.co/collections/lmms-lab/llava-onevision-15-68d385fe73b50bd22de23713)


