# training_model

## 整体训练流程架构

这是一个基于Megatron-LM框架的多模态（视觉-语言）模型训练脚本。训练流程从`__main__`入口开始：

```427:439:examples/qwen2_5_vl/pretrain_qwen.py
if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer'},
        extra_args_provider=get_patch_args,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        non_loss_data_func=run_online_eval,
    )
```

## 核心组件详解

### 1. **模型构建 (`model_provider`)**

```70:133:examples/qwen2_5_vl/pretrain_qwen.py
def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, vp_stage: Optional[int] = None
) -> Union[Qwen2_5VLModel]:
    args = get_args()
    print_rank_0("start building qwen2-vl model ...")

    # Config of vit, llm and projector
    config = core_transformer_config_from_args(args, Qwen2VLTransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen2-VL model is only implemented with TransformerEngine!")
    
    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.num_layers_in_first_pipeline_stage = None
    vision_projector_config = get_vision_projection_config(deepcopy(config), vision_config.hidden_size, vision_config.spatial_merge_size)
    
    print_rank_0("building Qwen2-5-VL model in TE...")
    # Layer Specs of vit, llm and projector
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.qk_layernorm)
    vision_model_spec = get_qwen2vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules

    model = Qwen2_5VLModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,

        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        drop_vision_class_token=False, # NOTE: no class token to drop?

        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec, 
        vision_projection_type='mlp',
        allow_missing_vision_projection_checkpoint= False, # TODO: may parameterized

        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        
        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,

        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage
    )

    model.freeze(
        freeze_language_model=getattr(args, 'freeze_LM', False), 
        freeze_vision_model=getattr(args, 'freeze_ViT', False), 
        freeze_vision_projection=False
    )

    return model
```

**作用**：构建Qwen2.5-VL多模态模型
- 创建三个组件的配置：语言模型、视觉模型、视觉投影层
- 使用TransformerEngine实现（强制要求）
- 支持冻结语言模型或视觉模型的参数

### 2. **数据加载 (`train_valid_test_dataloaders_provider`)**

```356:403:examples/qwen2_5_vl/pretrain_qwen.py
def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    if get_tensor_model_parallel_rank() != 0:
        return None, None, None

    args = get_args()

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )
    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0("loading dataloader checkpoint failed. Skipping. " + str(e))

    valid_dataloader = [
        EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
        for valid_ds in valid_ds1
    ]
    test_dataloader = None # NOTE: no test

    return EnergonDataloader(train_dataloader), valid_dataloader, None
```

**作用**：创建分布式数据加载器
- 只在tensor并行的rank 0上创建dataloader
- 使用Megatron Energon框架处理多模态数据
- 支持断点续训时恢复dataloader状态

### 3. **数据集创建 (`datasets_provider`)**

```316:353:examples/qwen2_5_vl/pretrain_qwen.py
def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    train_dataset = get_train_dataset(
        args.train_data_path[0],
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=1000,
        max_samples_per_sequence=100,
        shuffle_buffer_size=100,
        handler=print_error_handler,
        image_decode="pil",
    )

    val_datasets = get_val_datasets(
        args.valid_data_path[0],
        batch_size=args.micro_batch_size,
        # This is the total number over all workers
        # limit=args.eval_iters * get_num_microbatches(),
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        handler=print_error_handler,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = [
        # Limit the dataset to eval_iters * num_microbatches
        LimitDataset(
            # Repeat the inner dataset in case it's too short
            RepeatDataset(val_ds, worker_config=worker_config),
            length=args.eval_iters * get_num_microbatches(),
            worker_config=worker_config,
            reset_after_epoch=True,
        )
        for val_ds, _src_ds in val_datasets
    ]

    return train_dataset, val_datasets_without_source_datasets, None
```

**作用**：配置训练和验证数据集
- 使用`TaskEncoder`处理多模态任务
- 验证集使用`RepeatDataset`和`LimitDataset`包装，确保固定的评估步数

### 4. **批次数据获取 (`get_batch`)**

```166:229:examples/qwen2_5_vl/pretrain_qwen.py
def get_batch(model, data_iterator):
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    data_text =  broadcast_data(["text"], data, torch.int64)["text"]
    target =  broadcast_data(["target"], data, torch.int64)["target"]
    # shape: num_tiles x c x h x w
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]

    # shape: num_tiles x c x h w
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]
    # shape: n_image_samples
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]
    # shape: n_video_samples
    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']


    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        model, tokens, image_thw_grids, video_thw_grids, labels, tokenizer.pad_token_id, second_per_grid_ts
    )
    torch.cuda.nvtx.range_pop()

    return (
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids, 
        imgs, 
        videos,
        image_thw_grids, 
        video_thw_grids,
        image_input_mask, 
        video_input_mask
    )
```

**作用**：从数据迭代器获取并处理一个batch
- 使用`broadcast_data`在tensor并行组间广播数据（只有rank 0读取数据）
- 处理多模态输入：文本tokens、图像、视频
- 获取位置编码和损失掩码
- 使用NVTX标记进行性能分析

### 5. **位置编码和掩码生成 (`get_ltor_masks_and_position_ids`)**

```135:164:examples/qwen2_5_vl/pretrain_qwen.py
def get_ltor_masks_and_position_ids(
        model,
        input_ids, 
        image_thw_grids,
        video_thw_grids,
        target, 
        pad_token, 
        second_per_grid_ts,
        ignore_index=None
    ):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=input_ids != pad_token
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None

    return attention_mask, loss_mask, position_ids
```

**作用**：为多模态输入生成位置编码
- 使用特殊的RoPE索引处理图像/视频的时空网格信息
- `position_ids`是3维的（支持多模态RoPE）
- 创建损失掩码，忽略padding和可选的prompt部分

### 6. **前向传播 (`forward_step`)**

```260:300:examples/qwen2_5_vl/pretrain_qwen.py
def forward_step(data_iterator, model: Qwen2_5VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids, 
        imgs, 
        videos,
        image_thw_grids, 
        video_thw_grids,
        image_input_mask, 
        video_input_mask
    ) = get_batch(unwrap_model(model), data_iterator)
    timers("batch-generator").stop()

    vision_data = torch.cat([imgs, videos], dim=0)
    vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)

    output_tensor = model(
        input_ids = tokens,
        position_ids = position_ids,
        vision_data = vision_data,
        vision_grid_thw =  vision_grid,
        video_start_index = image_input_mask.sum().cpu().item(),
        image_input_mask = image_input_mask,
        video_input_mask = video_input_mask,
        attention_mask = attention_mask,
        labels = labels
    )

    return output_tensor, partial(loss_func, loss_mask)
```

**作用**：执行一次训练前向传播
- 合并图像和视频数据
- 通过`video_start_index`区分图像和视频
- 返回输出和损失函数（使用`partial`绑定loss_mask）

### 7. **损失计算 (`loss_func`)**

```231:258:examples/qwen2_5_vl/pretrain_qwen.py
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    return loss[0] / loss[1] * args.context_parallel_size, {"lm loss": averaged_loss}
```

**作用**：计算并聚合损失
- 应用loss_mask忽略特定tokens
- 在context并行组和data并行组间聚合损失
- 支持NaN检测
- 返回局部损失和全局平均损失

### 8. **辅助类和函数**

#### `EnergonDataloader`

```406:420:examples/qwen2_5_vl/pretrain_qwen.py
class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()
```

**作用**：包装Energon dataloader以兼容Megatron训练循环
- 提供无限循环的迭代器
- 支持保存和恢复状态

#### `cyclic_iter`

```422:425:examples/qwen2_5_vl/pretrain_qwen.py
def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x
```

**作用**：创建无限循环的迭代器

#### 评估相关函数

```302:314:examples/qwen2_5_vl/pretrain_qwen.py
def run_online_eval(model):
    """Run an evaluation benchmark during training."""
    # Do nothing.
    return []

def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)
```

## 训练流程总结

1. **初始化阶段**：
   - 解析命令行参数
   - 初始化分布式环境（TP、PP、DP、CP）
   - 创建tokenizer

2. **模型构建阶段**（`model_provider`）：
   - 创建语言模型配置
   - 创建视觉模型配置
   - 创建视觉投影层配置
   - 实例化`Qwen2_5VLModel`
   - 应用参数冻结策略

3. **数据准备阶段**（`train_valid_test_dataloaders_provider`）：
   - 配置分布式worker
   - 创建训练/验证数据集
   - 包装成Energon dataloader

4. **训练循环**（由Megatron的`pretrain`函数驱动）：
   - 获取batch（`get_batch`）
   - 前向传播（`forward_step`）
   - 计算损失（`loss_func`）
   - 反向传播（Megatron自动处理）
   - 参数更新（Megatron自动处理）
   - 定期验证和保存checkpoint

5. **关键特性**：
   - 支持图像和视频的混合训练
   - 使用3D RoPE位置编码处理时空信息
   - 支持模型并行（TP、PP、DP、CP）
   - 支持选择性参数冻结
   - 使用TransformerEngine加速

这是一个完整的多模态大模型训练pipeline，充分利用了Megatron-LM的分布式训练能力。
