# cr阅读记录

核心文件：
- vision_transformer_block.py
- transformer_block.py
- transformer_config.py
- layer_specs.py
- visionmodel.py
- model.py
- llm.py 
## model.py

最顶层的为model.py，组装visionmodel和llm

## layer_specs.py
layer_specs.py声明llm或者vit的整体架构，其定义和用法如下（以llm的架构为例子）：
```python
def get_gpt_layer_with_transformer_engine_spec(
    qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = get_mlp_module_spec(
        use_te=True, num_experts=None, moe_grouped_gemm=False
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Qwen2VLSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    k_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
```

这是transformer的一个layer的定义，在block文件中需要获取layer数然后组合成transformerblock：
```python
def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")
```

然后就会被使用构建transformer的架构（self.submodules就是上面函数的返回）：

```python
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )
```
最终到megatron/core/transformer/spec_utils.py的build_module函数中生成具体的模型结构类：
```python
def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    # If the passed `spec_or_module` is
    # a `Function`, then return it as it is
    # NOTE: to support an already initialized module add the following condition
    # `or isinstance(spec_or_module, torch.nn.Module)` to the following if check
    if isinstance(spec_or_module, types.FunctionType):
        return spec_or_module

    # If the passed `spec_or_module` is actually a spec (instance of
    # `ModuleSpec`) and it specifies a `Function` using its `module`
    # field, return the `Function` as it is
    if isinstance(spec_or_module, ModuleSpec) and isinstance(
        spec_or_module.module, types.FunctionType
    ):
        return spec_or_module.module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):
        module = spec_or_module
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):
        module = spec_or_module.module
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module)

    # If the imported module is actually a `Function` return it as it is
    if isinstance(module, types.FunctionType):
        return module

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code

    # Add the `submodules` argument to the module init call if it exists in the
    # spec.
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:
        kwargs["submodules"] = spec_or_module.submodules

    try:
        return module(
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
        )
    except Exception as e:
        # improve the error message since we hide the module name in the line above
        import sys

        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )
```
## 

## visionmodel+llm注意点
## 组合 Vision Model 和 LLM 需要注意的关键点

基于对代码的分析,我将从以下几个维度为您列举所有注意事项:

### **1. 配置兼容性 (TransformerConfig)**

#### 1.1 **Hidden Size 匹配**
- **Vision 输出维度** → **Projection 层** → **LLM 输入维度** 必须对齐
- 示例参考:
  - `qwen2_5_vl`: Vision hidden_size=1280, LLM需要通过projection映射
  - `llava_mcore`: Vision和LLM的hidden_size通过MultimodalProjector连接

#### 1.2 **序列并行 (Sequence Parallel) 配置**
- Vision model 和 LLM 的 `sequence_parallel` 设置需要协调
- Qwen2_5VL例子: Vision设置为False, LLM可以True
```python
# vision config
config.sequence_parallel = False

# language config  
config.sequence_parallel = True  # 需要配合TE的TEDotProductAttention
```

#### 1.3 **张量并行 (Tensor Parallel) 配置**
- `tp_comm_overlap` 设置需要一致
- Vision encoder通常设置为 False
- LLM可以根据需要启用

### **2. Position Embedding 兼容性**

#### 2.1 **Position Embedding 类型**
不同的position embedding类型**不兼容**,需要特别注意:

| 模型类型 | Position Embedding | 实现方式 |
|---------|-------------------|---------|
| CLIP ViT | Learned Absolute | `torch.nn.Embedding` |
| Qwen2_5 Vision | Custom RoPE | `VisionRotaryEmbedding` |
| LLaMA LLM | RoPE | `RotaryEmbedding` |
| Qwen2VL LLM | M-RoPE | `MultimodalRotaryEmbedding` |
| GPT | Learned Absolute | `torch.nn.Embedding` |

**关键注意**:
- Vision用Learned Absolute,LLM用RoPE → 需要特殊处理
- Qwen2VL用专门的M-RoPE(Multimodal RoPE),需要`mrope_section`参数
- Position embedding在embedding层和transformer层都会用到

#### 2.2 **RoPE参数对齐**
如果都使用RoPE,需要确保:
```python
rotary_base  # 基础频率(10000, 1000000等)
rotary_percent  # RoPE维度百分比
rotary_interleaved  # 是否交错
kv_channels  # 必须一致
```

### **3. Attention Mechanism 兼容性**

#### 3.1 **Attention Mask 类型**
```python
# Vision Model
attn_mask_type = AttnMaskType.no_mask  # 或 causal_pad

# Language Model  
attn_mask_type = AttnMaskType.causal  # 通常是causal
```

#### 3.2 **Query-Key-Value 配置**
- `num_attention_heads`: 注意力头数量
- `num_query_groups`: GQA (Grouped Query Attention) 配置
  - Vision通常 `num_query_groups = num_attention_heads` (无GQA)
  - LLM可能使用GQA (如Qwen2用4, LLaMA3用8)
  
#### 3.3 **QKV Bias**
```python
add_qkv_bias  # Vision和LLM可能不同
# Vision: 通常 True
# LLM: 取决于具体模型
```

### **4. MLP/FFN 层配置**

#### 4.1 **FFN Hidden Size**
```python
ffn_hidden_size  # 中间层维度
gated_linear_unit  # 是否使用门控(SwiGLU等)
```

- Vision通常不用gated (GELU激活)
- LLM常用SwiGLU (gated=True)

#### 4.2 **激活函数**
```python
# Vision models
activation_func = torch.nn.functional.gelu  # 或 quick_gelu

# Language models  
activation_func = F.silu  # (if gated_linear_unit=True)
```

### **5. Layer Normalization**

#### 5.1 **Normalization 类型**
```python
# Vision: 通常LayerNorm
normalization = 'LayerNorm'

# LLM: 常用RMSNorm  
normalization = 'RMSNorm'
layernorm_zero_centered_gamma = True/False
```

#### 5.2 **LayerNorm 位置**
- Vision: Pre-LN (ln_pre) + 可选Post-LN (CLIP vs SigLIP)
- LLM: Pre-LN (现代架构) 或 Post-LN

### **6. Vision Projection Layer**

#### 6.1 **Projection 配置**
这是连接Vision和LLM的**关键桥梁**:
```python
# Input: vision_hidden_size * (spatial_merge_size ** 2)
# Output: language_hidden_size

projection_config.ffn_hidden_size  # 中间层大小
projection_type  # "mlp", "affine"等
```

#### 6.2 **Spatial Merge**
Qwen2VL特有,需要注意:
```python
spatial_merge_size = 2  # 2x2合并
merge_hidden_size = hidden_size * spatial_merge_size ** 2
```

### **7. Pipeline Parallelism 相关**

#### 7.1 **pre_process / post_process 标志**
```python
# Vision Model
pre_process=True   # 包含patch embedding
post_process=True  # 包含projection

# Language Model
pre_process=True   # 包含word embedding  
post_process=True  # 包含output layer
```

#### 7.2 **add_encoder / add_decoder**
多模态模型特有:
```python
add_encoder = True   # 是否构建vision encoder
add_decoder = True   # 是否构建language decoder
```

不同pipeline stage需要不同配置。

### **8. 数据流和嵌入融合**

#### 8.1 **Embedding 融合方式**

**方式1: LLaVA式 - 位置替换**
```python
# 在特定位置(image_token_index=-200)插入vision embeddings
# final_embeddings = [text_emb, image_emb, text_emb, ...]
```

**方式2: Qwen2VL式 - Mask引导**
```python
# 使用image_input_mask和video_input_mask
# 在embedding层直接融合
language_embeddings = self.language_model.embedding(
    input_ids=input_ids,
    image_input_mask=image_input_mask,
    image_embeds=image_embeds
)
```

#### 8.2 **Position IDs 处理**
- LLaVA: 每个图像token占1个position
- Qwen2VL: 图像使用3D position (时间,高,宽), 文本使用1D

### **9. 特殊Token处理**

#### 9.1 **Vision Class Token**
```python
drop_vision_class_token  # 是否丢弃class token
class_token_len  # class token长度(0, 1, 8等)
```

- CLIP: 有class token (可选择丢弃)
- SigLIP: 无class token
- 需要与后续处理对齐

#### 9.2 **特殊Token ID**
```python
image_token_index = -200  # 或 151655
video_token_id = 151656
vision_start_token_id = 151652
```

不同模型的token ID定义不同,需要适配。

### **10. Checkpoint 加载**

#### 10.1 **参数名称映射**
不同来源的模型参数名可能不同:
```python
# Vision: "vision_model.*"  
# LLM: "language_model.*"
# Projection: "vision_projection.*"
```

#### 10.2 **允许缺失权重**
```python
allow_missing_vision_projection_checkpoint = True
# 当只有预训练vision和LLM,projection需要从头训练
```

### **11. Transformer Layer Specs**

#### 11.1 **Layer Spec 结构必须匹配**
```python
# Vision
ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=...,
        mlp=...,
    )
)

# 需要确保submodules的类型兼容
```

#### 11.2 **Core Attention实现**
```python
# 使用Transformer Engine时
TEDotProductAttention  # 支持flash attention

# 不使用TE时  
DotProductAttention
```

两者参数和行为可能不同。

### **12. 训练相关配置**

#### 12.1 **梯度累积和重计算**
```python
recompute_granularity  # 'full', 'selective'
recompute_method  # 'uniform', 'block'
recompute_num_layers
```

Vision和LLM可能需要不同的策略。

#### 12.2 **混合精度训练**
```python
fp16_lm_cross_entropy  # LM loss是否用fp16
bf16 / fp16  # 整体精度
fp8  # 是否启用FP8
```

#### 12.3 **冻结模块**
```python
freeze_language_model = True
freeze_vision_model = True  
freeze_vision_projection = False
```

### **13. 特定模型的特殊要求**

#### 13.1 **Qwen2VL 特殊项**
```python
mrope_section  # M-RoPE的section划分  
fullatt_block_indexes  # 全attention的block索引
window_size  # 窗口attention大小
temporal_patch_size  # 时间patch大小
```

#### 13.2 **DeepSeek-V2 MLA**
```python
multi_latent_attention  # 多潜在attention
q_lora_rank, kv_lora_rank  # LoRA rank
qk_rope_head_dim  # RoPE head维度
```

### **14. 输入数据格式**

#### 14.1 **Vision 输入**
```python
# CLIP/LLaVA
images: [num_tiles, 3, H, W]

# Qwen2VL  
vision_data: [total_thw, features]
grid_thw: [num_images, 3]  # temporal, height, width
```

#### 14.2 **文本输入**
```python
input_ids: [batch, seq_len]
position_ids: [batch, seq_len] 或 [3, batch, seq_len]  # Qwen2VL
attention_mask: [batch, 1, seq_len, seq_len]
```

### **15. 推理优化**

#### 15.1 **KV Cache**
```python
inference_params.key_value_memory_dict["image_tokens_count"]
# 存储image token数量,用于KV cache offset
```

#### 15.2 **Flash Decode**
```python
flash_decode = True
# 需要预计算的cos/sin for RoPE
```

---

## **实践建议**

### **最安全的组合策略:**

1. **选择相似架构的模型**
   - Vision: CLIP系列 + LLM: GPT系列 (都是标准Transformer)
   - Vision: Qwen2VL Vision + LLM: Qwen2 LLM (同系列)

2. **使用相同的Position Embedding类型**
   - 都用RoPE,或都用Learned Absolute
   - 避免混用

3. **统一Normalization和Activation**
   - 检查LayerNorm vs RMSNorm
   - 检查GELU vs SiLU

4. **仔细设计Projection层**
   - 确保维度正确映射
   - 考虑是否需要spatial merge

5. **测试Pipeline Parallel兼容性**
   - 从单GPU开始
   - 逐步增加PP和TP

### **调试检查清单:**

```python
# 1. 打印各层输出shape
print(f"Vision output: {vision_output.shape}")
print(f"After projection: {projected.shape}")  
print(f"LLM input: {llm_input.shape}")

# 2. 检查dtype一致性
assert vision_output.dtype == llm_input.dtype

# 3. 检查device
assert vision_output.device == llm_input.device

# 4. 检查config关键参数
print(f"Vision hidden: {vision_config.hidden_size}")
print(f"LLM hidden: {language_config.hidden_size}")
print(f"Projection out: {projection_config.hidden_size}")
```

希望这份详尽的清单能帮助您成功组合不同的Vision Model和LLM!如果您有具体的模型组合计划,我可以提供更针对性的建议。
