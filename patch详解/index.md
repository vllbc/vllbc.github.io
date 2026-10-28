# patch详解


**Patch** 就是把一张大图像切分成多个**小方块**（小图像块），然后把每个小方块当作一个"token"（类似文本中的单词）来处理。

#### **为什么需要 Patch？**

**传统 CNN**：
```
图像 [224×224×3] → Conv → Pool → ... → 特征
```

**Vision Transformer (ViT)**：
```
图像 [224×224×3] → 切成 Patches → Transformer → 特征
```

**核心思想**：让 Transformer 能像处理文本一样处理图像！

### 4.2 Patch 切分过程（图解）

#### **步骤 1: 原始图像**

```
原始图像：224×224 像素，RGB 三通道
┌─────────────────────────────────────┐
│                                     │
│         🖼️ 猫咪图片                 │  224 像素
│                                     │
│                                     │
└─────────────────────────────────────┘
           224 像素
```

#### **步骤 2: 切分成 Patches**

假设 `patch_size = 16`，图像被切成 16×16 的小块：

```
切分后：14×14 = 196 个 patches

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│P1│P2│P3│P4│P5│P6│P7│P8│P9│..│  │  │  │14│
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│15│16│17│18│19│20│21│22│23│..│  │  │  │28│
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│29│30│31│32│33│34│35│36│37│..│  │  │  │42│
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │ . │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│183│184│185│186│187│188│189│190│191│..│  │  │  │196│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

每个 Patch：16×16×3 = 768 个像素值
总共：196 个 patches
```

#### **步骤 3: 展平 Patch**

每个 Patch 从 3D 张量变成 1D 向量：

```python
Patch[0]: [16, 16, 3] → Flatten → [768]

┌─────────┐
│ R值×256 │
│ G值×256 │  ⟹  [768维向量]
│ B值×256 │
└─────────┘
```

#### **步骤 4: 线性投影（Embedding）**

将每个 768 维向量投影到模型的隐藏维度（如 1152）：

```python
[768] × W[768×1152] + b[1152] → [1152]

最终输出：[196个patches, 1152维] 的 Embedding
```

### 4.3 代码实现详解

#### **PatchEmbed 类**

```python
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,           # 每个patch的边长
        temporal_patch_size: int = 2,   # 视频时间维度patch大小
        in_channels: int = 3,           # 输入通道数（RGB=3）
        embed_dim: int = 1152,          # 输出嵌入维度
        bias: bool = False
    ):
        super().__init__()
        
        # 关键：使用 3D 卷积同时处理时间和空间维度
        kernel_size = [temporal_patch_size, patch_size, patch_size]  # [2, 14, 14]
        self.proj = nn.Conv3d(
            in_channels,              # 输入：3 通道
            embed_dim,                # 输出：1152 通道
            kernel_size=kernel_size,  # 卷积核大小
            stride=kernel_size,       # 步长=核大小（无重叠）
            bias=bias
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 shape: [n_patches, channels, t, h, w]
        # 例如：[N, 3, 2, 16, 16]
        
        # 重塑为 5D
        hidden_states = hidden_states.view(
            -1,                          # batch中的patch数
            self.in_channels,            # 3 (RGB)
            self.temporal_patch_size,    # 2
            self.patch_size,             # 16
            self.patch_size              # 16
        )
        # Shape: [N, 3, 2, 16, 16]
        
        # 3D卷积：一步完成切分和投影
        hidden_states = self.proj(hidden_states)
        # Shape: [N, 1152, 1, 1, 1]
        
        # 展平成 2D
        hidden_states = hidden_states.view(-1, self.embed_dim)
        # Shape: [N, 1152]
        
        return hidden_states
```

#### **为什么用 Conv3d？**

1. **高效实现**：一次卷积完成切分+投影
2. **支持视频**：时间维度也能 patch 化
3. **无重叠切分**：stride = kernel_size

### 4.4 2D Patch vs 3D Patch

#### **2D Patch（图像）**

```python
# 单张图片
image = [224, 224, 3]

# Patch化
patch_size = 16
num_patches_h = 224 / 16 = 14
num_patches_w = 224 / 16 = 14
total_patches = 14 × 14 = 196

# 每个patch
patch_shape = [16, 16, 3]
patch_dim = 16 × 16 × 3 = 768
```

#### **3D Patch（视频）**

```python
# 视频
video = [8帧, 224, 224, 3]

# Patch化
temporal_patch_size = 2
spatial_patch_size = 16

num_patches_t = 8 / 2 = 4
num_patches_h = 224 / 16 = 14
num_patches_w = 224 / 16 = 14
total_patches = 4 × 14 × 14 = 784

# 每个patch
patch_shape = [2, 16, 16, 3]
patch_dim = 2 × 16 × 16 × 3 = 1536
```

### 4.5 具体数值示例

#### **Qwen3VL-8B 配置**

```python
patch_size = 14              # 空间 patch 大小
temporal_patch_size = 2      # 时间 patch 大小
embed_dim = 1152             # 嵌入维度
spatial_merge_size = 2       # 后续合并大小
```

#### **单张图片处理**

```python
# 输入：448×448 图片
image = torch.randn(1, 3, 448, 448)  # [batch, C, H, W]

# 转为 5D（加时间维度 T=1）
image_5d = image.view(1, 3, 1, 448, 448)  # [B, C, T, H, W]

# Patch化
patches = patch_embed(image_5d)

# 计算patch数量
num_patches_t = 1 / 2 = 1（向上取整）
num_patches_h = 448 / 14 = 32
num_patches_w = 448 / 14 = 32
total_patches = 1 × 32 × 32 = 1024

print(patches.shape)  # [1024, 1152]
```

#### **视频处理**

```python
# 输入：8帧 224×224 视频
video = torch.randn(1, 3, 8, 224, 224)  # [B, C, T, H, W]

# Patch化
patches = patch_embed(video)

# 计算patch数量
num_patches_t = 8 / 2 = 4
num_patches_h = 224 / 14 = 16
num_patches_w = 224 / 14 = 16
total_patches = 4 × 16 × 16 = 1024

print(patches.shape)  # [1024, 1152]
```

### 4.6 Patch 的优势与挑战

#### **✅ 优势**

1. **全局建模**：Self-Attention 让所有 patches 都能交互
2. **可扩展性**：容易处理不同分辨率（通过位置编码插值）
3. **统一架构**：图像和文本都变成 token 序列
4. **并行性好**：所有 patches 可以并行处理

#### **⚠️ 挑战**

1. **计算代价高**：Self-Attention 是 O(n²)，patch 越多越慢
   - **解决方案**：Spatial Merge、Window Attention
2. **丢失细节**：Patch 内部的空间关系被忽略
   - **解决方案**：使用更小的 patch_size（但会增加计算）
3. **需要大数据**：ViT 在小数据集上表现不如 CNN
   - **解决方案**：使用预训练模型

---

## 5. 视觉 Token 在 LLM 中的实际占用

### 5.1 核心答案

**问题**：图片输入到 LLM 中占的 token 数就是 patch 数吗？

**答案**：**不等于原始 Patch 数，通常是 Patch 数的 1/4**（使用 Spatial Merge）

### 5.2 完整转换链路

以 Qwen3VL 处理 448×448 图像为例：

#### **步骤 0: 原始图像**
```python
image = [448, 448, 3]  # RGB图像
```

#### **步骤 1: Patch Embedding**
```python
patch_size = 14
num_patches_h = 448 / 14 = 32
num_patches_w = 448 / 14 = 32
total_patches = 32 × 32 = 1024

# 每个patch嵌入到 1152 维
vision_tokens_after_patch = [1024, 1152]
```

#### **步骤 2: Spatial Merge（关键！）**
```python
spatial_merge_size = 2  # Qwen3VL配置

# 相邻的 2×2=4 个patches合并成1个token
merged_h = 32 / 2 = 16
merged_w = 32 / 2 = 16
merged_tokens = 16 × 16 = 256

# 每个合并后的token维度变大
merge_hidden_size = 1152 × (2×2) = 4608
vision_tokens_after_merge = [256, 4608]
```

#### **步骤 3: Projection（投影到语言空间）**
```python
# 通过MLP投影到语言模型维度
language_hidden_size = 2560  # 8B模型

vision_tokens_final = [256, 2560]
```

#### **步骤 4: 插入到文本序列**
```python
# 最终在LLM中占用的token数
LLM_vision_tokens = 256  ← 答案！
```

### 5.3 计算公式

#### **通用公式**

```python
# 原始patch数量
num_patches = (H / patch_size) × (W / patch_size)

# LLM中的视觉token数量
llm_vision_tokens = num_patches / (spatial_merge_size²)
```

#### **不同图像尺寸的对比**

| 图像尺寸 | Patch Size | 原始Patches | Spatial Merge | **LLM中的Tokens** |
|---------|-----------|------------|---------------|------------------|
| 224×224 | 14 | 16×16 = 256 | 2 | 256/4 = **64** |
| 448×448 | 14 | 32×32 = 1024 | 2 | 1024/4 = **256** |
| 672×672 | 14 | 48×48 = 2304 | 2 | 2304/4 = **576** |
| 896×896 | 14 | 64×64 = 4096 | 2 | 4096/4 = **1024** |

### 5.4 代码证据

#### **1. Spatial Merge 配置**

```python
# visionmodel.py, line 94-95
self.merge_hidden_size = projection_config.ffn_hidden_size
self.square_merge_size = self.merge_hidden_size // embed_dim

# 例如：ffn_hidden_size = 4608, embed_dim = 1152
# square_merge_size = 4608 / 1152 = 4 = 2²
# 说明是 2×2 的merge
```

#### **2. LLM 中的视觉 Token 计算**

```python
# model.py, get_rope_index 函数
llm_grid_t, llm_grid_h, llm_grid_w = (
    t.item(),
    h.item() // spatial_merge_size,  # ← 除以 spatial_merge_size
    w.item() // spatial_merge_size,
)

# 视觉token数量
vision_token_count = llm_grid_t * llm_grid_h * llm_grid_w

# 例如：单张图片 (t=1, h=32, w=32, merge_size=2)
# vision_token_count = 1 × (32/2) × (32/2) = 1 × 16 × 16 = 256
```

### 5.5 视频的情况

#### **示例：8帧 224×224 视频**

```python
# 原始数据
video = [8, 224, 224, 3]

# 步骤1: 3D Patch Embedding
temporal_patch_size = 2
spatial_patch_size = 14

num_patches_t = 8 / 2 = 4
num_patches_h = 224 / 14 = 16
num_patches_w = 224 / 14 = 16
total_patches = 4 × 16 × 16 = 1024

# 步骤2: Spatial Merge（只merge空间，不merge时间）
spatial_merge_size = 2

merged_t = 4  # 时间维度不变
merged_h = 16 / 2 = 8
merged_w = 16 / 2 = 8
merged_tokens = 4 × 8 × 8 = 256

# LLM中的token数
llm_vision_tokens = 256
```

**注意**：Qwen3VL 使用时间戳方式，每帧单独处理：

```python
# Qwen3VL的实际方式（使用时间戳）
# 序列：<t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end> ...

# 每帧token数
tokens_per_frame = (224/14)² / 4 = 16² / 4 = 64

# 8帧总token数
total_tokens = 8 × 64 = 512
```

### 5.6 为什么要做 Spatial Merge？

#### **原因 1: 减少计算量**

```python
# 没有merge
num_tokens = 1024
attention_cost = O(1024²) = 1,048,576

# 有merge (2×2)
num_tokens = 256
attention_cost = O(256²) = 65,536

# 减少了 16 倍！✨
```

#### **原因 2: 更好的语义表示**

```python
# 单个patch: 14×14像素，可能只包含部分物体
# 合并后: 28×28像素，更容易包含完整的局部特征
```

#### **原因 3: 对齐语言模型的token粒度**

```python
# 文本token: 通常是一个词或子词
# 图像token: merge后的patch更接近"视觉词"的概念
```

### 5.7 实际序列示例

```python
# 输入
text = "这张图片中有什么？<image>这是一只可爱的猫。"
image = [448, 448, 3]

# Token化后
tokens = [
    "这", "张", "图片", "中", "有", "什么", "？",  # 7个文本tokens
    "<vision_start>",                             # 1个特殊token
    <256 个视觉 tokens>,                          # 256个图像tokens
    "<vision_end>",                               # 1个特殊token
    "这", "是", "一", "只", "可爱", "的", "猫", "。" # 8个文本tokens
]

# 总长度 = 7 + 1 + 256 + 1 + 8 = 273 tokens
```

**关键发现**：
- 原始 patches：1024 个
- **LLM 中实际占用**：256 个 ✅
- 减少比例：1024/256 = **4倍**（2×2 merge）

### 5.8 可视化对比

```
原始Patches: 32×32 = 1024
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│P1│P2│P3│P4│P5│P6│P7│P8│P9│..│  │  │  │  │  │32│
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

          ↓  Spatial Merge (2×2)

LLM Tokens: 16×16 = 256
┌─────┬─────┬─────┬─────┬─────┬─────┬────┐
│ T1  │ T2  │ T3  │ T4  │ ... │     │ 16 │
│(4个)│(4个)│(4个)│(4个)│     │     │    │
├─────┼─────┼─────┼─────┼─────┼─────┼────┤
│     │     │     │     │     │     │    │
└─────┴─────┴─────┴─────┴─────┴─────┴────┘

每个LLM Token = 2×2 = 4个原始patches的信息
```

### 5.9 不同模型对比

| 模型 | Patch Size | Spatial Merge | 448×448图像的Token数 |
|------|-----------|---------------|-------------------|
| **Qwen3VL** | 14 | 2×2 | 256 |
| **Qwen2VL** | 14 | 2×2 | 256 |
| **LLaVA-1.5** | 14 | 无 | 1024 |
| **CLIP ViT** | 14 | 无 | 1024 |
| **Qwen-VL** | 14 | 无 | 1024 |

**结论**：新版多模态模型普遍使用 Spatial Merge 来减少视觉 token 数量！

### 5.10 总结

#### **核心要点**

1. **原始 Patch 数** ≠ **LLM 中的 Token 数**
2. **Qwen3VL 的转换**：Patches → Spatial Merge (÷4) → LLM Tokens
3. **实际占用公式**：
   ```python
   LLM_tokens = (H/patch_size) × (W/patch_size) / (merge_size²)
   ```

4. **具体例子（448×448 图像）**：
   ```
   Patches:     1024 个
   LLM Tokens:  256 个  ← 实际占用
   比例:        1:4
   ```

#### **设计原因**

✅ **效率**：减少 Attention 计算量（O(n²)）  
✅ **语义**：合并后的 token 包含更丰富的局部信息  
✅ **灵活**：可以处理更高分辨率的图像  
✅ **性能**：实验证明这样效果更好  

---

## 附录

### A. 关键术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| Patch | Patch | 图像的小方块切片 |
| Spatial Merge | Spatial Merge | 空间合并，将相邻patches合并 |
| MRoPE | Multi-modal RoPE | 多模态旋转位置编码 |
| DeepStack | DeepStack | 多层视觉特征注入机制 |
| ModuleSpec | ModuleSpec | 模块规范，延迟实例化 |
| Pipeline Parallel | Pipeline Parallelism | 流水线并行 |
| Tensor Parallel | Tensor Parallelism | 张量并行 |

### B. 参考资源

- Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
- Qwen3VL 论文: [待补充]
- Vision Transformer 论文: https://arxiv.org/abs/2010.11929
- RoFormer (RoPE): https://arxiv.org/abs/2104.09864

---

**文档生成时间**: 2025年10月28日  
**文档版本**: v1.0  
**作者**: AI Assistant based on Pai-Megatron-Patch codebase analysis

