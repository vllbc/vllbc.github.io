# hf2megatron

需要：
- rice vit的hf权重
- llm的hf权重
- 转换代码


| 对比项          | Qwen2.5 ViT      | RiCE ViT         |
| ------------ | ---------------- | ---------------- |
| Attention 类型 | Window Attention | Global Attention |
| Class Token  | 无                | 使用 Class Token   |
| 激活函数         | SiLU             | GeLU             |
| layer数       | 32               | 24               |


