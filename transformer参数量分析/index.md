# transformer参数量分析

进入大模型时代，基本上所有大模型都使用 decoder 部分，因此本文只分析 decoder 部分的参数量。
Transformer 的 decoder 每一层由 attention 和 mlp 组成，一般有 l 层。

## Self-attention
Self-attention 层由 $W_{Q}$ 、$W_{K}$、$W_{V}$ 和输出矩阵 $W_{O}$ 和它们的偏置组成，权重矩阵的形状为 $[h,h]$，偏置形状为 $[h]$，则 self-attention 部分的参数量为 $4h^2+4h$


## MLP

MLP 由 2 个线性层构成，第一个线性层将维度从 h 变为 4 h，第二个将维度由 4 h 变为 h，第一个权重矩阵形状为 $[h,4h]$，偏置为 $[4h]$，第二个形状为 $[4h,h]$，偏置为 $[h]$，则参数量为 $8h^2+5h$

## Layer norm
在 self-attention 和 mlp 中都存在 layer norm，有 2 个可训练参数：缩放参数 $\gamma$ 和平移参数 $\beta$，形状都是 $[h]$，2 个 layer norm 的参数量为 4 h。

## 词嵌入
词嵌入矩阵的参数量和词表大小 V 有关，而且输入和输出一般公用一个矩阵，因此参数量为 $Vh$

## 位置编码

如果是可训练式的位置编码，则占据一定的参数量，否则不占参数量


## 总参数

综上所述，transformer 一个层的参数量为 $12h^2+13h$，l 层就是 $l(12h^2+13h)$，再加上词嵌入矩阵，总参数量为 $l(12h^2+13h)+Vh$，h 较大时，可以忽略一次项，近似为 $12lh^2$

| 实际参数量 | 隐藏维度h | 层数l | 12lh^2         |
| ----- | ----- | --- | -------------- |
| 6.7B  | 4096  | 32  | 6,442,450,944  |
| 13.0B | 5120  | 40  | 12,582,912,000 |
| 32.5B | 6656  | 60  | 31,897,681,920 |
| 65.2B | 8192  | 80  | 64,424,509,440 |
表来自 [分析 transformer 模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
## 参考

- [# 分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
- [Transformer Math (Part 1) - Counting Model Parameters](https://michaelwornow.net/2024/01/18/counting-params-in-transformer)

