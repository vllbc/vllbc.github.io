# flops分析

> FLOPs, floating point operations, 表示浮点数运算次数，衡量了计算量的大小。
如何计算矩阵乘法的 FLOPs 呢？
对于 $A\in R^{1\times n},B\in R^{n\times1}$ ,计算 $AB$ 需要进行 $n$ 次乘法运算和 $n$ 次加法运算，共计 $2n$ 次浮点数运算，需要 $2n$ 的 FLOPs。对于 $A\in R^{m\times n},B\in R^{n\times p}$ ,计算 $AB$ 需要的浮点数运算次数为 $2mnp$ 。

在一次训练迭代中，假设输入数据的形状为 $[b,s]$ 。我们先分析 self-attention 块的计算，计
算公式如下：

$$Q=xW_Q,K=xW_K,V=xW_V$$
$$x_{out}=softmax(\frac{QK^T}{\sqrt{h}})\cdot V\cdot W_o+x$$

1. 计算 $Q,K,V:$ 矩阵乘法的输入和输出形状为 $[b,s,h]\times[h,h]\to[b,s,h]$ 。计算量为
$3*2bsh^2=6bsh^2$ 。

$2.QK^T$ 矩阵乘法的输入和输出形状为
$[b,head\_num,s,per\_head\_hidden\_size]$
$\times[b,head\_num,per\_head\_hidden\_size,s]\rightarrow[b,head\_num,s,s]$
。计算量为 $2bs^2h$ 。
3. 计算在 $V$ 上的加权 $score\cdot V$, 矩阵乘法的输入和输出形状为
$[b,head\_num,s,s]\times[b,head\_num,s,per\_head\_hidden\_size]$
。计算量为 $2bs^2h$ 。
4. Attention 后的线性映射，矩阵乘法的输入和输出形状为 $[b,s,h]\times[h,h]\to[b,s,h]$ 。计
算量为 $2bsh^2$ 。
接下来分析 MLP 块的计算，计算公式如下：


$$x=f_{gelu}(x_{out}W_1)W_2+x_{out}$$

1. 第一个线性层，矩阵乘法的输入和输出形状为 $[b,s,h]\times[h,4h]\to[b,s,4h]$。计算量
为 $8bsh^2$ 。
2. 第二个线性层，矩阵乘法的输入和输出形状为 $[b,s,4h]\times[4h,h]\to[b,s,h]$。计算量
为 $8bsh^2$ 。
将上述计算量相加，得到每个 transformer 层的计算量大约为 $24bsh^2+4bs^2h$ 。
此外，另一个计算量的大头是 logits 的计算，将隐藏向量映射为词表大小。矩阵乘法的输入和
输出形状为 $[b,s,h]\times[h,V]\to[b,s,V]$ ,计算量为 $2bshV$ 。
因此，对于一个 $l$ 层的 transformer 模型，输入数据形状为 $[b,s]$ 的情况下，一次训练迭代的
计算量为 $l*(24bsh^2+4bs^2h)+2bshV$ 。

## 计算量与参数量的关联

当隐藏维度 $h$ 比较大，且远大于序列长度 $s$ 时，我们可以忽略一次项，计算量可以近似为
$24bsh^2*l$ 。前面提到当模型参数量为 $12lh^2$, 输入的 tokens 数为 $bs$ ,存在等式
$\frac{24bsh^2l}{12h^2\times bs}=2$。我们可以近似认为：在一次前向传递中，对于每个 token，每个模型参数，需要进行 2 次浮点数运算，即一次乘法法运算和一次加法运算。
一次训练迭代包含了前向传递和后向传递，后向传递的计算量是前向传递的 2 倍。因此，前向传递+后向传递的系数=1+2=3 。一次训练迭代中，对于每个 token，每个模型参数，需要进行 2*3=6 次浮点数运算。
接下来，我们可以估计训练 GPT 3-175 B 所需要的计算量。对于 GPT 3，每个 token，每个参数进行了 6 次浮点数运算，再乘以参数量和总 tokens 数就得到了总的计算量。GPT 3 的模型参数量为 174600 $M$ ,训练数据量为 $300B$ tokens。

$$6\times174600\times10^6\times300\times10^9=3.1428\times10^{23}flops$$

## 训练时间

$$
训练时间=\frac{8 * tokens数 *模型参数量}{GPU数 * GPU峰值flops*GPU利用率}
$$

也就是训练时的 flops / gpu 的 flops。8 是前向传播、后向传播、[Activation checkpointing](Activation%20checkpointing.md) 。前向传播系数为 1，后向传播是前向传播计算量的 2 倍，再加上在反向传播时需要前向传播一次，因此总系数为 4，1 个参数需要 2 次浮点数运算，这就是 8 怎么来的。 
