# Activation checkpointing

## 为什么存储激活值？
首先回顾为什么要存储激活值。
简单来说，模型参数是根据导数更新的。为了有效地计算这些导数，必须缓存某些张量。激活内存是这些缓存张量的内存成本。
具体来说，以 $f$ 是矩阵乘法运算：

$$y=f(x)=W\cdot x$$
$W$是一个可学习的权重矩阵。假设我们有关于 早期反向传播阶段的手头输出，$\frac{\partial L}{\partial y}$,我们
需要计算 两个额外的梯度：
1.关于$W$,以便我们可以更新此权重。
2.关于$x$,这样我们就可以继续反向传播算法 到产生的任何作$x$

前导数是 

$$\frac{\partial L}{\partial W}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial W}=\frac{\partial L}{\partial y}\times x$$

而后者的导数是

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\cdot W$$


因此，如下图所示，我们需要缓存输入张量 $x$ 为了能够计算我们关心的导数。节省的成本 $x$ 是激活 memory。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250712010846.png)

我们发现只需要保存 x，而对于 y 不需要保存，也就是说我们只保存反向传播时绝对需要的激活值，其它的临时变量要立即释放。

## MLP


```python
class MLP(nn.Module):
    """
    Basic MLP (multi-layer perceptron) layer with optional Dropout.
    """

    def __init__(
        self,
        d_model: int,
        act_fn: nn.Module,
        dropout_prob: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        factory_kwargs = {"device": device, "dtype": dtype}

        self.lin_0 = nn.Linear(self.d_model, 4 * self.d_model, **factory_kwargs)
        self.lin_1 = nn.Linear(4 * self.d_model, self.d_model, **factory_kwargs)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lin_0(inputs)
        x = self.act_fn(x)
        x = self.lin_1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
```

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250712012621.png)


Maybe saved 的是否 是根据激活函数的类型来的：
如果是 GELU 激活函数

$$y=\frac x2\times\tanh\left(\sqrt{\frac2\pi}\left(x+.044715x^3\right)\right)$$
这时候计算梯度的话还是需要输入 x，这是需要存储的情况

而如果是一些特殊的激活函数的话，比如 ReLU

$$y=\mathsf{ReLU}(x)=\begin{cases}x&\mathrm{if~}x>0\\0&\mathrm{if~}x<0&\end{cases}$$

梯度为：

$$\frac{dy}{dx}=\frac{d\text{ ReLU}(x)}{dx}=\begin{cases}1&\mathrm{if~}x>0\\0&\mathrm{if~}x<0&\end{cases}$$

就不需要存储。
Tanh 也是同理：

$$
\frac{dy}{dx} = \frac{d\tanh (x)}{dx} = 1-\tanh(x)^2 = 1-y^2
$$

如果想要性能，选择 GELU，想要低显存，使用 ReLU。

## Activation checkpointing

当使用该方法时，我们只需要保存一些关键的激活值，可以舍弃一些激活值从而在反向传播的过程中重新计算，通常有两种策略：

Full：我们在 Transformer 模型每一层之间的过渡点检查激活值。这通常被称为“完整”策略，因为它需要每层都进行一次前向传播，即在反向传播过程中增加了一次完整的前向传播。这种策略节省的内存最多，但在计算方面成本最高。它通常会使计算成本和时间增加高达 30-40%，这一影响非常显著。

Select：总体而言，我们可以比全面优化做得更好。那篇关于重新计算的论文的作者进行了详细分析，研究了哪些激活值的增长最大，并且以每秒浮点运算次数（FLOPS）为标准，其重新计算成本最低。结果表明，注意力计算属于这一类别，因此我们通常可以丢弃它们，而专注于对昂贵的前馈计算进行检查点设置。对于 GPT-3（1750 亿参数）模型而言，这意味着在计算成本仅增加 2.7%的情况下，激活值内存减少了 70%。

如今，大多数训练框架都使用 FlashAttention，它通过在向后传递中重新计算注意力得分和矩阵，而不是存储它们，将激活重新计算集成到其优化策略中。因此，大多数使用 FlashAttention 的人已经在使用选择性重新计算。
正如您现在所理解的，由于重新计算，激活重新计算略微增加了 FLOPS 的数量，同时显著降低了内存访问开销。
这种权衡在 GPU 等高速内存有限的硬件上特别有利，因为访问内存通常比执行计算慢。尽管涉及额外的操作，但总体效果通常是计算速度更快，内存占用更少。

## 参考
[Activation Memory: A Deep Dive using PyTorch \| Determined AI](https://www.determined.ai/blog/act-mem-2)
