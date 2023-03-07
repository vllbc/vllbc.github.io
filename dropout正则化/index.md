# Dropout正则化



# Dropout

在标准dropout正则化中，通过按保留（未丢弃）的节点的分数进行归一化来消除每一层的偏差。换言之，每个中间激活值h以保留概率概率p由随机变量替换

$$
h^{'}=
\begin{cases}
0, \quad 概率为1-p \\\\
\frac{h}{p}, \quad 概率为p
\end{cases}
$$

注意期望不要变，即

$$
E[h^{'}] = (1-p)*0 + p *\frac{h}{p} = h
$$
![](image/Pasted%20image%2020220830153314.png)

**注意**：正则项（Dropout）只在训练过程中使用，因为其会影响模型参数的更新  
所以在推理过程中，丢弃法直接返回输入。

## 代码

```python
import torch
from torch import nn 
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data

def dropout_layer(X,dropout):
    assert 0<=dropout <= 1
    # 如果keep_prob设置为1，全部元素被保留
    if dropout == 1:
        return X
    # 如果keep_prob设置为0，全部元素被丢弃
    if dropout == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < dropout).float() #使用mask而不是直接置零是为了提高计算效率
    return mask * X/(dropout)

```
