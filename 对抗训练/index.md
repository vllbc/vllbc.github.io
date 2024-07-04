# 对抗训练

# Min-Max公式

$$
\min_{\theta} \mathbb{E}_{(x,y) \sim \mathbb{D}}U[\max_{r_{adv}\in \mathbb{S}}L(\theta,x+r_{adv},y)]

$$

1. 内部max是为了找到worst-case的扰动，也就是攻击，其中， $L$为损失函数， $\mathbb{S}$ 为扰动的范围空间。
2. 外部min是为了基于该攻击方式，找到最鲁棒的模型参数，也就是防御，其中 $\mathbb{D}$ 是输入样本的分布。
简单理解就是**在输入上进行梯度上升(增大loss)，在参数上进行梯度下降(减小loss)**

# 加入扰动后的损失函数

$$
\min_{\theta} -\log P(y |x+r_{adv};\theta)

$$
那扰动要如何计算呢？Goodfellow认为，**神经网络由于其线性的特点，很容易受到线性扰动的攻击。**
# Fast Gradient Sign Method (FGSM)
$$
r_{adv} =  \epsilon \cdot sgn(\nabla_{x}L(\theta,x,y))
$$
# FGM
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20240220215658.png)
## 代码
代码来自[1]，注意的是一般扰动加在了embedding矩阵上，相当于x+r。
```python
import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name为embedding矩阵参数对应名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

使用时：

```python
# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```
# 参考文献

1. [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/91269728)

