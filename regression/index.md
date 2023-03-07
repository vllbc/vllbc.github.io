# regression



# 一个线性回归的神经网络模型

```python
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
```


```python
x = torch.unsqueeze(torch.linspace(-10,10,100),dim=1)
y = x*2+10+torch.rand(x.size())
# scatter = go.Scatter(x = torch.squeeze(x),y = torch.squeeze(y),mode = 'markers')
# fig = go.Figure(scatter)
# fig.show()
```


```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(1,10)
        self.linser = torch.nn.Linear(10,1)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        out = self.linser(x)
        return out
x = Variable(x)
y = Variable(y)
net = Net()
net

```




    Net(
      (hidden): Linear(in_features=1, out_features=10, bias=True)
      (linser): Linear(in_features=10, out_features=1, bias=True)
    )




```python
net.parameters()
```




    <generator object Module.parameters at 0x000001DCE4384570>




```python
loss_fun = torch.nn.MSELoss()
optim = torch.optim.SGD(net.parameters(),lr=1e-2)
print(loss_fun,optim)
```

    MSELoss() SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.01
        momentum: 0
        nesterov: False
        weight_decay: 0
    )




```python
for i in range(1000):
    predict = net(x)
    loss = loss_fun(predict,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i%50 == 0:
        print(f"损失率为{loss.item():.4f}")
#         scatter1 = go.Scatter(x=x.squeeze(1).data.numpy(),y=y.squeeze(1).data.numpy(),mode='markers')
#         scatter2 = go.Scatter(x=x.squeeze(1).data.numpy(),y=predict.squeeze(1).data.numpy(),mode='lines')
#         fig = go.Figure([scatter1,scatter2])
#         fig.update_layout(title=f'损失率为{loss.item()}')
#         fig.show()
    
    
```

    损失率为237.3235
    损失率为18.0279
    损失率为12.1344
    损失率为8.4456
    损失率为6.4861
    损失率为5.1898
    损失率为4.2126
    损失率为3.4631
    损失率为2.8457
    损失率为2.3588
    损失率为1.8926
    损失率为1.5829
    损失率为1.3023
    损失率为1.0384
    损失率为0.9485
    损失率为0.8678
    损失率为0.7937
    损失率为0.6795
    损失率为0.5726
    损失率为0.5546



```python
arrays = torch.Tensor([5]).unsqueeze(1)
net(arrays)
```




    tensor([[21.3448]], grad_fn=<AddmmBackward>)
