# start


# 开始！

用`import torch`导入`pytorch`库

不要打成`import pytorch`哦~

下面是我的学习记录：

```python
import torch#导入模块
```


```python
x = torch.rand(5,3)#生成随机张量
x
```




    tensor([[0.8241, 0.9623, 0.8265],
            [0.8875, 0.6775, 0.0678],
            [0.8438, 0.5565, 0.0824],
            [0.7778, 0.7368, 0.5326],
            [0.6096, 0.5767, 0.5788]])




```python
x = x.new_ones(5,3,dtype=torch.double)#生成值为1的张量，并定义数据类型
x
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)




```python
x = torch.randn_like(x,dtype=torch.float)#改变数据类型，不改变数
x
```




    tensor([[-1.1167,  0.1029,  0.5996],
            [-1.2067,  1.4284, -1.0661],
            [-0.0320, -0.3634,  1.4178],
            [ 0.2564, -1.0210, -2.3204],
            [-0.0476, -0.2605, -0.1166]])




```python
x.size()#获取尺寸
```




    torch.Size([5, 3])




```python
y = torch.rand(5,3)
```


```python
torch.add(x,y)#也可以写成x+y
```




    tensor([[-0.8348,  0.5407,  0.6893],
            [-0.9977,  1.4544, -0.6345],
            [ 0.7664, -0.3510,  2.3684],
            [ 0.4159, -0.4354, -1.6096],
            [ 0.0588, -0.1941,  0.5014]])




```python
result = torch.empty(5,3)#空张量
torch.add(x,y,out = result)#把运算结果储存在result里
```




    tensor([[-0.8348,  0.5407,  0.6893],
            [-0.9977,  1.4544, -0.6345],
            [ 0.7664, -0.3510,  2.3684],
            [ 0.4159, -0.4354, -1.6096],
            [ 0.0588, -0.1941,  0.5014]])




```python
x,x[:,1]#类似于numpy的切片操作,取第二列
```




    (tensor([[-1.1167,  0.1029,  0.5996],
             [-1.2067,  1.4284, -1.0661],
             [-0.0320, -0.3634,  1.4178],
             [ 0.2564, -1.0210, -2.3204],
             [-0.0476, -0.2605, -0.1166]]),
     tensor([ 0.1029,  1.4284, -0.3634, -1.0210, -0.2605]))




```python
x = torch.rand(4,4)
y = x.view(16)#类似于numpy的resize()但用法不太相同
z = x.view(-1,8)
x.size(),y.size(),z.size()
```




    (torch.Size([4, 4]), torch.Size([16]), torch.Size([2, 8]))




```python
x = torch.rand(1)
print(x)
print(x.item())#取值
```

    tensor([0.5160])
    0.5160175561904907



```python
import numpy as np
```


```python
a = torch.ones(5)
a
```




    tensor([1., 1., 1., 1., 1.])




```python
b=a.numpy()#将张量转换为numpy的array
b
```




    array([1., 1., 1., 1., 1.], dtype=float32)




```python
a.add_(1) #a自加1，b也跟着改变
a,b
```




    (tensor([2., 2., 2., 2., 2.]), array([2., 2., 2., 2., 2.], dtype=float32))




```python
a = np.ones(5)
b=torch.from_numpy(a)#将array转换为张量
np.add(a,1,out=a)
a,b
```




    (array([2., 2., 2., 2., 2.]),
     tensor([2., 2., 2., 2., 2.], dtype=torch.float64))




```python
x = torch.ones(2,2,requires_grad=True)#requires_grad参数用于说明当前量是否需要在计算中保留对应的梯度信息以线性回归为例，为了得到最合适的参数值，我们需要设置一个相关的损失函数，根据梯度回传的思路进行训练。
x
```




    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)




```python
y = x+2
y
```




    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)




```python
y.grad_fn#用于指导反向传播，我现在也不太懂
```




    <AddBackward0 at 0x25b91b2c710>




```python
z = y*y*3
out = torch.mean(z)
z,out
```




    (tensor([[27., 27.],
             [27., 27.]], grad_fn=<MulBackward0>),
     tensor(27., grad_fn=<MeanBackward0>))




```python
a = torch.randn(2,2)
print(a)
print(a.requires_grad)
a.requires_grad_(True)#修改requires_grad的值
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)
```

    tensor([[-0.6831,  1.5310],
            [-0.5836,  0.4117]])
    False
    True
    <SumBackward0 object at 0x0000025B91B39828>

和`numpy`的互相转换:

```python
import numpy as np
import torch
a = torch.Tensor([])
b=a.numpy()#a为张量
a = np.arange.randn()
b = torch.from_numpy(a)#a为array
```






