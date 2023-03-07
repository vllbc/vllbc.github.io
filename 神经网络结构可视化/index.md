# 神经网络结构可视化



# pytorch神经网络结构可视化

参考：[https://zhuanlan.zhihu.com/p/220403674](https://zhuanlan.zhihu.com/p/220403674)



## torchviz

```python
from torchviz import make_dot
make_dot(model(torch.from_numpy(X_train[0].reshape(1, -1)).float()), params=dict(model.named_parameters()))
```

这是一个简单的例子，别的都可以套用，model是神经网络模型的实例。
