# Muon

Muon 算法流程如下图所示：

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250718103437.png)

其中最主要的部分是 NewtonSchulz 5 算法，流程如下：

```python

def newtonschulz5(G, steps=5, eps=1e-7):
	assert G.ndim == 2
	
	a, b, c = (3.4445, -4.7750, 2.0315)
	
	X = G.bfloat16()
	
	X /= (X.norm() + eps)
	
	if G.size(0) > G.size(1):
	
	X = X.T
	
	for _ in range(steps):
	
	A = X @ X.T
	
	B = b * A + c * A @ A
	
	X = a * X + B @ X
	
	if G.size(0) > G.size(1):
	
	X = X.T
	
	return X
```

这个算法的作用是将 G 近似为一个最接近他的半正交矩阵，即：

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250718104749.png)

> 对于经验动机，我们观察到，基于手动检查，SGD-momentum 和 Adam 对基于 Transformer 的神经网络中的 2D 参数产生的更新通常具有非常高的条件数。也就是说，它们几乎是低秩矩阵，所有神经元的更新仅由几个方向主导。我们推测正交化有效地增加了其他“罕见方向”的规模，这些方向在更新中幅度较小，但对学习仍然很重要。


## Muon in Moonlight
##   QK-clip
