# gather和scatter

## gather 
参数：

- **input** ([Tensor](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/tensors.html%23torch.Tensor)) – the source tensor
- **dim** ([int](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23int)) – the axis along which to index
- **index** (_LongTensor_) – the indices of elements to gather
- **out** ([Tensor](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/tensors.html%23torch.Tensor)_,__optional_) – the destination tensor
- **sparse_grad** ([bool](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23bool)_,optional_) – If `True`, gradient w.r.t. `input` will be a sparse tensor.
> gather操作是scatter操作的**逆操作**，如果说scatter是根据index和src求self(_input_)，那么gather操作是根据self(input)和index求src。具体来说gather操作是根据index指出的索引，沿dim指定的轴收集input的值。

```python
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```
对于gather操作来说，有三个约束需要满足：

（1）对于所有的维度d != dim，有input.size(d) == index.size(d)，对于维度dim来说，有index.size(d) >= 1；
（2）张量out的维度大小必须和index相同；
（3）和scatter一样，index中的索引值必须在input.size(dim)范围内。
### example 
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219172408.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219172502.png)
### code example 
```python
import torch

t = torch.Tensor([[1, 2], 
                  [3, 4]])
# t = 1 2
#     3 4
index = torch.LongTensor([[0, 0], 
                          [1, 0]])
# index = 0 0
#         1 0
# dim = 0 : [[1,2],[3,2]]
# dim = 1 : [[1,1],[4,3]]

# index = 0
#         1
# dim = 0 : [[1],[3]]
# dim = 1 : [[1],[4]]

# index = 0 1
# dim = 0 : [[1, 4]]
# dim = 1 : [[1, 2]]
```
## scatter

```
Writes all values from the tensor into at the indices specified in the tensor. For each value in , its output index is specified by its index in for and by the corresponding value in for .`src``self``index``src``src``dimension != dim``index``dimension = dim`

For a 3-D tensor, is updated as:`self`


```
参数：
- **dim** ([int](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23int)) – the axis along which to index
- **index** (_LongTensor_) – the indices of elements to scatter, can be either empty or the same size of src. When empty, the operation returns identity
- **src** ([Tensor](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/tensors.html%23torch.Tensor)) – the source element(s) to scatter, incase value is not specified
- **value** ([float](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/functions.html%23float)) – the source element(s) to scatter, incase src is not specified
```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

看了上面这个操作就理解了。
由此可以得出以下约束：
1. 张量self，张量index和张量src的维度数量必须相同（即三者的.dim()必须相等，注意不是维度大小）；
2. 对于每一个维度d，有index.size(d)<=src.size(d)；
3. 对于每一个维度d，如果d!=dim，有index.size(d)<=self.size(d)；
对于index也有一些约束： 
1. 张量index中的每一个值大小必须在[0, self.size(dim)-1]之间；
2. 张量index沿dim维的那一行中所有值都必须是唯一的（弱约束，违反不会报错，但是会造成没有意义的操作）。

### example
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219154916.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219154947.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219155034.png)
### code example 
```python
import torch

a = torch.arange(10).reshape(2,5).float()
print(f"a: \n{a}")
b = torch.zeros(3, 5)
print(f"b: \n{b}")
b_= b.scatter(dim=0, 
              index=torch.LongTensor([[1, 2, 1, 1, 2],
                                      [2, 0, 2, 1, 0]]),
              src=a)
print(f"b_: \n{b_}")
 
# tensor([[0, 6, 0, 0, 9],
#        [0, 0, 2, 8, 0],
#        [5, 1, 7, 0, 4]])
```


## scatter_add_
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219155034.png)
这个函数和scatter基本上没有任何区别，区别在于上图中的对于self中同一位置的填入是随机的，`self[3,0]`不确定是7还是9，`self[0,1]`不确定是8还是10，但是使用scatter_add就将所有即将填入同一位置的数相加，例子如下：
### example 
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20241219165509.png)

## 参考
- https://zhuanlan.zhihu.com/p/158993858
