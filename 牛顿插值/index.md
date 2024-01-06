# 牛顿插值



# 牛顿插值

## 差商
定义：设 $f(x)$ 在互异节点$x_i$处的函数值为$f_i,  i=0,1,\dots,n$，称$f[x_i,x_j]=\frac{f_i-f_j}{x_i-x_j}$为$f(x)$关于节点$x_i,x_j$的一阶差商，$f[x_i,x_j,x_k]=\frac{f[x_i,x_j]-f[x_j,x_k]}{x_i-x_k}$为$f(x)$关于$x_i,x_j,x_k$的二阶差商，以此类推k阶差商：

$$
f[x_0,x_1,\dots ,x_k-1,x_k] = \frac{f[x_0,x_1,\dots ,x_{k-1}]-f[x_1,\dots ,x_k]}{x_0-x_k}
$$

## 牛顿基本插值

$$
\begin{aligned}
&N_n(x) = a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\dots +a_n(x-x_0)(x-x_1)\dots(x-x_{n-1})\\\\
&= f_0 + \sum_{k=1}^{k-1}f[x_0,x_1,\dots,x_k]\omega_k(x) \\\\
&其中\omega_k(x)=\prod_{j=0}^{k-1}(x-x_j)
\end{aligned}
$$

## 差分

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/%E5%B7%AE%E5%88%86.jpg)

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/%E5%B7%AE%E5%95%86%E4%B8%8E%E5%B7%AE%E5%88%86.png)

## 优缺点

- 优点：计算简单
- 缺点：和拉格朗日插值方法相同，插值曲线在节点处有尖点，不光滑，节点处不可导


## 代码
```python
# 牛顿插值法
import numpy as np
import matplotlib.pyplot as plt
 
#递归求差商
def get_diff_quo(xi, fi):
    if len(xi) > 2 and len(fi) > 2:
        return (get_diff_quo(xi[:len(xi)-1], fi[:len(fi)-1]) - get_diff_quo(xi[1:len(xi)], fi[1:len(fi)])) / float(xi[0] - xi[-1])
    return (fi[0]-fi[1]) / float(xi[0]-xi[1])
 
#求w，使用闭包函数
def get_w(i, xi):
    def wi(x):
        result = 1.0
        for j in range(i):
            result *= (x - xi[j])
        return result
    return wi
 
#做插值
def get_Newton(xi, fi):
    def Newton(x):
        result = fi[0]
        for i in range(2, len(xi)):
            result += (get_diff_quo(xi[:i], fi[:i]) * get_w(i-1, xi)(x))
        return result
    return Newton
 
#已知结点
xn = [i for i in range(-50, 50, 10)]
fn = [i**2 for i in xn]
 
#插值函数
Nx = get_Newton(xn, fn)
 
#测试用例
tmp_x = [i for i in range(-50, 51)]
tmp_y = [Nx(i) for i in tmp_x]
 
#作图
plt.plot(xn, fn, 'r*')
plt.plot(tmp_x, tmp_y, 'b-')
plt.title('Newton Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
