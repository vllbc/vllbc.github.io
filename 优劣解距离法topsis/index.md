# 优劣解距离法Topsis


# 优劣解距离法Topsis

## 步骤
### 原始矩阵正向化
有四种指标：

- 极大型
- 极小型
- 中间型
- 区间型

正向化就是将其它的指标转化为极大型指标
极小型 -> 极大型：
中间型 -> 极大型：
区间型 -> 极大型：
### 正向化矩阵标准化

$$
z_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{n}x_{ij}^2}}
$$

### 计算得分并归一化


## 如何计算得分
### 只有一个指标时
构造评分的公式：

$$
\frac{x-min}{max-min}
$$

本质上为

$$
\frac{x-min}{(max-x)+(x-min)}
$$

即

$$
\frac{\text{x与最小值的距离}}{\text{x与最大值的距离}+\text{x与最小值的距离}}
$$

### 多个指标
这里就需要矩阵了
先把矩阵标准化

$$
z_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{n}x_{ij}^2}}
$$

定义最大值

$$
Z^+=(Z_1^+,Z_2^+, \cdots, Z_m^+) \\\\
   =(max\{z_{11},z_{21},\cdots,z_{n1}\},max\{z_{12},z_{22},\cdots,z_{n2}\},\cdots,max\{z_{1m},z_{2m},\cdots,z_{nm}\})
$$

定义第$i(i=1,2,\cdots,n)$个评价对象与最大值的距离

$$
D_i^+=\sqrt{\sum_{j=1}^m(Z_j^+-z_{ij})^2}
$$

定义第$i(i=1,2,\cdots,n)$个评价对象与最小值的距离

$$
D_i^+=\sqrt{\sum_{j=1}^m(Z_j^--z_{ij})^2}
$$

这样就可以得到第$i(i=1,2,\cdots,n)$个评价对象未归一化的得分$S_i = \frac{D_i^-}{D_i^++D_i^-}$

## 权重怎么求？
可以用层次分析法求权重，但使用层次分析法求出的权重主观性太强，因为判断矩阵依赖于专家。

较为客观的方法是熵权法。

熵权法要求标准化后的矩阵都是非负的，因此要使用max-min归一化方法。

对于某个指标来说，其概率值为$p_{ij} = \frac{z_{ij}}{\sum_{i=1}^{n}z_{ij}}$

对于第j个指标，其信息熵的计算公式为：$e_{j}=-\frac{1}{\ln n}\sum_{i=1}^{n}p_{ij}\ln (p_{ij})$

1.为什么除以$\ln n$？

H(x)最大值就是取$\ln n$，这里除以$\ln n$就是使得信息熵能始终位于0-1区间内

信息效用值：$d_j=1-e_j$ 信息效用值越大，其对应的信息越多

将信息效用值归一化，就能得到每个指标的熵权：$W_{j}=\frac{d_j}{\sum_{j=1}^{m}d_j}$

依据的原理： 指标的变异程度越小（方差小），所反映的信息量也越少（信息熵大，信息效用值小），其对应的权值也应该越低。（客观= 数据本身就可以告诉我们权重）

## 附：python实现topsis代码。
```python
import pandas as pd
import numpy as np

# 对极小型指标进行处理
def Min2Max(data):
    return data.max() - data

# 对中间型指标进行处理
def Mid2Max(data, best):
    M = np.max(np.abs(data-best))
    return 1 - np.abs(data-best)/M

# 对区间型指标进行处理
def Inter2Max(data, a, b):
    row = data.shape[0]
    M = max([a - np.min(data), np.max(data) - b])
    res = np.zeros(row)
    for i in range(row):
        if data[i] < a:
            res[i] = 1 - (a - data[i])/M
        elif data[i] > b:
            res[i] = 1 - (data[i] - b)/M
        else:
            res[i] = 1
    return res
data = pd.read_excel("20条河流的水质情况数据.xlsx").values

# 原始矩阵正向化
data[:, 3] = Min2Max(data[:, 3])
data[:, 2] = Mid2Max(data[:, 2], 7)
data[:, 4] = Inter2Max(data[:, 4], 10 ,20)


# 正向化矩阵标准化
data[:, 1:] = data[:, 1:] / np.sum(data[:, 1:]**2, axis=0)**0.5

# 计算得分并归一化
temp = data[:, 1:]
Z_zheng = np.max(temp, axis=0)
Z_fu = np.min(temp, axis=0)
w = np.ones(temp.shape[1]) # 在这里修改权重
D_zheng = np.sum(w*(Z_zheng-temp)**2, axis=1)**0.5  
D_fu = np.sum(w*(Z_fu-temp)**2, axis=1)**0.5  
S = D_fu / (D_zheng + D_fu)
Stand_S = S / np.sum(S)

# 降序排列
inx = np.argsort(-Stand_S)
Stand_S = Stand_S[inx]
print(Stand_S)

```



