# PCA



# 主成分分析(PCA)

主成分分析（Principal components analysis，以下简称PCA）是最重要的降维方法之一。在数据压缩消除冗余和数据噪音消除等领域都有广泛的应用。注意的是PCA属于无监督学习。

PCA降维的原则是投影方差最大。

使用PCA时如果有不同种类的数据，PCA会把这些数据混合在一起降维。

PCA顾名思义，就是找出数据里最主要的方面，用数据里最主要的方面来代替原始数据。具体的，假如我们的数据集是n维的，共有m个数据(x(1),x(2),...,x(m))。我们希望将这m个数据的维度从n维降到n'维，希望这m个n'维的数据集尽可能的代表原始数据集。我们知道数据从n维降到n'维肯定会有损失，但是我们希望损失尽可能的小。那么如何让这n'维的数据尽可能表示原来的数据呢？

## 协方差矩阵

在统计学中，方差是用来度量单个随机变量的离散程度，而协方差则一般用来刻画两个随机变量的相似程度，其中，方差的计算公式为:

$$
\sigma_x^2 = \frac{1}{n}\sum_{i=1}^n(x-\bar{x})^2
$$

协方差的公式为:

$$
\sigma(x,y) = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})
$$

根据方差的定义，给定d个随机变量$x_k,k=1,2,\dots,d$，这些随机变量的方差为

$$
\sigma(x_k,x_k) = \frac{1}{n-1}\sum_{i=1}^n(x_{ki}-\bar{x_k})^2,k=1,2,\dots,d
$$

因此可以求出两两之间的协方差

$$
\sigma(x_m,x_k) = \frac{1}{n-1}\sum_{i=1}^n(x_{mi}-\bar{x_m})(x_{ki}-\bar{x_k})
$$



因此，协方差矩阵为


$$
\sum = \begin{bmatrix}
\sigma(x_1,x_1) & \cdots & \sigma(x_1,x_d) \\\\
\vdots & \ddots & \vdots \\\\
\sigma(x_d,x_1) & \cdots & \sigma(x_d,x_d)
\end{bmatrix}
$$


## 拉格朗日乘数法优化
设原始数据矩阵 X 对应的协方差矩阵为 C，而 P 是一组基按行组成的矩阵，设 Y=PX，则 Y 为 X 对 P 做基变换后的数据。设 Y 的协方差矩阵为 D，我们推导一下 D 与 C 的关系：

$$
D = \frac{1}{m}YY^T \\\\
	=\frac{1}{m}(PX)(PX)^T \\\\
	=\frac{1}{m}PXX^TP^T \\\\
	=P(\frac{1}{m}XX^T)P^T \\\\
	=PCP^T
$$

我们令P=$w^T$,令原本的协方差为A，于是我们有优化目标如下：

$$
\begin{cases}
\max{w^TAw} \\\\
s.t. w^Tw = 1
\end{cases}
$$

然后构造拉格朗日函数：

$$
L(w) = w^TAw + \lambda(1-w^Tw)
$$

对w求导：

$$
Aw = \lambda w
$$

则方差$D(x) = w^TAw = \lambda w^Tw = \lambda$

于是我们发现，x 投影后的方差就是协方差矩阵的特征值。我们要找到最大方差也就是协方差矩阵最大的特征值，最佳投影方向就是最大特征值所对应的特征向量，次佳就是第二大特征值对应的特征向量，以此类推。

## 对角矩阵
由上文知道，协方差矩阵 C 是一个是对称矩阵，在线性代数中实对称矩阵有一系列非常好的性质：

- 实对称矩阵不同特征值对应的特征向量必然正交。
- 设特征向量$\lambda$ 重数为 r，则必然存在 r 个线性无关的特征向量对应于 $\lambda$ ，因此可以将这 r 个特征向量单位正交化。
- 实对称矩阵一定可以对角化

由上面两条可知，一个 n 行 n 列的实对称矩阵一定可以找到 n 个单位正交特征向量，设这 n 个特征向量为 $e_1,e_2,\cdots,e_n$，我们将其按列组成矩阵： $E=(e_1,e_2,\cdots,e_n)$。
对于协方差矩阵C有以下结论:


$$
E^TCE = 
\begin{bmatrix}
\lambda_1 \\\\
&\lambda_2 \\\\
&& \ddots \\\\
&&& \lambda_n
\end{bmatrix}
$$

注：因为E为正交矩阵，则$E^{-1}$ = $E^T$ ,这个过程成为相似对角化，$\lambda_n$为C的特征值，对应的特征向量为$e_n$

这都是线代的基础知识。

## SVD
复制一下将协方差矩阵写成中心化的形式：

$$
\begin{align}S&=\frac{1}{N}\sum\limits_{i=1}^N(x_i-\overline{x})(x_i-\overline{x})^T\nonumber\\\\
&=\frac{1}{N}(x_1-\overline{x},x_2-\overline{x},\cdots,x_N-\overline{x})(x_1-\overline{x},x_2-\overline{x},\cdots,x_N-\overline{x})^T\nonumber\\\\
&=\frac{1}{N}(X^T-\frac{1}{N}X^T\mathbb{I}_{N1}\mathbb{I}_{N1}^T)(X^T-\frac{1}{N}X^T\mathbb{I}_{N1}\mathbb{I}_{N1}^T)^T\nonumber\\\\
&=\frac{1}{N}X^T(E_N-\frac{1}{N}\mathbb{I}_{N1}\mathbb{I}_{1N})(E_N-\frac{1}{N}\mathbb{I}_{N1}\mathbb{I}_{1N})^TX\nonumber\\\\
&=\frac{1}{N}X^TH_NH_N^TX\nonumber\\\\
&=\frac{1}{N}X^TH_NH_NX=\frac{1}{N}X^THX
\end{align}
$$

对中心化后的数据集进行奇异值分解：
$$
HX=U\Sigma V^T,U^TU=E_N,V^TV=E_p,\Sigma:N\times p
$$

于是：
$$
S=\frac{1}{N}X^THX=\frac{1}{N}X^TH^THX=\frac{1}{N}V\Sigma^T\Sigma V^T
$$
因此，我们直接对中心化后的数据集进行 SVD，就可以得到特征值$\Sigma^2$和特征向量 $V$，在新坐标系中的坐标就是：
$$
HX\cdot V
$$
## 步骤
总结一下 PCA 的算法步骤：

设有 m 条 n 维数据。

1) 对所有的样本进行中心化： $x^{(i)} = x^{(i)}-\frac{1}{m}\sum_{j=1}^mx^{(j)}$

2) 计算样本的协方差矩阵 $XX^T$
3) 对矩阵$XX^T$进行特征值分解
4）取出最大的n'个特征值对应的特征向量$(w_1,w_2,...,w_{n′})$, 将所有的特征向量标准化后，组成特征向量矩阵W。

5）对样本集中的每一个样本$x^{(i)}$,转化为新的样本$z^{(i)}=W^Tx^{(i)}$
6) 得到输出样本集$D^′=(z^{(1)},z^{(2)},...,z^{(m)})$

## 实例
假设我们的数据集有10个二维数据(2.5,2.4), (0.5,0.7), (2.2,2.9), (1.9,2.2), (3.1,3.0), (2.3, 2.7), (2, 1.6), (1, 1.1), (1.5, 1.6), (1.1, 0.9)，需要用PCA降到1维特征。

首先我们对样本中心化，这里样本的均值为(1.81, 1.91),所有的样本减去这个均值向量后，即中心化后的数据集为(0.69, 0.49), (-1.31, -1.21), (0.39, 0.99), (0.09, 0.29), (1.29, 1.09), (0.49, 0.79), (0.19, -0.31), (-0.81, -0.81), (-0.31, -0.31), (-0.71, -1.01)。

现在我们开始求样本的协方差矩阵。
然后求出特征值$(0.0490833989,1.28402771)$，对应的特征向量为 $(0.735178656,0.677873399)^T$,$(−0.677873399,−0.735178656)^T$，,由于最大的k=1个特征值为1.28402771,对于的k=1个特征向量为$(−0.677873399,−0.735178656)^T$. 则我们的W=$(−0.677873399,−0.735178656)^T$。
们对所有的数据集进行投影$z^{(i)}=W^Tx^{(i)}$，得到PCA降维后的10个一维数据集为：(-0.827970186， 1.77758033， -0.992197494， -0.274210416， -1.67580142， -0.912949103， 0.0991094375， 1.14457216, 0.438046137， 1.22382056)

## 性质
1. 缓解维度灾难：PCA 算法通过舍去一部分信息之后能使得样本的采样密度增大（因为维数降低了），这是缓解维度灾难的重要手段；
2. 降噪：当数据受到噪声影响时，最小特征值对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到降噪的效果；
3. 过拟合：PCA 保留了主要信息，但这个主要信息只是针对训练集的，而且这个主要信息未必是重要信息。有可能舍弃了一些看似无用的信息，但是这些看似无用的信息恰好是重要信息，只是在训练集上没有很大的表现，所以 PCA 也可能加剧了过拟合；
4. 特征独立：PCA 不仅将数据压缩到低维，它也使得降维之后的数据各特征相互独立；

## 代码

```python
# 手动实现PCA算法
import numpy as np

def pca(data):
    """
    主成分分析
    :param data: 数据集
    :return:
    """
    # 数据集的行数
    num_data, num_feat = data.shape
    # 对每一列的数据进行平均值的计算
    mean_vec = np.mean(data, axis=0)
    # 对数据集中每一行的数据进行平均值的计算
    data_mean_centered = data - mean_vec
    # 计算协方差矩阵
    sigma = np.dot(data_mean_centered.T, data_mean_centered) / num_data 
    # 计算特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(sigma)
    # 对特征值进行排序
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # 要降维的维数，这里以2为例。将特征向量以列向量形式拼合。
    matrix_w = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(2)])
    # 将原始数据进行投影。
    res = data.dot(matrix_w)
    print(res)
pca(np.array([[1, 2, 5], [3, 4, 6], [5, 6, 9], [3, 2 ,5]]))
```

```
[[ 4.78637704 -1.46926342] 
 [ 7.62278435 -0.82094287] 
 [11.68194836 -0.79767573] 
 [ 5.77746434  0.2656909 ]]
```



# 参考文章



> [https://zhuanlan.zhihu.com/p/77151308](https://zhuanlan.zhihu.com/p/77151308)
>
> [https://www.cnblogs.com/pinard/p/6239403.html](https://www.cnblogs.com/pinard/p/6239403.html)
