# 典型相关分析



# 典型相关分析

参考：[CCA](https://www.cnblogs.com/pinard/p/6288716.html)

> 典型关联分析(Canonical Correlation Analysis，以下简称CCA)是最常用的挖掘数据关联关系的算法之一。比如我们拿到两组数据，第一组是人身高和体重的数据，第二组是对应的跑步能力和跳远能力的数据。那么我们能不能说这两组数据是相关的呢？CCA可以帮助我们分析这个问题。

CCA使用的方法是将多维的X和Y都用线性变换为1维的X'和Y'，然后再使用相关系数来看X'和Y'的相关性。将数据从多维变到1位，也可以理解为CCA是在进行降维，将高维数据降到1维，然后再用相关系数进行相关性的分析。下面我们看看CCA的算法思想。

## 算法思想

现在我们具体来讨论下CCA的算法思想。假设我们的数据集是X和Y，X为$n_1\times m$的样本矩阵。Y为$n_2\times m$的样本矩阵.其中m为样本个数，而$n_1,n_2$分别为X和Y的特征维度。

对于X矩阵，我们将其投影到1维，或者说进行线性表示，对应的投影向量或者说线性系数向量为a, 对于Y矩阵，我们将其投影到1维，或者说进行线性表示，对应的投影向量或者说线性系数向量为b, 这样X ,Y投影后得到的一维向量分别为X',Y'。我们有

$$
X'=a^TX,Y'=b^TY
$$

我们CCA的优化目标是最大化ρ(X′,Y′)得到对应的投影向量a,b，即

$$
\underbrace{argmax}_{a,b}\frac{cov(X',Y')}{\sqrt{D(X')D(Y')}}
$$

在投影前，我们一般会把原始数据进行标准化，得到均值为0而方差为1的数据X和Y。这样我们有：


$$
cov(X',Y')=cov(a^TX,b^TY)=a^Tcov(X,Y)b \\\\
D(X') = D(a^TX)= a^TD(X)a \\\\
D(Y') = D(b^TY)= b^TD(Y)b
$$


令$S_{XY} = cov(X,Y)$优化目标可以简化为


$$
\underbrace{argmax}_{a,b}\frac{a^TS_{XY}b}{\sqrt{a^TS_{XX}a}\sqrt{b^TS_{YY}b}}
$$

由于分子分母增大相同的倍数，优化目标结果不变，我们可以采用和SVM类似的优化方法，固定分母，优化分子，具体的转化为：


$$
\underbrace{argmax}_{a,b}\quad a^TS_{XY}b \\\\
s.t.\quad a^TS_{XX}a=1,b^TS_{YY}b=1 \\\\
\text{因为已经标准化了，因此方差为0}
$$

也就是说，我们的CCA算法的目标最终转化为一个凸优化过程，只要我们求出了这个优化目标的最大值，就是我们前面提到的多维X和Y的相关性度量，而对应的a,ba,b则为降维时的投影向量，或者说线性系数。

这个函数优化一般有两种方法，第一种是奇异值分解SVD，第二种是特征分解，两者得到的结果一样。



## 优化

### svd求解

不想敲公式了，直接截图吧

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/CCA.png)

### 特征分解

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/CCA%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3.png)

## CCA算法流程

这里我们对CCA的算法流程做一个总结，以SVD方法为准。

输入：各为m个的样本X和Y，X和Y的维度都大于1

输出：X,Y的相关系数$\rho$,X和Y的线性系数向量a和b

1.计算X的方差$S_{XX}$，Y的方差$S_{YY}$，X和Y的协方差$S_{XY}$，Y和X的协方差$S_{YX}=S_{XY}^T$

2.计算矩阵$M=S_{XX}^{-\frac{1}{2}}S_{XY}S_{YY}^{-\frac{1}{2}}$

3.对矩阵M进行奇异值分解，得到最大的奇异值$\rho$，和最大的奇异值对应的左右奇异向量u,v

4.计算X和Y的线性向量a,b，$a=S_{XX}^{-\frac{1}{2}}u, b=S_{YY}^{-\frac{1}{2}}v$
