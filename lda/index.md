# LDA


# 线性判别分析LDA(Linear Discriminant Analysis)
线性判别分析，也就是LDA（与主题模型中的LDA区分开），现在常常用于数据的降维中，但从它的名字中可以看出来它也是一个分类的算法，而且属于硬分类，也就是结果不是概率，是具体的类别，一起学习一下吧。

## 主要思想
1. 类内方差小
2. 类间方差大

## 推导
这里以二类为例，即只有两个类别。

首先是投影，我们假定原来的数据是向量 $x$，那么顺着 $ w$ 方向的投影就是标量：
$$
z=w^T\cdot x(=|w|\cdot|x|\cos\theta)
$$
对第一点，相同类内部的样本更为接近，我们假设属于两类的试验样本数量分别是 $N_1$和 $N_2$，那么我们采用方差矩阵来表征每一个类内的总体分布，这里我们使用了协方差的定义，用 $S$ 表示原数据的协方差：

$$
\begin{aligned}
C_1:Var_z[C_1]&=\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(z_i-\bar{z_{c1}})(z_i-\bar{z_{c1}})^T\nonumber\\\\\\\\
&=\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(w^Tx_i-\frac{1}{N_1}\sum\limits_{j=1}^{N_1}w^Tx_j)(w^Tx_i-\frac{1}{N_1}\sum\limits_{j=1}^{N_1}w^Tx_j)^T\nonumber\\\\\\\\
&=w^T\frac{1}{N_1}\sum\limits_{i=1}^{N_1}(x_i-\bar{x_{c1}})(x_i-\bar{x_{c1}})^Tw\nonumber\\\\
=w^TS_1w\\\\\\\\
C_2:Var_z[C_2]&=\frac{1}{N_2}\sum\limits_{i=1}^{N_2}(z_i-\bar{z_{c2}})(z_i-\bar{z_{c2}})^T\nonumber\\\\
=w^TS_2w
\end{aligned}
$$

所以类内距离为：

$$
\begin{align}
Var_z[C_1]+Var_z[C_2]=w^T(S_1+S_2)w
\end{align}
$$


对于第二点，我们可以用两类的均值表示这个距离：

$$
\begin{align}
(\bar{z_{c1}}-\bar{z_{c2}})^2&=(\frac{1}{N_1}\sum\limits_{i=1}^{N_1}w^Tx_i-\frac{1}{N_2}\sum\limits_{i=1}^{N_2}w^Tx_i)^2\nonumber\\\\
&=(w^T(\bar{x_{c1}}-\bar{x_{c2}}))^2\nonumber\\\\
&=w^T(\bar{x_{c1}}-\bar{x_{c2}})(\bar{x_{c1}}-\bar{x_{c2}})^Tw
\end{align}
$$

合这两点，由于协方差是一个矩阵，于是我们用将这两个值相除来得到我们的损失函数，并最大化这个值：

$$
\begin{align}
\hat{w}=\mathop{argmax}\limits_wJ(w)&=\mathop{argmax}\limits_w\frac{(\bar{z_{c1}}-\bar{z_{c2}})^2}{Var_z[C_1]+Var_z[C_2]}\nonumber\\\\
&=\mathop{argmax}\limits_w\frac{w^T(\bar{x_{c1}}-\bar{x_{c2}})(\bar{x_{c1}}-\bar{x_{c2}})^Tw}{w^T(S_1+S_2)w}\nonumber\\\\\\\\
&=\mathop{argmax}\limits_w\frac{w^TS_bw}{w^TS_ww}
\end{align}
$$

这样，我们就把损失函数和原数据集以及参数结合起来了。下面对这个损失函数求偏导，注意我们其实对w的绝对值没有任何要求，只对方向有要求，因此只要一个方程就可以求解了：

$$
\begin{aligned}
&\frac{\partial}{\partial w}J(w)=2S_bw(w^TS_ww)^{-1}-2w^TS_bw(w^TS_ww)^{-2}S_ww=0\nonumber\\\\\\\\
&\Longrightarrow S_bw(w^TS_ww)=(w^TS_bw)S_ww\nonumber\\\\\\\\
&\Longrightarrow w\propto S_w^{-1}S_bw=S_w^{-1}(\bar{x_{c1}}-\bar{x_{c2}})(\bar{x_{c1}}-\bar{x_{c2}})^Tw\propto S_w^{-1}(\bar{x_{c1}}-\bar{x_{c2}})
\end{aligned}
$$

也就是说最后我们的结果就是$w=S_w^{-1}(\bar{x_{c1}}-\bar{x_{c2}})$
可以归一化求得单位的w值。

## 多类情况
前面的很容易类比二类的情况，现在的目标函数变成了：

$$
\frac{W^TS_bW}{W^TS_wW}
$$
现在的问题就是这些都是矩阵，不能像上面那样直接优化，需要替换优化目标。

$$
\underbrace{arg\;max}\_W\;\;J(W) = \frac{\prod\limits_{diag}W^TS_bW}{\prod\limits_{diag}W^TS_wW}
$$
其中 $\prod_{diag}A$为A的主对角线元素的乘积,W为$n \times d$的矩阵，n为原来的维度，d为映射到超平面的维度，则最终的目标就变成了：

$$
J(W) = \frac{\prod\limits_{i=1}^dw_i^TS_bw_i}{\prod\limits_{i=1}^dw_i^TS_ww_i} = \prod\limits_{i=1}^d\frac{w_i^TS_bw_i}{w_i^TS_ww_i}
$$
根据广式瑞利商，最大值是矩阵$S_w^{-1}S_b$的最大特征值,最大的d个值的乘积就是矩阵的$S_w^{-1}S_b$最大的d个特征值的乘积,此时对应的矩阵$W$为这最大的d个特征值对应的特征向量张成的矩阵。


## 总结

LDA是一种监督学习的降维技术，也就是说它的数据集的每个样本是有类别输出的。这点和PCA不同。PCA是不考虑样本类别输出的无监督降维技术。LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”。什么意思呢？ 我们要将数据在低维度上进行投影，投影后希望每一种类别数据的投影点尽可能的接近，而不同类别的数据的类别中心之间的距离尽可能的大。



实际上LDA除了可以用于降维以外，还可以用于分类。一个常见的LDA分类基本思想是假设各个类别的样本数据符合高斯分布，这样利用LDA进行投影后，可以利用极大似然估计计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

LDA用于降维，和PCA有很多相同，也有很多不同的地方，因此值得好好的比较一下两者的降维异同点。

首先我们看看相同点：

1）两者均可以对数据进行降维。

2）两者在降维时均使用了矩阵特征分解的思想。

3）两者都假设数据符合高斯分布。

我们接着看看不同点：

1）LDA是有监督的降维方法，而PCA是无监督的降维方法

2）LDA降维最多降到类别数k-1的维数，而PCA没有这个限制。

3）LDA除了可以用于降维，还可以用于分类。

4）LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。

## 代码

```python
mean_list = []
for i in range(2):
    mean_list.append(np.mean(X_train[y_train==i], axis=0))
mean_list = np.array(mean_list)
S_W = np.zeros((X_train.shape[1], X_train.shape[1])) # 类内散度矩阵
for c, mv in zip(range(2), mean_list):
    class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
    for row in X_train[y_train==c]:
        row, mv = row.reshape(X_train.shape[1], -1), mv.reshape(X_train.shape[1], -1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter

over_all_mean = np.mean(X_train, axis=0)
S_B = np.zeros((X_train.shape[1], X_train.shape[1])) # 类间散度矩阵
for i, mean_vec in enumerate(mean_list):
    n = X_train[y_train==i, :].shape[0]
    mean_list_temp = mean_list[i, :].reshape(1, -1)
    over_all_mean = over_all_mean.reshape(X_train.shape[1], 1)
    S_B += n*(mean_vec-over_all_mean).dot((mean_vec-over_all_mean).T)


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
 

eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# eigv_sum = sum(eig_vals)
# for i, j in enumerate(eig_pairs):
#     print('eigenvalue {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum).real)) # 根据百分比显示特征值，从而选取最大的n个特征值
W = np.hstack((eig_pairs[0][1].reshape(X_train.shape[1], 1), eig_pairs[1][1].reshape(X_train.shape[1], 1)))
```






