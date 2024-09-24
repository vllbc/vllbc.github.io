# 高斯混合聚类


基础就是高斯混合模型，假设我们熟知的高斯分布的概率密度函数为$p(x\mid \mu, \Sigma)$。则高斯混合分布为：

$$
p_{\mathcal{M}}(\boldsymbol{x})=\sum_{i=1}^k \alpha_i \cdot p\left(\boldsymbol{x} \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i\right)
$$

分布共由 $k$ 个混合成分组成, 每个混合成分对应一个高斯分布. 其中 $\mu_i$ 与 $\Sigma_i$ 是第 $i$ 个高斯混合成分的参数, 而 $\alpha_i>0$ 为相应的 “混合系数” (mixture coefficient), $\sum_{i=1}^k \alpha_i=1$。
假设样本的生成过程由高斯混合分布给出: 首先, 根据 $\alpha_1, \alpha_2, \ldots, \alpha_k$ 定义 的先验分布选择高斯混合成分, 其中 $\alpha_i$ 为选择第 $i$ 个混合成分的概率; 然后, 根 据被选择的混合成分的概率密度函数进行采样, 从而生成相应的样本。

## 聚类原理
如何利用高斯混合分布进行聚类？观察这个混合系数，思路就是有多少个混合的模型，就代表要聚多少类，对于给定数据集，可以定义

$$
\gamma_{j k}= \begin{cases}1, & \text { 第 } j \text { 个观测来自第 } k \text { 个分模型 } \\\\ 0, & \text { 否则 }\end{cases}
$$

则样本j的簇标记$\lambda_j= \underbrace{\arg \max}_{i\in {1,2, \dots ,k}} \gamma_{jk}$

## EM算法

如何计算$\gamma_{jk}$呢，如下式所示，其中$\alpha_k$为混合系数，$\theta_k$为第k个高斯分布的参数

$$
\begin{aligned}
\hat{\gamma}_{j k} &=E\left(\gamma_{j k} \mid y, \theta\right)=P\left(\gamma_{j k}=1 \mid y, \theta\right) \\\\
&=\frac{P\left(\gamma_{j k}=1, y_j \mid \theta\right)}{\sum_{k=1}^K P\left(\gamma_{j k}=1, y_j \mid \theta\right)} \\\\
&=\frac{P\left(y_j \mid \gamma_{j k}=1, \theta\right) P\left(\gamma_{j k}=1 \mid \theta\right)}{\sum_{k=1}^K P\left(y_j \mid \gamma_{j k}=1, \theta\right) P\left(\gamma_{j k}=1 \mid \theta\right)} \\\\
&=\frac{\alpha_k \phi\left(y_j \mid \theta_k\right)}{\sum_{k=1}^K \alpha_k \phi\left(y_j \mid \theta_k\right)}, \quad j=1,2, \cdots, N ; \quad k=1,2, \cdots, K
\end{aligned}
$$

式子里的y其实就是观测样本。

那么模型的参数要怎么估计呢，很显然可以使用EM算法，$\gamma$为隐变量，其实我这里的叙述顺序是有问题的，其实是EM算法中求Q函数的过程中需要计算的一个值，详细的过程在本博客的EM算法里面。总之得到了$\gamma_{jk}$后，就得到了Q函数:

$$
Q\left(\theta, \theta^{(i)}\right)=\sum_{k=1}^K\{n_k \log \alpha_k+\sum_{j=1}^N \hat{\gamma}_{j k}\left[\log \left(\frac{1}{\sqrt{2 \pi}}\right)-\log \sigma_k-\frac{1}{2 \sigma_k^2}\left(y_j-\mu_k\right)^2\right]\}
$$

极大似然估计Q函数就可以得到参数的下一轮估计值：

$$
\theta^{(i+1)}=\arg \max_\theta Q\left(\theta, \theta^{(i)}\right)
$$

用 $\hat{\mu}_k, \hat{\sigma}_k^2$ 及 $\hat{\alpha}_k, k=1,2, \cdots, K$, 表示 $\theta^{(i+1)}$ 的各参数。求 $\hat{\mu}_k, \hat{\sigma}_k^2$ 只需分别对 $\mu_k, \sigma_k^2$ 求偏导数并令其为 0 , 即可得到; 求 $\hat{\alpha}_k$ 是在 $\sum_{k=1}^K \alpha_k=1$ 条件 下求偏导数并令其为 0 得到的。结果如下:

$$
\begin{gathered}
\hat{\mu}_k=\frac{\sum_{j=1}^N \hat{\gamma}_{j k} y_j}{\sum_{j=1}^N \hat{\gamma}_{j k}}, \quad k=1,2, \cdots, K \\\\
\hat{\sigma}_k^2=\frac{\sum_{j=1}^N \hat{\gamma}_{j k}\left(y_j-\mu_k\right)^2}{\sum_{j=1}^N \hat{\gamma}_{j k}}, \quad k=1,2, \cdots, K \\\\
\hat{\alpha}_k=\frac{n_k}{N}=\frac{\sum_{j=1}^N \hat{\gamma}_{j k}}{N}, \quad k=1,2, \cdots, K
\end{gathered}
$$

得到参数后，再进行新的一轮迭代，计算$\gamma$值，如此反复。
算法收敛后，就可以对样本进行聚类，根据$\lambda_j= \underbrace{\arg \max}_{i\in {1,2, \dots ,k}} \gamma_{jk}$可以得到每个样本的簇标记。具体的流程如下：
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020221107214541.png)

## 总结

高斯混合分布的形式就注定了它可以用来进行聚类，并且还有EM算法如此强大的数学工具进行模型参数的学习，高斯混合聚类与Kmeans都属于原型聚类。




