# 交叉熵损失函数



# Softmax理解

主要记录了在使用softmax这个函数中遇到的一些问题，比较基础，但确实困扰了一段时间。

在学习word2vec中, 使用的一般都是如下的损失函数：

$$
\begin{aligned}
Loss = J(\theta) = -\frac{1}{T}logL(\theta)= -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\neq0} logP(w_{t+j}|w_t, \theta) \\\\ =\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\neq0}J_{t,j}(\theta).
\end{aligned}
$$

$$
\begin{aligned}
J_{t,j}(\theta) = -logP(miss|xwh) = -log\frac{exp(u_{miss}^Tv_{xwh})}{\sum_{o\in V}exp(u_o^Tv_{xwh})} = \\\\ -u_{miss}^Tv_{xwh}+log\sum_{o\in V} exp(u_o^Tv_{xwh})
\end{aligned}
$$

但是说起交叉熵往往是下面的式子：

$$
L = -\sum_{c=1}^Cy_clog(p_c)
$$

在学习的时候就疑惑，这两种形式有什么区别与联系呢，最近看到一篇文章正好解答了这个疑惑。
下面给出结论：
第一种形式是只针对正确类别的对应点输出，将这个位置的softmax即概率最大化，而第二种形式是直接衡量真实分布和实际输出之间的距离，因为交叉熵就是由KL散度变形得来的。

## 交叉熵
### 信息量
一条信息的信息量大小和它的不确定性有很大的关系。一句话如果需要很多外部信息才能确定，我们就称这句话的信息量比较大。
将信息的定义为下：

$I(x_0) = -log(p(x_0))$

### 熵
信息量是对于单个事件来说的，但是实际情况一件事有很多种发生的可能，比如掷骰子有可能出现6种情况，明天的天气可能晴、多云或者下雨等等。**熵是表示随机变量不确定的度量，是对所有可能发生的事件产生的信息量的期望**。公式如下：

$$
H(X) = -\sum_{i=1}^np(x_i)log(p(x_i))
$$
### 相对熵
相对熵也称之为KL散度，用于衡量对于同意随机变量x的两个分布p(x)和q(x)之间的差异。在机器学习中，p(x)通常描述样本的真实分布，例如[1, 0, 0, 0]表示样本属于第一类，而q(x)常常用于表示预测的分布，例如[0.7, 0.1, 0.1, 0.1]
KL散度的定义公式如下：

$$
	D_{KL}(p||q) = \sum_{i=1}^np(x_i)log(\frac{p(x_i)}{q(x_i)}) 
$$

KL越小则说明二者越接近
### 交叉熵
将KL散度变形

$$
D_{KL}(p||q) = \sum_{i=1}^np(x_i)log(p(x_i)) -\sum_{i=1}^np(x_i)log(q(x_i)) = -H(p(x)) - \sum_{i=1}^np(x_i)log(q(x_i))
$$

后半部分就是我们的交叉熵，常常用于评估predict和label之间的差别。
## 理解
以一个三分类的问题来说，假设真实分布为[0, 1, 0]，则对于第二个式子来说

$$
L = -\sum_{c=1}^Cy_clog(p_c) = -0 \times log(p_0) - 1\times log(p_1) - 0\times log(p_3) = -log(p_1)
$$

对于第一个式子就是

$$
loss_1 = -log(\frac{e^{z_1}}{\sum_{c=1}^C e^{z_c}}) = -log(p_1) = -z_1+log\sum_{c=1}^Ce^{z_c}
$$

所以说实际上这俩是一样的，只是出发点不一样。在skip-gram中，使用的就是第一种式子，对它来说，正确的类别就是背景词，就直接将背景词的概率最大，即损失函数最小。

`loss = (-output_layer[:, Y] + torch.log(torch.sum(torch.exp(output_layer), dim=1))).mean()`

这行代码就是对第一个式子的实现，需要注意的是所有的操作都是基于batch的。负采样的形式有一些不同，这里不再讨论，到这里我才终于稍稍理解了softmax这个函数。

## 与二分类的联系
具体的说二分类就是两个类别，可以使用上述的方法定义，对于正类和负类都用不同的概率表示，也可以只计算正类，负类就等于1-正类，上述出现的y都是向量的形式，对于二分类，向量的长度就是2，就可以直接展开，最后的结果就是熟知的损失函数的形式。


## 参考

> https://zhuanlan.zhihu.com/p/105722023
