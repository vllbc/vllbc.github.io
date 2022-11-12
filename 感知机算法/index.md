# 感知机算法


# 感知机算法
感知机印象中没有系统学习过但是是一个很简单的算法，最近看了一下李航老师的统计学习方法，发现感知机的思想和svm十分类似，并且比svm简单的多，不需要间隔最大，只需要分开就可以。同时老师在课堂上面讲的版本也有点不一样，主要是计算上的不同，本质还是一样的。然后就打算整理一下这一块。

## 感知机模型

假设输入空间（特征空间) 是 $\mathcal{X} \subseteq \mathbf{R}^n$, 输出空间是 $\mathcal{Y}=\{+1,-1\}$ 。输入 $x \in \mathcal{X}$ 表示实例的特征向量, 对应于输入空间 (特征空间 ) 的点; 输出 $y \in \mathcal{Y}$ 表示实例的类别。由输入空间到输出空间的如下函数

$$
f(x)=\operatorname{sign}(w \cdot x+b)
$$

称为感知机。其中, $w$ 和 $b$ 为感知机模型参数, $w \in \mathbf{R}^n$ 叫作权值 (weight) 或权值向 量 (weight vector), $b \in \mathbf{R}$ 叫作偏置 (bias), $w \cdot x$ 表示 $w$ 和 $x$ 的内积。sign 是符号 函数, 即

$$
\operatorname{sign}(x)=\left\{\begin{array}{cc}
+1, & x \geqslant 0 \\\\
-1, & x<0
\end{array}\right.
$$


## 损失函数

假设训练数据集是线性可分的, 感知机学习的目标是求得一个能够将训练集正实 例点和负实例点完全正确分开的分离超平面。为了找出这样的超平面, 即确定感知 机模型参数 $w, b$, 需要确定一个学习策略, 即定义 (经验) 损失函数并将损失函数极 小化。
损失函数的一个自然选择是误分类点的总数。但是, 这样的损失函数不是参数 $w$, $b$ 的连续可导函数, 不易优化。损失函数的另一个选择是误分类点到超平面 $S$ 的总距 离, 这是感知机所采用的。为此, 首先写出输入空间 $\mathbf{R}^n$ 中任一点 $x_0$ 到超平面 $S$ 的 距离:

$$
\frac{1}{\|w\|}\left|w \cdot x_0+b\right|
$$

这里, $\|w\|$ 是 $w$ 的 $L_2$ 范数。
其次, 对于误分类的数据 $\left(x_i, y_i\right)$ 来说,

$$
-y_i\left(w \cdot x_i+b\right)>0
$$

成立。因为当 $w \cdot x_i+b>0$ 时, $y_i=-1$; 而当 $w \cdot x_i+b<0$ 时, $y_i=+1$ 。 因此, 误 分类点 $x_i$ 到超平面 $S$ 的距离是

$$
-\frac{1}{\|w\|} y_i\left(w \cdot x_i+b\right)
$$

这样, 假设超平面 $S$ 的误分类点集合为 $M$, 那么所有误分类点到超平面 $S$ 的总 距离为

$$
-\frac{1}{\|w\|} \sum_{x_i \in M} y_i\left(w \cdot x_i+b\right)
$$

不考虑 $\frac{1}{\|w\|}$, 就得到感知机学习的损失函数。

即

$$
L(w, b) = -\sum_{x_i \in M} y_i\left(w \cdot x_i+b\right)
$$

使用梯度下降算法更新参数，对损失函数求导得到：

$$
\begin{aligned}
&\nabla_w L(w, b)=-\sum_{x_i \in M} y_i x_i \\\\
&\nabla_b L(w, b)=-\sum_{x_i \in M} y_i
\end{aligned}
$$


随机选取一个误分类点 $\left(x_i, y_i\right)$, 对 $w, b$ 进行更新:

$$
\begin{gathered}
w \leftarrow w+\eta y_i x_i \\\\
b \leftarrow b+\eta y_i
\end{gathered}
$$
## 算法流程
总结可得，感知机算法流程如下：

输入: 训练数据集 $T=\left\{\left(x_1, y_1\right),\left(x_2, y_2\right), \cdots,\left(x_N, y_N\right)\right\}$, 其中 $x_i \in \mathcal{X}=\mathbf{R}^n, y_i \in$ $\mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$; 学习率 $\eta(0<\eta \leqslant 1)$;
输出: $w, b$; 感知机模型 $f(x)=\operatorname{sign}(w \cdot x+b)$ 。
(1) 选取初值 $w_0, b_0$;
(2) 在训练集中选取数据 $\left(x_i, y_i\right)$;
(3) 如果 $y_i\left(w \cdot x_i+b\right) \leqslant 0$,

$$
\begin{aligned}
&w \leftarrow w+\eta y_i x_i \\\\
&b \leftarrow b+\eta y_i
\end{aligned}
$$

(4) 转至 (2), 直至训练集中没有误分类点。

这很容易理解。就是求解最佳参数$w$和$b$，使用梯度下降算法，对于每个样本，如果其真实的标签与预测的结果符号不一致，也就是sign函数之前的结果不同号，则说明分类错误，则就需要更新参数，不断地继续更新直到所有的样本都分类正确。

## 另一种表达方式

感知器: 用数据训练线性模型 $g({x})={w}^T {x}+w_0$
增广的样本向量:

$$
{y}=\left(1 ; x_1 ; x_2 ; \ldots ; x_d\right)
$$

增广的权向量:

$$
{\alpha}=\left(w_0 ; w_1 ; \ldots ; w_d\right)
$$

线性判别函数:

$$
g({y})={\alpha}^T {y}
$$

决策规则: 如果 $g({y})>0$, 则 $y \in \omega_0$; 如果 $g({y})<0$, 则 $y \in \omega_1$

若定义新变量 $y^{\prime}$, 使

$$
y_i^{\prime}=\left\{\begin{array}{lll}
y_i,  \text { 若 } & {y}_i \in \omega_0 \\\\
-{y}_i, \text { 若 } & {y}_i \in \omega_1
\end{array} \quad i=1,2, \ldots, m\right.
$$

样本可分性条件变为：存在 $\alpha$, 使

$$
{\alpha}^T {y}_i^{\prime}>0, i=1,2, \ldots, m
$$

$y^{\prime}$ 称作规范化增广样本向量, 仍记作 $y$ 。

可以用这样的形式定义损失函数为：

$$
J(\alpha) = \sum_{\alpha^Ty_k \leq 0} (-\alpha^Ty_k)
$$
其中w和b合并为了$\alpha$。$y_k$为原来的x加上了1用于与偏置b对应。

### 梯度下降迭代法求解

$$
\boldsymbol{\alpha}(t+1)=\boldsymbol{\alpha}(t)-\rho_t \nabla J_P(\boldsymbol{\alpha})
$$

下一时刻的权向量是把当前时刻的权向量向目标函数的负梯度方向调整一个修
正量, $\rho_t$ 为调整的步长 (“学习率”)。

$$
\nabla J_P(\boldsymbol{\alpha})=\frac{\partial J_P(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}}=\sum_{\alpha^T y_k \leq 0}\left(-y_k\right)
$$

所以

$$
\alpha(t+1)=\alpha(t)+\rho_t \sum_{\alpha^T y_k \leq 0} y_k
$$

即每次迭代时把错分的样本按照某个系数加到权向量上。
当没有错分样本时, 得到一个合适的解 $\alpha^*$ 。


### 固定增量法

（1）任意选择初始权向量 $\alpha(0)$;
(2) 对样本 $y_j$, 若 $\alpha(t)^T y_j \leq 0$, 则 $\alpha(t+1)=\alpha(t)+y_j$ (假设 $\left.\rho_t=1\right)$, 否则继 续;
(3) 对所有样本重复 (2), 直至对所有的样本都有 $\alpha(t)^T y_j>0$, 即 $J_P(\boldsymbol{\alpha})=0$

与梯度下降法的区别就是每次只对一个样本更新，可以这样理解： 原始的数据，对于第二类则增广之后取负数就可以理解为前面第一种表达的$y_i\left(w \cdot x_i+b\right)$， 大于0则说明分类正确，否则说明分类错误，就需要更新参数。


