# SVM


# SVM

## kernel

### 介绍

其实核函数和映射关系并不大，kernel可以看作是一个运算技巧。

一般认为，原本在低维线性不可分的数据集在足够高的维度存在线性可分的超平面。

围绕这个，那么我们所做的就是要在Feature Space套用原本在线性可分情况下的Input Space中使用过的优化方法，来找到那个Maximaizing Margin的超平面。原理机制一模一样，是二次规划，唯一不同是代入数据的不同，将原来的$x_i$替换成了高维空间中的$\phi(x_i)$，这就是映射函数，映射到高维空间。

具体的技巧(trick)，就是简化计算二次规划中间的一步内积计算。也即中间步骤有一步必须求得$\phi(x_i) \phi(x_j)$，我们可以定义核函数$K(x_i,x_j) = \phi(x_i)\phi(x_j)$，这样我们不需要显式计算每一个$\phi(x_i)$，甚至不需要知道它的形式，就可以直接计算结果出来。

也就是说，核函数、内积、相似度这三个词是等价的。因为inner product其实就是一种similarity的度量。核函数和映射是无关的。

### 例子

举一个例子：

考虑一个带有特征映射的二维输入空间 $\chi \subseteq \mathbb{R}^{2}$ :
特征映射二维到三维: $\quad \Phi: x=\left(x_{1}, x_{2}\right) \rightarrow \Phi(x)=\left(x_{1}^{2}, x_{2}^{2}, \sqrt{2} x_{1} x_{2}\right) \in F=\mathbb{R}^{3}$
特征空间中的内积：

$$
\begin{aligned}
\langle\Phi(x), \Phi(z)\rangle &=\left\langle\left(x_{1}^{2}, x_{2}^{2}, \sqrt{2} x_{1} x_{2}\right),\left(z_{1}^{2}, z_{2}^{2}, \sqrt{2} z_{1} z_{2}\right)\right\rangle \\\\
&=x_{1}^{2} z_{1}^{2}+x_{2}^{2} z_{2}^{2}+2 x_{1} x_{2} z_{1} z_{2} \\\\
&=\left\langle x_{1} z_{1}+x_{2} z_{2}\right\rangle^{2} \\\\
&=\langle x, z\rangle^{2}
\end{aligned}
$$


根据上面可得，核函数$k(x,z) = \langle x,z \rangle^2=\phi(x)^T \phi(z)$

而这里为什么映射函数是这样的形式呢，其实可以是反推出来的，我也不知道，反正凑巧通过这种映射函数可以得到这个核函数。

### 常用核函数理解
以高斯核函数为例，
$$
\kappa\left(x_{1}, x_{2}\right)=\exp \left(-\frac{\left|x_{1}-x_{2}\right|^{2}}{2 \sigma^{2}}\right)
$$
我们假设 $\sigma=1$ ，则

$$
\begin{aligned}
\kappa\left(x_{1}, x_{2}\right) &=\exp \left(-\frac{\left|x_{1}-x_{2}\right|^{2}}{2 \sigma^{2}}\right) \\\\
&=\exp \left(-\left(x_{1}-x_{2}\right)^{2}\right) \\\\
&=\exp \left(-x_{1}^{2}\right) \exp \left(-x_{2}^{2}\right) \exp \left(2 x_{1} x_{2}\right) \\\\
& \text { Taylor } \\\\
&=\exp \left(-x_{1}^{2}\right) \exp \left(-x_{2}^{2}\right)\left(\sum_{i=0}^{\infty} \frac{\left(2 x_{1} x_{2}\right)^{i}}{i !}\right) \\\\
&=\sum_{i=0}^{\infty}\left(\exp \left(-x_{1}^{2}\right) \exp \left(-x_{2}^{2}\right) \sqrt{\left.\frac{2^{i}}{i !} \sqrt{\frac{2^{i}}{i !}} x_{1}^{i} x_{2}^{i}\right)}\right.\\\\
&=\sum_{i=0}^{\infty}\left(\left[\exp \left(-x_{1}^{2}\right) \sqrt{\frac{2^{i}}{i !}} x_{1}^{i}\right]\left[\exp \left(-x_{2}^{2}\right) \sqrt{\frac{2^{i}}{i !}} x_{2}^{i}\right]\right) \\\\
&=\phi\left(x_{1}\right)^{T} \phi\left(x_{2}\right)
\end{aligned}
$$

w这不，已经有了定义的那种形式，对于 $\phi(x)$ ，由于

$$
\phi(x)=\exp \left(-x^{2}\right) \cdot\left(1, \sqrt{\frac{2^{1}}{1 !}} x, \sqrt{\frac{2^{2}}{2 !}} x^{2}, \cdots\right)
$$

所以，可以映射到任何一个维度上。

### 核函数类别
![](image/Pasted%20image%2020220802000231.png)其实常用的就那几个，高斯核函数最为常用。

### 参考
>[https://www.cnblogs.com/damin1909/p/12955240.html](https://www.cnblogs.com/damin1909/p/12955240.html)
>[https://blog.csdn.net/mengjizhiyou/article/details/103437423](https://blog.csdn.net/mengjizhiyou/article/details/103437423)

## 线性可分支持向量机

### 线性可分
![](image/Pasted%20image%2020220827233624.png)

在二维空间上，两类点被一条直线完全分开叫做线性可分。

### 最大间隔超平面

以最大间隔把两类样本分开的超平面，也称之为最大间隔超平面。

### 支持向量
![](image/Pasted%20image%2020220827233700.png)
样本中距离超平面最近的一些点，这些点叫做支持向量。

### 最优化问题
SVM 想要的就是找到各类样本点到超平面的距离最远，也就是找到最大间隔超平面。任意超平面可以用下面这个线性方程来描述：

$$
w^Tx+b=0
$$

二维空间点(x,y)到直线$Ax+By+C=0$的距离公式为：

$$
\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}
$$

扩展到n维空间中，$x=(x_1, x_2,\dots, x_n)$到直线$w^Tx+b=0$的距离为：

$$
\frac{|w^Tx+b|}{||w||}
$$
如图所示，根据支持向量的定义我们知道，支持向量到超平面的距离为 d，其他点到超平面的距离大于 d。

![](image/Pasted%20image%2020220827234112.png)

于是我们有这样的一个公式：

![](image/Pasted%20image%2020220827234141.png)

之后得到:

![](image/Pasted%20image%2020220827234154.png)

分母都是正数，因此可以令它为1。
![](image/Pasted%20image%2020220827235615.png)

合并得：
![](image/Pasted%20image%2020220827235627.png)

至此我们就可以得到最大间隔超平面的上下两个超平面：

![](image/Pasted%20image%2020220827235656.png)

每个支持向量到超平面的距离可以写为：
![](image/Pasted%20image%2020220827235715.png)

所以我们得到：
![](image/Pasted%20image%2020220827235827.png)

最大化这个距离：

![](image/Pasted%20image%2020220827235837.png)

这里乘上 2 倍也是为了后面推导，对目标函数没有影响。刚刚我们得到支持向量$y(w^Tx+b) = 1$，所以我们得到：

$$
\max \frac{2}{||w||}
$$

对目标进行转换：

$$
\min \frac{1}{2}||w||^2
$$

所以得到的最优化问题是：

![](image/Pasted%20image%2020220828000128.png)

### 对偶问题

#### 拉格朗日乘数法、拉格朗日对偶和KKT条件

参考：[https://zhuanlan.zhihu.com/p/38163970](https://zhuanlan.zhihu.com/p/38163970)
给定约束优化问题：

$$
\begin{aligned}
&\min f(x) \\\\
& s.t. g(x) = 0
\end{aligned}
$$

为方便分析，假设 f 与 g 是连续可导函数。Lagrange乘数法是等式约束优化问题的典型解法。定义Lagrangian函数

$$
L(x, \lambda) = f(x) + \lambda g(x)
$$
其中 λ 称为Lagrange乘数。Lagrange乘数法将原本的约束优化问题转换成等价的无约束优化问题
计算 L 对 x 与 λ 的偏导数并设为零，可得最优解的必要条件：
![](image/Pasted%20image%2020220828151556.png)

接下来是不等式约束：

$$
\begin{aligned}
& \min f(x) \\\\
& s.t. g(x) \leq 0 
\end{aligned}
$$

据此我们定义可行域(feasible region)$K=x\in R^n | g(x)\leq 0$。设$x^*$为满足条件的最佳解，分情况讨论：

1. $g(x^*) < 0$，最佳解位于K的内部，为内部解，这时的约束是无效的。
2. $g(x^*) = 0$，最佳解落在K的边界，称为边界解，此时的约束是有效的。
这两种情况的最佳解具有不同的必要条件。

具有不同的必要条件：

1. 内部解：在约束条件无效的情况下，$g(x)$不起作用，约束优化问题退化为无约束优化问题，因此$x^*$满足$\lambda = 0$
2. 边界解：在约束有效的情况下，约束不等式变为等式$g(x)=0$。此时拉格朗日函数在$x^*$的梯度为0，即$\nabla f = -\lambda \nabla g$，$f(x)$的极小值在边界取到，那么可行域内部的$f(x)$应该都是大于这个极小值，则$\nabla f(x)$的方向是可行域内部。而$\nabla g$的方向为可行域外部，因为约束条件是$g(x) \leq 0$，也就是可行域外部都是$g(x) > 0$，所以梯度方向就是指向函数增加的方向。说明两个函数的梯度方向相反，要想上面的等式成立，必须有$\lambda \geq 0$，这就是对偶可行性。
因此，不论是内部解或边界解， $\lambda g(x)=0$ 恒成立

整合上述两种情况，最佳解的必要条件包括Lagrangian函数的定常方程式、原始可行性、对偶可行性，以及互补松弛性：
![](image/Pasted%20image%2020220828153456.png)

这就是KKT条件。

上面结果可推广至多个约束等式与约束不等式的情况。考虑标准约束优化问题(或称非线性规划)：
![](image/Pasted%20image%2020220828153544.png)

定义Lagrangian 函数

![](image/Pasted%20image%2020220828153556.png)

则KKT条件为
![](image/Pasted%20image%2020220828153608.png)



#### 应用

已知svm优化的主要问题：

![](image/Pasted%20image%2020220828000902.png)

那么求解线性可分的 SVM 的步骤为：

**步骤1：**

构造拉格朗日函数：

![](image/Pasted%20image%2020220828000939.png)

**步骤2：**

利用强对偶性转化：

![](image/Pasted%20image%2020220828001013.png)

现对参数 w 和 b 求偏导数：

![](image/Pasted%20image%2020220828001025.png)

具体步骤：
![](image/Pasted%20image%2020220828001050.png)

在前面的步骤中即为：

![](image/Pasted%20image%2020220828001120.png)

我们将这个结果带回到函数中可得：

![](image/Pasted%20image%2020220828001143.png)

也就是说：

![](image/Pasted%20image%2020220828001152.png)

**步骤3：**
![](image/Pasted%20image%2020220828001208.png)

由上述过程需要满足KKT条件（$\alpha$就是本文中的$\lambda$）：
![](image/Pasted%20image%2020220828002201.png)

易得，当$\lambda_i$大于0，则必有$y_if(x_i)=1$,所对应的样本点是一个支持向量，即位于最大间隔边界上。

我们可以看出来这是一个二次规划问题，问题规模正比于训练样本数，我们常用 SMO(Sequential Minimal Optimization) 算法求解。

SMO(Sequential Minimal Optimization)，序列最小优化算法，其核心思想非常简单：每次只优化一个参数，其他参数先固定住，仅求当前这个优化参数的极值。我们来看一下 SMO 算法在 SVM 中的应用。

我们刚说了 SMO 算法每次只优化一个参数，但我们的优化目标有约束条件，没法一次只变动一个参数。所以我们选择了一次选择两个参数。具体步骤为：

1. 选择两个需要更新的参数$\lambda _i$和$\lambda_j$，固定其他参数。于是我们有以下约束：
![](image/Pasted%20image%2020220828001631.png)

其中$c = -\sum_{k\neq i, j}\lambda_ky_k$， 因此可以得出$\lambda_j = \frac{c-\lambda_iy_i}{y_j}$，这样就相当于把目标问题转化成了仅有一个约束条件的最优化问题，仅有的约束是$\lambda_i>0$

2. 对于仅有一个约束条件的最优化问题，我们完全可以在$\lambda_i$上对优化目标求偏导，令导数为零，从而求出变量值$\lambda_{inew}$，从而求出$\lambda_{jnew}$
3. 多次迭代直至收敛。
通过 SMO 求得最优解$\lambda^*$

**步骤4：**

我们求偏导数时得到：

![](image/Pasted%20image%2020220828002003.png)

由上式可求得 w。

由于所有$\lambda_i>0$的点都是支持向量，可以随便找一个支持向量代入$y_s(w^Tx_s+b)=1$，求出b即可。

两边同时乘以$y_s$，最后得$b = y_s-wx_s$

为了更具鲁棒性，我们可以求得支持向量的均值：

![](image/Pasted%20image%2020220828002938.png)

**步骤5：**
w 和 b 都求出来了，我们就能构造出最大分割超平面：$w^Tx+b=0$

分类决策函数：$f(x) = sign(w^Tx+b)$

![](image/Pasted%20image%2020220828003127.png)

将新样本点导入到决策函数中既可得到样本的分类。



## 线性支持向量机与软间隔

### 软间隔
在实际应用中，完全线性可分的样本是很少的，如果遇到了不能够完全线性可分的样本，我们应该怎么办？比如下面这个：

![](image/Pasted%20image%2020220828105535.png)

  
于是我们就有了软间隔，相比于硬间隔的苛刻条件，我们允许个别样本点出现在间隔带里面，即允许出现分类错误的样本：

![](image/Pasted%20image%2020220828105552.png)

我们允许部分样本点不满足约束条件：


$$
y_i(w^Tx_i+b) \geq 1
$$

则优化目标变成了

$$
\min _{w, b} \frac{1}{2}\|{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left({w}^{\mathrm{T}} {x}_{i}+b\right)-1\right),
$$


其中 $C>0$ 是一个常数, $\ell_{0 / 1}$ 是 “ $0 / 1$ 损失函数”

$$
\ell_{0 / 1}(z)= \begin{cases}1, & \text { if } z<0 \\\\ 0, & \text { otherwise. }\end{cases}
$$


显然, 当 $C$ 为无穷大时, $\xi_i$必然无穷小，如此一来线性svm就又变成了线性可分svm，当$C$为有限值时，才会允许部分样本不遵循约束条件

然而, $\ell_{0 / 1}$ 非凸、非连续, 数学性质不太好, 使得不易直接求解. 于 是, 人们通常用其他一些函数来代替 $\ell_{0 / 1}$, 称为 “替代损失” (surrogate loss). 替代损失函数一般具有较好的数学性质, 如它们通常是凸的连续函数且是 $\ell_{0 / 1}$ 的上界. 给出了三种常用的替代损失函数:

hinge 损失: $\ell_{\text {hinge }}(z)=\max (0,1-z)$;

指数损失(exponential loss): $\ell_{\exp }(z)=\exp (-z)$;

对率损失(logistic loss): $\ell_{\log }(z)=\log (1+\exp (-z))$.

若采用 hinge 损失, 则变成

$$
\min _{w, b} \frac{1}{2}\|{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left({w}^{\mathrm{T}} {x}_{i}+b\right)\right)
$$


为了度量这个间隔软到何种程度，我们为每个样本引入一个松弛变量$\xi_i$，令$\xi_i \geq 0$，且$1-y_i(w^Tx_i+b)-\xi_i\leq 0$，如下图：
![](image/Pasted%20image%2020220828111154.png)

### 优化目标与求解
优化目标：

![](image/Pasted%20image%2020220828112312.png)

**步骤1：**

构造拉格朗日函数：
![](image/Pasted%20image%2020220828112251.png)

**步骤2：**
分别求导，得出以下关系：

![](image/Pasted%20image%2020220828112422.png)

将这些关系带入拉格朗日函数中，得到：

![](image/Pasted%20image%2020220828112445.png)

则：

![](image/Pasted%20image%2020220828112459.png)

我们可以看到这个和硬间隔的一样，只是多了个约束条件。

然后使用SMO算法求$\lambda^*$

#### 软间隔KKT条件
![](image/Pasted%20image%2020220828113029.png)
其中$\alpha$对应本文的$\lambda$，$\mu$对应本文的$\mu$

因此由第三个式子得必有$\lambda_i =0$或者$y_if(x_i) - 1+\xi_i \geq 0$
$\lambda_i=0$，则该样本对其没有任何影响。
$\lambda_i > 0$，则样本为支持向量。
若$\lambda_i <C$,则$\mu_i > 0$，进而有$\xi_i=0$，则样本恰在最大间隔边界上。也是支持向量。
若$\lambda_i=C$,则有$\mu_i=0$，此时若$\xi_i\leq 1$，则样本落在最大间隔内部。若$\xi_i>1$则样本被错误分类。

再看一下下面这图就理解了。
![](image/Pasted%20image%2020220828111154.png)
**步骤3：**

![](image/Pasted%20image%2020220828112623.png)

然后我们通过上面两个式子求出 w 和 b，最终求得超平面

**这边要注意一个问题，在间隔内的那部分样本点是不是支持向量？**

我们可以由求参数 w 的那个式子可看出，只要 $\lambda_i > 0$的点都能影响我们的超平面，因此都是支持向量。

## 非线性支持向量机

我们刚刚讨论的硬间隔和软间隔都是在说样本的完全线性可分或者大部分样本点的线性可分。

但我们可能会碰到的一种情况是样本点不是线性可分的，比如：

![](image/Pasted%20image%2020220828135755.png)

这种情况的解决方法就是：将二维线性不可分样本映射到高维空间中，让样本点在高维空间线性可分，比如：

![](image/Pasted%20image%2020220828135803.png)

对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到支持向量机，就是非线性 SVM。

我们用 x 表示原来的样本点，用$\phi(x)$表示 x 映射到特征新的特征空间后到新向量。那么分割超平面可以表示为: $f(x) = w\phi(x)+b$

对于非线性 SVM 的对偶问题就变成了：

![](image/Pasted%20image%2020220828143014.png)

区别就在于优化目标中的内积。

### 核函数

我们不禁有个疑问：只是做个内积运算，为什么要有核函数的呢？

这是因为低维空间映射到高维空间后维度可能会很大，如果将全部样本的点乘全部计算好，这样的计算量太大了。

但如果我们有这样的一核函数$k(x,y) = (\phi(x), \phi(y))$，x与y在特征空间中的内积，就等于它们在原始空间中通过函数$k(x,y)$计算的结果，我们就不需要知道映射函数和计算高维空间中的内积了。

有关内容看本文一开始对kernel的介绍。

## 总结
SVM是深度学习流行之前的首选分类方法，在许多任务上都有很好的效果，稍微修改后可以用于回归任务中。总结一下svm算法的优缺点。

### 优点
-   有严格的数学理论支持，可解释性强，不依靠统计方法，从而简化了通常的分类和回归问题；
-   能找出对任务至关重要的关键样本（即：支持向量）；
-   采用核技巧之后，可以处理非线性分类/回归任务；
-   最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”。

### 缺点
- 训练时间长。当采用 SMO 算法时，每次都需要挑选一对参数
- 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为$O(N^2)$
- 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高。
## 参考

>[https://zhuanlan.zhihu.com/p/77750026](https://zhuanlan.zhihu.com/p/77750026)



