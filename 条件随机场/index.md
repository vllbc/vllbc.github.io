# 条件随机场



# CRF

## 概率图模型与无向图
图是由结点和连接结点的边组成的集合。结点和边分别记作v和e，结点和边的集合分别记作V和E，图记作$G=(V, E)$。

无向图是指没有方向的图。

概率图模型是由图表示的概率分布。设有联合概率分布P(Y), Y是一组随机变量，由无向图$G=(V,E)$表示概率分布P(Y)，即在图G中，结点$v\in V$表示一个随机变量$Y_v$，$Y=(Y_v)\_{v\in V}$，边e表示随机变量之间的依赖关系。

## 概率无向图模型

设有联合概率分布P(Y)，由无向图$G=(V,E)$表示，在图G中，结点表示随机变量，边表示随机变量之间的依赖关系。如果联合概率分布满足成对、局部或全局马尔科夫性，就称此联合概率分布称为概率无向图模型，或马尔科夫随机场。

## 因子分解

首先给出无向图的团和最大团的定义：

>无向图G中任何两个结点均有边连接的结点子集称为团。若C是无向图G的一个团，并且不能再加进任何一个G的结点使其成为更大的团，则称此C为最大团。


将无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，称为概率无向图模型的因子分解。

给定概率无向图模型, 设其无向图为 $G, C$ 为 $G$ 上的最大团, $Y_C$ 表示 $C$ 对应的 随机变量。那么概率无向图模型的联合概率分布 $P(Y)$ 可写作图中所有最大团 $C$ 上的 函数 $\Psi_C\left(Y_C\right)$ 的乘积形式, 即

$$
P(Y)=\frac{1}{Z} \prod_C \Psi_C\left(Y_C\right)
$$

其中, $Z$ 是规范化因子 (normalization factor), 由式

$$
Z=\sum_Y \prod_C \Psi_C\left(Y_C\right)
$$

给出。规范化因子保证 $P(Y)$ 构成一个概率分布。函数 $\Psi_C\left(Y_C\right)$ 称为势函数 (potential function)。这里要求势函数 $\Psi_C\left(Y_C\right)$ 是严格正的, 通常定义为指数函数:

$$
\Psi_C\left(Y_C\right)=\exp \\{-E\left(Y_C\right)\\\}
$$


## 条件随机场

条件随机场是指给定随机变量X的条件下，随机变量Y的马尔科夫随机场。一般的条件随机场主要是指线性链条件随机场，可以用于标注等问题。这里的$P(Y|X)$中，Y是输出变量，表示标注序列，X是输入变量，表示需要标注的观察序列。

### 一般的条件随机场
(条件随机场) 设 $X$ 与 $Y$ 是随机变量, $P(Y \mid X)$ 是在给定 $X$ 的条件 下 $Y$ 的条件概率分布。若随机变量 $Y$ 构成一个由无向图 $G=(V, E)$ 表示的马尔可夫 随机场, 即

$$
P\left(Y_v \mid X, Y_w, w \neq v\right)=P\left(Y_v \mid X, Y_w, w \sim v\right)
$$

对任意结点 $v$ 成立, 则称条件概率分布 $P(Y \mid X)$ 为条件随机场。式中 $w \sim v$ 表示在 图 $G=(V, E)$ 中与结点 $v$ 有边连接的所有结点 $w, w \neq v$ 表示结点 $v$ 以外的所有结 点, $Y_v, Y_u$ 与 $Y_w$ 为结点 $v, u$ 与 $w$ 对应的随机变量。

### 线性链条件随机场

设$X=(X_1,X_2, \dots, X_n), \quad Y=(Y_1, Y_2, \dots , Y_n)$均为线性链表示的随机变量序列，若在给定随机变量序列X的条件下，随机变量Y的条件概率分布$P(Y|X)$构成条件随机场，即满足马尔科夫性

$$
P\left(Y_i \mid X, Y_1, \cdots, Y_{i-1}, Y_{i+1}, \cdots, Y_n\right)=P\left(Y_i \mid X, Y_{i-1}, Y_{i+1}\right)
$$

$i=1,2, \cdots, n$ (在 $i=1$ 和 $n$ 时只考虑单边)
则称 $P(Y \mid X)$ 为线性链条件随机场。在标注问题中, $X$ 表示输入观测序列, $Y$ 表示对 应的输出标记序列或状态序列。

### 线性链条件随机场参数化形式
根据因子分解, 可以给出线性链条件随机场 $P(Y \mid X)$ 的因子分解式, 各因子是定 义在相邻两个结点 (最大团) 上的势函数。
(线性链条件随机场的参数化形式) 设 $P(Y \mid X)$ 为线性链条件随机 场, 则在随机变量 $X$ 取值为 $x$ 的条件下, 随机变量 $Y$ 取值为 $y$ 的条件概率具有如下 形式:

$$
P(y \mid x)=\frac{1}{Z(x)} \exp \left(\sum_{i, k} \lambda_k t_k\left(y_{i-1}, y_i, x, i\right)+\sum_{i, l} \mu_l s_l\left(y_i, x, i\right)\right)
$$

其中,

$$
Z(x)=\sum_y \exp \left(\sum_{i, k} \lambda_k t_k\left(y_{i-1}, y_i, x, i\right)+\sum_{i, l} \mu_l s_l\left(y_i, x, i\right)\right)
$$

式中, $t_k$ 和 $s_l$ 是特征函数, $\lambda_k$ 和 $\mu_l$ 是对应的权值。 $Z(x)$ 是规范化因子, 求和是在所 有可能的输出序列上进行的。
这两个式子是线性链条件随机场模型的基本形式, 表示给定输入序列 $x$, 对输出序列 $y$ 预测的条件概率。$t_k$ 是定义在边上的特 征函数, 称为转移特征, 依赖于当前和前一个位置; $s_l$ 是定义在结点上的特征函数, 称为状态特征, 依赖于当前位置。 $t_k$ 和 $s_l$ 都依赖于位置, 是局部特征函数。通常, 特 征函数 $t_k$ 和 $s_l$ 取值为 1 或 0 ; 当满足特征条件时取值为 1 , 否则为 0 。条件随机场完 全由特征函数 $t_k, s_l$ 和对应的权值 $\lambda_k, \mu_l$ 确定。
线性链条件随机场也是对数线性模型 (log linear model)。

### 条件随机场的简化形式

为简便起见, 首先将转移特征和状态特征及其权值用统一的符号表示。设有 $K_1$ 个转移特征, $K_2$ 个状态特征, $K=K_1+K_2$, 记

$$
f_k\left(y_{i-1}, y_i, x, i\right)= \begin{cases}t_k\left(y_{i-1}, y_i, x, i\right), & k=1,2, \cdots, K_1 \\\\ s_l\left(y_i, x, i\right), & k=K_1+l ; l=1,2, \cdots, K_2\end{cases}
$$

然后, 对转移与状态特征在各个位置 $i$ 求和, 记作

$$
f_k(y, x)=\sum_{i=1}^n f_k\left(y_{i-1}, y_i, x, i\right), \quad k=1,2, \cdots, K
$$

用 $w_k$ 表示特征 $f_k(y, x)$ 的权值, 即

$$
w_k= \begin{cases}\lambda_k, & k=1,2, \cdots, K_1 \\\\ \mu_l, & k=K_1+l ; l=1,2, \cdots, K_2\end{cases}
$$

于是, 条件随机场可表示为

$$
\begin{aligned}
P(y \mid x) &=\frac{1}{Z(x)} \exp \sum_{k=1}^K w_k f_k(y, x) \\\\
Z(x) &=\sum_y \exp \sum_{k=1}^K w_k f_k(y, x)
\end{aligned}
$$

若以 $w$ 表示权值向量, 即

$$
w=\left(w_1, w_2, \cdots, w_K\right)^{\mathrm{T}}
$$

以 $F(y, x)$ 表示全局特征向量, 即

$$
F(y, x)=\left(f_1(y, x), f_2(y, x), \cdots, f_K(y, x)\right)^{\mathrm{T}}
$$

则条件随机场可以写成向量 $w$ 与 $F(y, x)$ 的内积的形式:

$$
P_w(y \mid x)=\frac{\exp (w \cdot F(y, x))}{Z_w(x)}
$$

其中,

$$
Z_w(x)=\sum_y \exp (w \cdot F(y, x))
$$


### 矩阵形式
对每个 标记序列引进特殊的起点和终点状态标记 $y_0=$ start 和 $y_{n+1}=s t o p$, 这时标注序列 的概率 $P_w(y \mid x)$ 可以通过矩阵形式表示并有效计算。
对观测序列 $x$ 的每一个位置 $i=1,2, \cdots, n+1$, 由于 $y_{i-1}$ 和 $y_i$ 在 $m$ 个标记中 取值, 可以定义一个 $m$ 阶矩阵随机变量

$$
M_i(x)=\left[M_i\left(y_{i-1}, y_i \mid x\right)\right]
$$

矩阵随机变量的元素为

$$
\begin{aligned}
&M_i\left(y_{i-1}, y_i \mid x\right)=\exp \left(W_i\left(y_{i-1}, y_i \mid x\right)\right) \\\\
&W_i\left(y_{i-1}, y_i \mid x\right)=\sum_{k=1}^K w_k f_k\left(y_{i-1}, y_i, x, i\right)
\end{aligned}
$$

这里 $w_k$ 和 $f_k$ 分别由前面的式子给出, $y_{i-1}$ 和 $y_i$ 是标记随机变量 $Y_{i-1}$ 和 $Y_i$ 的取值。
这样, 给定观测序列 $x$, 相应标记序列 $y$ 的非规范化概率可以通过该序列 $n+1$ 个矩阵的适当元素的乘积 $\prod_{i=1}^{n+1} M_i\left(y_{i-1}, y_i \mid x\right)$ 表示。于是, 条件概率 $P_w(y \mid x)$ 是

$$
P_w(y \mid x)=\frac{1}{Z_w(x)} \prod_{i=1}^{n+1} M_i\left(y_{i-1}, y_i \mid x\right)
$$

其中, $Z_w(x)$ 为规范化因子, 是 $n+1$ 个矩阵的乘积的 (start, stop) 元素, 即

$$
Z_w(x)=\left[M_1(x) M_2(x) \cdots M_{n+1}(x)\right]_{\text {start,stop }}
$$

注意, $y_0=$ start 与 $y_{n+1}=$ stop 表示开始状态与终止状态, 规范化因子 $Z_w(x)$ 是以 start 为起点 stop为终点通过状态的所有路径 $y_1 y_2 \cdots y_n$ 的非规范化概率 $\prod_{i=1}^{n+1} M_i\left(y_{i-1}, y_i \mid x\right)$ 之和。

## 概率计算问题

与HMM类似，引入前向和后向变量，递归的计算概率和一些期望值。

### 前向-后向算法

对每个指标 $i=0,1, \cdots, n+1$, 定义前向向量 $\alpha_i(x)$ :

$$
\alpha_0(y \mid x)= \begin{cases}1, & y=\text { start } \\\\ 0, & \text { 否则 }\end{cases}
$$

递推公式为

$$
\alpha_i^{\mathrm{T}}\left(y_i \mid x\right)=\alpha_{i-1}^{\mathrm{T}}\left(y_{i-1} \mid x\right)\left[M_i\left(y_{i-1}, y_i \mid x\right)\right], \quad i=1,2, \cdots, n+1
$$

又可表示为

$$
\alpha_i^{\mathrm{T}}(x)=\alpha_{i-1}^{\mathrm{T}}(x) M_i(x)
$$

$\alpha_i\left(y_i \mid x\right)$ 表示在位置 $i$ 的标记是 $y_i$ 并且从 1 到 $i$ 的前部分标记序列的非规范化概 率, $y_i$ 可取的值有 $m$ 个, 所以 $\alpha_i(x)$ 是 $m$ 维列向量。
同样, 对每个指标 $i=0,1, \cdots, n+1$, 定义后向向量 $\beta_i(x)$ :

$$
\begin{aligned}
\beta_{n+1}\left(y_{n+1} \mid x\right) &= \begin{cases}1, & y_{n+1}=\text { stop } \\\\
0, & \text { 否则 }\end{cases} \\\\
\beta_i\left(y_i \mid x\right) &=\left[M_{i+1}\left(y_i, y_{i+1} \mid x\right)\right] \beta_{i+1}\left(y_{i+1} \mid x\right)
\end{aligned}
$$

又可表示为

$$
\beta_i(x)=M_{i+1}(x) \beta_{i+1}(x)
$$

$\beta_i\left(y_i \mid x\right)$ 表示在位置 $i$ 的标记为 $y_i$ 并且从 $i+1$ 到 $n$ 的后部分标记序列的非规范化 概率。

### 概率计算
按照前向-后向向量的定义, 很容易计算标记序列在位置 $i$ 是标记 $y_i$ 的条件概率 和在位置 $i-1$ 与 $i$ 是标记 $y_{i-1}$ 和 $y_i$ 的条件概率:

$$
P\left(Y_i=y_i \mid x\right)=\frac{\alpha_i^{\mathrm{T}}\left(y_i \mid x\right) \beta_i\left(y_i \mid x\right)}{Z(x)}
$$

$$
P\left(Y_{i-1}=y_{i-1}, Y_i=y_i \mid x\right)=\frac{\alpha_{i-1}^{\mathrm{T}}\left(y_{i-1} \mid x\right) M_i\left(y_{i-1}, y_i \mid x\right) \beta_i\left(y_i \mid x\right)}{Z(x)}
$$

其中,

$$
Z(x)=\alpha_n^{\mathrm{T}}(x) \mathbf{1}=1 \beta_1(x)
$$



## 预测问题

### 维特比算法

还是使用维特比算法。

$$
\begin{aligned}
y^*  &=\arg \max_y P_w(y \mid x) \\\\
&=\arg \max_y \frac{\exp (w \cdot F(y, x))}{Z_w(x)} \\\\
&=\arg \max_y \exp (w \cdot F(y, x)) \\\\
&=\arg \max_y(w \cdot F(y, x))
\end{aligned}
$$

于是, 条件随机场的预测问题成为求非规范化概率最大的最优路径问题

$$
\max_y(w \cdot F(y, x))
$$

这里, 路径表示标记序列。其中,

$$
\begin{aligned}
w &=\left(w_1, w_2, \cdots, w_K\right)^{\mathrm{T}} \\\\
F(y, x) &=\left(f_1(y, x), f_2(y, x), \cdots, f_K(y, x)\right)^{\mathrm{T}} \\\\
f_k(y, x) &=\sum_{i=1}^n f_k\left(y_{i-1}, y_i, x, i\right), \quad k=1,2, \cdots, K
\end{aligned}
$$

注意, 这时只需计算非规范化概率, 而不必计算概率, 可以大大提高效率。为了求解最 优路径, 写成如下形式:

$$
\max_y \sum_{i=1}^n w \cdot F_i\left(y_{i-1}, y_i, x\right)
$$

其中,

$$
F_i\left(y_{i-1}, y_i, x\right)=\left(f_1\left(y_{i-1}, y_i, x, i\right), f_2\left(y_{i-1}, y_i, x, i\right), \cdots, f_K\left(y_{i-1}, y_i, x, i\right)\right)^{\mathrm{T}}
$$

是局部特征向量。
下面叙述维特比算法。首先求出位置 1 的各个标记 $j=1,2, \cdots, m$ 的非规范化概率:

$$
\delta_1(j)=w \cdot F_1\left(y_0=\text { start, } y_1=j, x\right), \quad j=1,2, \cdots, m
$$

一般地, 由递推公式, 求出到位置 $i$ 的各个标记 $l=1,2, \cdots, m$ 的非规范化概率的最 大值, 同时记录非规范化概率最大值的路径

$$
\begin{gathered}
\delta_i(l)=\max_{1 \leqslant j \leqslant m}\left\\{\delta_{i-1}(j)+w \cdot F_i\left(y_{i-1}=j, y_i=l, x\right)\right\\\}, \quad l=1,2, \cdots, m \\\\
\Psi_i(l)=\arg \max_{1 \leqslant j \leqslant m}\left\\{\delta_{i-1}(j)+w \cdot F_i\left(y_{i-1}=j, y_i=l, x\right)\right\\\}, \quad l=1,2, \cdots, m
\end{gathered}
$$

直到 $i=n$ 时终止。这时求得非规范化概率的最大值为

$$
\operatorname{max}_y(w \cdot F(y, x))=\max_{1 \leqslant j \leqslant m} \delta_n(j)
$$

及最优路径的终点

$$
y_n^*  =\arg \max_{1 \leqslant j \leqslant m} \delta_n(j)
$$

由此最优路径终点返回,

$$
y_i^*  =\Psi_{i+1}\left(y_{i+1}^*  \right), \quad i=n-1, n-2, \cdots, 1
$$

求得最优路径 $y^* =\left(y_1^* , y_2^* , \cdots, y_n^* \right)^{\mathrm{T}}$ 。

综上所述, 得到条件随机场预测的维特比算法。


(条件随机场预测的维特比算法)
输入: 模型特征向量 $F(y, x)$ 和权值向量 $w$, 观测序列 $x=\left(x_1, x_2, \cdots, x_n\right)$;
输出: 最优路径 $y^* =\left(y_1^* , y_2^* , \cdots, y_n^* \right)$ 。
(1) 初始化

$$
\delta_1(j)=w \cdot F_1\left(y_0=\operatorname{start}, y_1=j, x\right), \quad j=1,2, \cdots, m
$$

(2) 递推。对 $i=2,3, \cdots, n$

$$
\begin{gathered}
\delta_i(l)=\max_{1 \leqslant j \leqslant m}\left\\{\delta_{i-1}(j)+w \cdot F_i\left(y_{i-1}=j, y_i=l, x\right)\right\\\}, \quad l=1,2, \cdots, m \\\\
\Psi_i(l)=\arg \max_{1 \leqslant j \leqslant m}\left\\{\delta_{i-1}(j)+w \cdot F_i\left(y_{i-1}=j, y_i=l, x\right)\right\\\}, \quad l=1,2, \cdots, m
\end{gathered}
$$

（3）终止

$$
\begin{gathered}
\max_y(w \cdot F(y, x))=\max_{1 \leqslant j \leqslant m} \delta_n(j) \\\\
y_n^* =\arg \max_{1 \leqslant j \leqslant m} \delta_n(j)
\end{gathered}
$$

(4) 返回路径

$$
y_i^* =\Psi_{i+1}\left(y_{i+1}^* \right), \quad i=n-1, n-2, \cdots, 1
$$

求得最优路径 $y^* =\left(y_1^* , y_2^* , \cdots, y_n^* \right)\_{\text {。 }}$

自己的理解就是非规范化概率每个i代表时间步i，要对所有的$\lambda_k t_k$和 $\mu_ks_k$ 进行筛选，找出符合条件的相加，这里要注意下标的理解。括号里的和HMM的类似，代表y的取值。

### 实例

![](image/Pasted%20image%2020220903225846.png)

![](image/Pasted%20image%2020220903225857.png)

这里使用维特比算法求解

![](image/Pasted%20image%2020220903225959.png)


## 参数学习问题


## 总结
![](image/Pasted%20image%2020220904001447.png)
