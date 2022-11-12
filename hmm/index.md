# HMM


# 隐马尔科夫模型

## 介绍

HMM可以看做是处理序列模型的传统方法。
一般来说HMM解决三个问题：

1. 评估观察序列概率。给定模型$\lambda=(A,B,\prod)$和观察序列$O=\\{o_1,o_2,\dots,o_T\\}$，计算在模型$\lambda$下观测序列O出现的概率$P(O\lvert \lambda)$，这个问题需要用到前向后向算法，属于三个问题中最简单的。
2. 预测问题，也叫解码问题。即给定模型$\lambda = (A,B,\prod)$和观测序列$O=\\{o_1,o_2,\dots,o_T\\}$，求在给定观测序列条件下，最可能出现的对应的状态序列，这个问题的求解需要用到基于动态规划的维特比算法，这个问题属于三个问题中复杂度居中的算法。
3. 模型参数学习问题。即给定观测序列$O=\\{o_1,o_2,\dots,o_T\\}$，估计模型$\lambda = (A,B,\prod)$的参数，使得该模型下观测序列的条件概率$P(O\lvert\lambda)$最大，这个问题的求解需要用到基于EM算法的鲍姆-韦尔奇算法。属于三个问题中最复杂的。  

## 定义
设 $Q$ 是所有可能的状态的集合, $V$ 是所有可能的观测的集合:

$$
Q=\left\\{q_1, q_2, \cdots, q_N\right\\\}, \quad V=\left\\{v_1, v_2, \cdots, v_M\right\\\}
$$

其中, $N$ 是可能的状态数, $M$ 是可能的观测数。
$I$ 是长度为 $T$ 的状态序列, $O$ 是对应的观测序列:

$$
I=\left(i_1, i_2, \cdots, i_T\right), \quad O=\left(o_1, o_2, \cdots, o_T\right)
$$

$A$ 是状态转移概率矩阵:

$$
A=\left[a_{i j}\right]_{N \times N}
$$

其中,

$$
a_{i j}=P\left(i_{t+1}=q_j \mid i_t=q_i\right), \quad i=1,2, \cdots, N ; \quad j=1,2, \cdots, N
$$

是在时刻 $t$ 处于状态 $q_i$ 的条件下在时刻 $t+1$ 转移到状态 $q_j$ 的概率。
$B$ 是观测概率矩阵:

$$
B=\left[b_j(k)\right]_{N \times M}
$$

其中,

$$
b_j(k)=P\left(o_t=v_k \mid i_t=q_j\right), \quad k=1,2, \cdots, M ; \quad j=1,2, \cdots, N
$$

是在时刻 $t$ 处于状态 $q_j$ 的条件下生成观测 $v_k$ 的概率。
$\pi$ 是初始状态概率向量:

$$
\pi=\left(\pi_i\right)
$$

其中,

$$
\pi_i=P\left(i_1=q_i\right), \quad i=1,2, \cdots, N
$$

是时刻 $t=1$ 处于状态 $q_i$ 的概率。
隐马尔可夫模型由初始状态概率向量 $\pi$ 、状态转移概率矩阵 $A$ 和观测概率矩阵 $B$ 决定。 $\pi$ 和 $A$ 决定状态序列, $B$ 决定观测序列。因此, 隐马尔可夫模型 $\lambda$ 可以用三元 符号表示, 即

$$
\lambda=(A, B, \pi)
$$

$A, B, \pi$ 称为隐马尔可夫模型的三要素。

状态转移概率矩阵 $A$ 与初始状态概率向量 $\pi$ 确定了隐藏的马尔可夫链, 生成不 可观测的状态序列。观测概率矩阵 $B$ 确定了如何从状态生成观测, 与状态序列综合确 定了如何产生观测序列。

## 两个基本假设

(1) 齐次马尔可夫性假设, 即假设隐藏的马尔可夫链在任意时刻 $t$ 的状态只依赖 于其前一时刻的状态, 与其他时刻的状态及观测无关, 也与时刻 $t$ 无关:

$$
P\left(i_t \mid i_{t-1}, o_{t-1}, \cdots, i_1, o_1\right)=P\left(i_t \mid i_{t-1}\right), \quad t=1,2, \cdots, T
$$

(2) 观测独立性假设, 即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状 态, 与其他观测及状态无关:

$$
P\left(o_t \mid i_T, o_T, i_{T-1}, o_{T-1}, \cdots, i_{t+1}, o_{t+1}, i_t, i_{t-1}, o_{t-1}, \cdots, i_1, o_1\right)=P\left(o_t \mid i_t\right)
$$

## 观测序列生成的过程

输入: 隐马尔可夫模型 $\lambda=(A, B, \pi)$, 观测序列长度 $T$;
输出: 观测序列 $O=\left(o_1, o_2, \cdots, o_T\right)$ 。
(1) 按照初始状态分布 $\pi$ 产生状态 $i_1$;
(2) 令 $t=1$;
(3) 按照状态 $i_t$ 的观测概率分布 $b_{i_t}(k)$ 生成 $o_t$ :
(4) 按照状态 $i_t$ 的状态转移概率分布 $\left\\{a_{i_t i_{t+1}}\right\\\}$ 产生状态 $i_{t+1}, i_{t+1}=1,2, \cdots, N$;
(5) 令 $t=t+1$; 如果 $t<T$, 转步 (3); 否则, 终止。

## 概率计算问题

### 直接计算（复杂度太高）
给定模型 $\lambda=(A, B, \pi)$ 和观测序列 $O=\left(o_1, o_2, \cdots, o_T\right)$, 计算观测序列 $O$ 出现 的概率 $P(O \mid \lambda)$ 。最直接的方法是按概率公式直接计算。通过列举所有可能的长度为 $T$ 的状态序列 $I=\left(i_1, i_2, \cdots, i_T\right)$, 求各个状态序列 $I$ 与观测序列 $O=\left(o_1, o_2, \cdots, o_T\right)$ 的联合概率 $P(O, I \mid \lambda)$, 然后对所有可能的状态序列求和, 得到 $P(O \mid \lambda)$ 。
状态序列 $I=\left(i_1, i_2, \cdots, i_T\right)$ 的概率是:

$$
P(I \mid \lambda)=\pi_{i_1} a_{i_1 i_2} a_{i_2 i_3} \cdots a_{i_{T-1} i_T}
$$

对固定的状态序列 $I=\left(i_1, i_2, \cdots, i_T\right)$, 观测序列 $O=\left(o_1, o_2, \cdots, o_T\right)$ 的概率是:

$$
P(O \mid I, \lambda)=b_{i_1}\left(o_1\right) b_{i_2}\left(o_2\right) \cdots b_{i_T}\left(o_T\right)
$$

$O$ 和 $I$ 同时出现的联合概率为

$$
\begin{aligned}
P(O, I \mid \lambda) &=P(O \mid I, \lambda) P(I \mid \lambda) \\\\
&=\pi_{i_1} b_{i_1}\left(o_1\right) a_{i_1 i_2} b_{i_2}\left(o_2\right) \cdots a_{i_{T-1} i_T} b_{i_T}\left(o_T\right)
\end{aligned}
$$

然后, 对所有可能的状态序列 $I$ 求和, 得到观测序列 $O$ 的概率 $P(O \mid \lambda)$, 即

$$
\begin{aligned}
P(O \mid \lambda) &=\sum_I P(O \mid I, \lambda) P(I \mid \lambda) \\\\
&=\sum_{i_1, i_2, \cdots, i_T} \pi_{i_1} b_{i_1}\left(o_1\right) a_{i_1 i_2} b_{i_2}\left(o_2\right) \cdots a_{i_{T-1} i_T} b_{i_T}\left(o_T\right)
\end{aligned}
$$


这种算法复杂度太高，计算量太大，有效算法为前向算法和后向算法。


### 前向算法
首先定义前向概率。
给定隐马尔可夫模型 $\lambda$, 定义到时刻 $t$ 部分观测序列为 $o_1, o_2, \cdots, o_t$ 且状态为 $q_i$ 的概率为前向概率, 记作

$$
\alpha_t(i)=P\left(o_1, o_2, \cdots, o_t, i_t=q_i \mid \lambda\right)
$$



可以递推地求得前向概率 $\alpha_t(i)$ 及观测序列概率 $P(O \mid \lambda)$ 。
(观测序列概率的前向算法)
输入: 隐马尔可夫慔型 $\lambda$, 观测序列 $O$;
输出: 观测序列概率 $P(O \mid \lambda)$ 。
(1) 初值

$$
\alpha_1(i)=\pi_i b_i\left(o_1\right), \quad i=1,2, \cdots, N
$$

（2）递推 对 $t=1,2, \cdots, T-1$,

$$
\alpha_{t+1}(i)=\left[\sum_{j=1}^N \alpha_t(j) a_{j i}\right] b_i\left(o_{t+1}\right), \quad i=1,2, \cdots, N
$$


(3) 终止

$$
P(O \mid \lambda)=\sum_{i=1}^N \alpha_T(i)
$$

前向算法, 步骤 (1) 初始化前向概率, 是初始时刻的状态 $i_1=q_i$ 和观测 $o_1$ 的 联合概率。步骤 (2) 是前向概率的递推公式, 计算到时刻 $t+1$ 部分观测序列为 $o_1, o_2, \cdots, o_t, o_{t+1}$ 且在时刻 $t+1$ 处于状态 $q_i$ 的前向概率, 如图 $10.1$ 所示。在式 (10.16) 的方括弧里, 既然 $\alpha_t(j)$ 是到时刻 $t$ 观测到 $o_1, o_2, \cdots, o_t$ 并在时刻 $t$ 处于状态 $q_j$ 的前向概率, 那么乘积 $\alpha_t(j) a_{j i}$ 就是到时刻 $t$ 观测到 $o_1, o_2, \cdots, o_t$ 并在时刻 $t$ 处于 状态 $q_j$ 而在时刻 $t+1$ 到达状态 $q_i$ 的联合概率。对这个乘积在时刻 $t$ 的所有可能的
#### 统计学习方法中前向算法的例子

![](image/Pasted%20image%2020220831195049.png)
### 后向算法
给定隐马尔可夫模型 $\lambda$, 定义在时刻 $t$ 状态为 $q_i$ 的条件下, 从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1}, o_{t+2}, \cdots, o_T$ 的概率为后向概率, 记作

$$
\beta_t(i)=P\left(o_{t+1}, o_{t+2}, \cdots, o_T \mid i_t=q_i, \lambda\right)
$$

可以用递推的方法求得后向概率 $\beta_t(i)$ 及观测序列概率 $P(O \mid \lambda)$ 。
(观测序列概率的后向算法)
输入: 隐马尔可夫模型 $\lambda$, 观测序列 $O$;
输出: 观测序列概率 $P(O \mid \lambda)$ 。
(1)

$$
\beta_T(i)=1, \quad i=1,2, \cdots, N
$$

(2) 对 $t=T-1, T-2, \cdots, 1$

$$
\beta_t(i)=\sum_{j=1}^N a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j), \quad i=1,2, \cdots, N
$$

后向算法到这一步只用到了第二个观测值，还有第一个观测值没有用到。
因此最后要乘上。
(3)

$$
P(O \mid \lambda)=\sum_{i=1}^N \pi_i b_i\left(o_1\right) \beta_1(i)
$$


### 结合

利用前向概率和后向概率的定义可以将观测序列概率 $P(O \mid \lambda)$ 统一写成

$$
P(O \mid \lambda)=\sum_{i=1}^N \sum_{j=1}^N \alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j), \quad t=1,2, \cdots, T-1
$$


也可以写成：

$$
P(O\mid \lambda) = \sum_{i=1}^N[\alpha_t(i)\beta_t(i)], \quad t=1,2,\cdots, T-1
$$


## 一些概率和期望问题

利用前向概率和后向概率, 可以得到关于单个状态和两个状态概率的计算公式。
1. 给定模型 $\lambda$ 和观测 $O$, 在时刻 $t$ 处于状态 $q_i$ 的概率。记

$$
\gamma_t(i)=P\left(i_t=q_i \mid O, \lambda\right)
$$

可以通过前向后向概率计算。事实上，

$$
\gamma_t(i)=P\left(i_t=q_i \mid O, \lambda\right)=\frac{P\left(i_t=q_i, O \mid \lambda\right)}{P(O \mid \lambda)}
$$

由前向概率 $\alpha_t(i)$ 和后向概率 $\beta_t(i)$ 定义可知:

$$
\alpha_t(i) \beta_t(i)=P\left(i_t=q_i, O \mid \lambda\right)
$$

于是得到:

$$
\gamma_t(i)=\frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)}=\frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \beta_t(j)}
$$

2. 给定模型 $\lambda$ 和观测 $O$, 在时刻 $t$ 处于状态 $q_i$ 且在时刻 $t+1$ 处于状态 $q_j$ 的概 率。记

$$
\xi_t(i, j)=P\left(i_t=q_i, i_{t+1}=q_j \mid O, \lambda\right)
$$

可以通过前向后向概率计算:

$$
\xi_t(i, j)=\frac{P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}{P(O \mid \lambda)}=\frac{P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}{\sum_{i=1}^N \sum_{j=1}^N P\left(i_t=q_i, i_{t+1}=q_j, O \mid \lambda\right)}
$$

而

$$
	P(i_t=q_i,i_{t+1}=q_j,O | \lambda) = \alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)
$$

所以

$$
\xi_t(i, j)=\frac{\alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^N \sum_{j=1}^N \alpha_t(i) a_{i j} b_j\left(o_{t+1}\right) \beta_{t+1}(j)}
$$

3. 将 $\gamma_t(i)$ 和 $\xi_t(i, j)$ 对各个时刻 $t$ 求和, 可以得到一些有用的期望值。
(1) 在观测 $O$ 下状态 $i$ 出现的期望值:

$$
\sum_{t=1}^T \gamma_t(i)
$$

（2）在观测 $O$ 下由状态 $i$ 转移的期望值:

$$
\sum_{t=1}^{T-1} \gamma_t(i)
$$

(3) 在观测 $O$ 下由状态 $i$ 转移到状态 $j$ 的期望值:

$$
\sum_{t=1}^{T-1} \xi_t(i, j)
$$


## 预测问题

### 近似算法
近似算法的想法是, 在每个时刻 $t$ 选择在该时刻最有可能出现的状态 $i_t^* $, 从而得 到一个状态序列 $I^* =\left(i_1^* , i_2^* , \cdots, i_T^* \right)$, 将它作为预测的结果。
给定隐马尔可夫模型 $\lambda$ 和观测序列 $O$, 在时刻 $t$ 处于状态 $q_i$ 的概率 $\gamma_t(i)$ 是

$$
\gamma_t(i)=\frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)}=\frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \beta_t(j)}
$$

在每一时刻 $t$ 最有可能的状态 $i_t^* $ 是

$$
i_t^* =\arg \max_{1 \leqslant i \leqslant N}\left[\gamma_t(i)\right], \quad t=1,2, \cdots, T
$$

从而得到状态序列 $I^* =\left(i_1^* , i_2^* , \cdots, i_T^* \right)$ 。
近似算法的优点是计算简单, 其缺点是不能保证预测的状态序列整体是最有可能 的状态序列, 因为预测的状态序列可能有实际不发生的部分。事实上, 上述方法得到 的状态序列中有可能存在转移概率为 0 的相邻状态, 即对某些 $i, j, a_{i j}=0$ 时。尽管 如此, 近似算法仍然是有用的。

近似算法就是一种贪心的算法，每个时刻都取最有可能的状态，但整体序列并不一定是最优解。

### 维特比算法
维特比算法实际上是用动态规划解隐马尔科夫模型预测问题，用动态规划求解概率最大路径，一条路径对应着一条状态序列。

首先导入两个变量 $\delta$ 和 $\Psi$ 。定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $\left(i_1, i_2, \cdots, i_t\right)$ 中概率最大值为

$$
\delta_t(i)=\max_{i_1, i_2, \cdots, i_{t-1}} P\left(i_t=i, i_{t-1}, \cdots, i_1, o_t, \cdots, o_1 \mid \lambda\right), \quad i=1,2, \cdots, N
$$

后面的部分与前向算法的部分有点类似。

由定义可得变量 $\delta$ 的递推公式:

$$
\begin{aligned}
\delta_{t+1}(i) &=\max_{i_1, i_2, \cdots, i_t} P\left(i_{t+1}=i, i_t, \cdots, i_1, o_{t+1}, \cdots, o_1 \mid \lambda\right) \\\\
&=\max_{1 \leqslant j \leqslant N}\left[\delta_t(j) a_{j i}\right] b_i\left(o_{t+1}\right), \quad i=1,2, \cdots, N ; \quad t=1,2, \cdots, T-1
\end{aligned}
$$

定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $\left(i_1, i_2, \cdots, i_{t-1}, i\right)$ 中概率最大的路径的 第 $t-1$ 个结点为

$$
\Psi_t(i)=\arg \max_{1 \leqslant j \leqslant N}\left[\delta_{t-1}(j) a_{j i}\right], \quad i=1,2, \cdots, N
$$

可以简单理解为找到使得从t-1的j到t的i式子$\delta_{t-1}(j)a_{ji}$最大的j，也就是说$\Psi_t(i)$代表t-1时刻的最佳状态值，如果t时刻的最佳状态值是i的话，那么t-1时刻的最佳状态值就是$\Psi_t(i)$，后面回溯要用到

下面介绍维特比算法。

输入: 模型 $\lambda=(A, B, \pi)$ 和观测 $O=\left(o_1, o_2, \cdots, o_T\right)$;
输出: 最优路径 $I^* =\left(i_1^* , i_2^* , \cdots, i_T^* \right)$ 。
（1）初始化

$$
\begin{gathered}
\delta_1(i)=\pi_i b_i\left(o_1\right), \quad i=1,2, \cdots, N \\\\
\Psi_1(i)=0, \quad i=1,2, \cdots, N
\end{gathered}
$$

前者和前向算法的初始化是一样的。

(2) 递推。对 $t=2,3, \cdots, T$

$$
\begin{array}{ll}
\delta_t(i)=\max_{1 \leqslant j \leqslant N}\left[\delta_{t-1}(j) a_{j i}\right] b_i\left(o_t\right), \quad i=1,2, \cdots, N \\\\
\Psi_t(i)=\arg \max_{1 \leqslant j \leqslant N}\left[\delta_{t-1}(j) a_{j i}\right], \quad i=1,2, \cdots, N
\end{array}
$$


(3) 终止

$$
\begin{gathered}
P^* =\max_{1 \leqslant i \leqslant N} \delta_T(i) \\\\
i_T^* =\arg \max_{1 \leqslant i \leqslant N}\left[\delta_T(i)\right]
\end{gathered}
$$

这里得到的是最后一个时刻的最佳状态值，然后进行回溯。

(4) 最优路径回溯。对 $t=T-1, T-2, \cdots, 1$

$$
i_t^* =\Psi_{t+1}\left(i_{t+1}^* \right)
$$

求得最优路径 $I^* =\left(i_1^* , i_2^* , \cdots, i_T^* \right)$ 。

#### 书上的例子

看一个例子就很容易理解了

![](image/Pasted%20image%2020220831204603.png)
![](image/Pasted%20image%2020220831204613.png)
![](image/Pasted%20image%2020220831204619.png)
#### 代码
```python
def viterbi(obs, states, start_p, trans_p, emit_p):

    V = [{}] # 列表idx代表时间t，字典的键代表状态值，值代表概率

    path = {} # 最佳路径

    for y in states:

        V[0][y] = start_p[y] * emit_p[y].get(obs[0], 1e-5)

        path[y] = [0] # 都初始化为0

    for t in range(1, len(obs)):

        V.append({})

        for y in states:

            em_p = emit_p[y].get(obs[t], 1e-5) # 取出观测值对应的概率

            (prob, state) = max([(V[t-1][y0]*trans_p[y0][y]*em_p, y0) for y0 in states])

            V[t][y] = prob

            path[y] = path[y] + [state] # 记录路径，state是当前时间t状态为y时t-1的最佳状态，也就是从state转移到y的概率最大。如果最后时刻的最佳状态是y，则回溯从y开始，最后的状态也是y。

    (prob, state) = max((V[len(obs)-1][y], y) for y in states) # 求最后时刻的最大概率和状态。

  

    return path[state][1:] + [state] # 初始状态是0，所以去掉第一个0，再加上最后时刻的最大概率的状态，结果就是最佳路径。这里键对值的过程相当于回溯了。

  

A = {

    0 : {0:0.5, 1:0.2, 2:0.3},

    1 : {0:0.3, 1:0.5, 2:0.2},

    2 : {0:0.2, 1:0.3, 2:0.5}

}

B = {

    0: {'红': 0.5, '白': 0.5},

    1: {'红': 0.4, '白': 0.6},

    2: {'红': 0.7, '白': 0.3}

}

π = {0:0.2, 1:0.4, 2:0.4}

viterbi(['红', '白', '红'], [0, 1, 2], π, A, B)
```
拿上面的例子进行实验，思路是完全按照李航老师书的思路来的。还是比较容易理解的，只是回溯的实现不太一样，这个严格来说不能叫做回溯。


## 学习算法


## 参考

> 《统计学习方法》李航
> 
> [https://www.52nlp.cn/hmm-learn-best-practices-one-introduction](https://www.52nlp.cn/hmm-learn-best-practices-one-introduction)
> 
> [https://www.cnblogs.com/pinard/p/6945257.html](https://www.cnblogs.com/pinard/p/6945257.html)


