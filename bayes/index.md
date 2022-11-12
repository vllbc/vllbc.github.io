# bayes



## 条件概率

$P(B|A) = \frac{P(AB)}{P(A)}$

## 乘法公式
如果P(A) > 0
$P(AB) = P(A)P(B|A)$
如果$P(A_1 \dots A_{n-1})$ > 0
则

$$
\begin{aligned}
P(A_1A_2\dots A_n) = P(A_1A_2\dots A_{n-1})P(A_n | A_1A_2\dots A_{n-1})  \\\\ = P(A_1)P(A_2|A_1)P(A_3|A_1A_2)\dots P(A_n|A_1A_2\dots A_{n-1})
\end{aligned}
$$

其中第一步使用了乘法公式，然后再对前者继续使用乘法公式，以此类推，就可以得到最后的结果。


## 全概率公式

$$
P(A) = \sum_{i=1}^n P(B_i)P(A\lvert B_i)
$$

特例为:

$$
P(A)=P(A\lvert B)P(B) + P(A\lvert \bar{B})P(\bar{B})
$$

全概率公式的意义：
将复杂的事件A划分为较为简单的事件

$$
AB_1,AB_2,\ldots,AB_n
$$

再结合加法公式和乘法公式计算出A的概率
## 贝叶斯公式

先引入一个小例子。

$$
P(X=玩LOL)=0.6;\\\\
P(X=不玩LOL)=0.4
$$

这个概率是根据统计得到或者根据自身经验给出的一个概率值，我们称之为**先验概率(prior probability)**
此外

$$
P(Y=男性\lvert X=玩LOL)=0.8,\quad P(Y=小姐姐\vert X=玩LOL)=0.2\\\\
P(Y=男性\lvert X=不玩LOL)=0.2，\quad P(Y=小姐姐\vert X=不玩LOL)=0.8
$$

求在已知玩家为男性的情况下，他是LOL玩家的概率是多少：
根据贝叶斯准则

$$
P(X=玩LOL\lvert Y=男性)=P(Y=男性\lvert X=玩LOL)\frac{P(X=玩LOL)}{[P(Y=男性\lvert X=玩LOL)P(X=玩LOL)+P(Y=男性\lvert X=不玩LOL)]P(X=不玩LOL)}
$$


分母为全概率公式

下面是贝叶斯公式的推导。

$$
P(B\lvert A)=\frac{P(AB)}{P(A)}=\frac{P(BA)}{P(A)}\iff \frac{P(B)P(A\lvert B)}{\displaystyle \sum_{j=1}^n P(B_j)P(A\lvert B_j)}
$$

贝叶斯公式的意义：
在事件A已经发生的条件下，贝叶斯公式可用来寻找导致A发生各种“原因”Bi的概率。
对于先验概率和后验概率来说，

$$
\begin{aligned}
P(B\lvert A)为后验概率 \\\\
P(B)和P(A)为先验概率  \\\\
P(A\vert B)为可能性
\end{aligned}
$$



## 介绍
朴素贝叶斯属于生成式模型， 其主要用于分类，属于是最简单的概率图模型，主要用到概率论中学到的贝叶斯公式，其中需要对模型进行假设，即贝叶斯假设。

## 贝叶斯假设
条件独立性假设(最简单的概率图模型(有向图))，目的是简化计算

## 推导
对于数据集$\\{(x_i, y_i)\\}^N_{i=1}$，$x_i \in R^p , \quad y_i \in \\{ 0, 1\\}$


$$
\begin{aligned}
\hat{y} &= \arg \maxP(y|X) \\\\
& = \arg \max\frac{P(X,y)}{P(X)}  \\\\ 
& = \arg \max\frac{P(y)P(X|y)}{P(X)} \\\\
& = \arg \maxP(y) P(X|y) \\\\
& = \arg \maxP(y)P(x_1,x_2,\dots x_p| y)
\end{aligned}
$$


其中由于我们的条件独立性假设，因此$P(X|y)$可以写为$\prod_{j=1}^pP(x_j|y)$
即最终的式子就是

$$
\hat{y} = \arg \maxP(y)\prod_{j=1}^p P(x_j|y)
$$

这就是朴素贝叶斯的主要推导。
注意术语：

- $P(y)$为先验概率
- $P(y|X)$为后验概率
- $P(X,y)$为联合概率
- MAP，即最大后验估计，选择有最高后验概率的类。

## 后验概率最大化的含义
$\lambda_{ij}$是将一个真实标记为$c_j$的样本分类为$c_i$所产生的损失。基于后验概率$P(c_i\mid x)$可以获得将样本x分类为$c_i$所产生的期望损失，即条件风险。N为类别数。

$$
R(c_i\mid x) = \sum_{j=1}^N\lambda_{ij}P(c_j\mid x)
$$

如何理解这个式子，对于每个预测的类别$c_i$，$P(c_i\mid x)$ 代表预测正确，其余的代表预测错误，对于每个预测的类别，都可以计算出这个期望损失，期望损失最小的类别就是预测的类别。直观地理解，就是后验概率$P(c_i\mid x)$概率越高，则$R(c_i\mid x)$越小，最小的就是预测的类别，也就是最大的后验概率，当然这是0-1损失的情况，如果$\lambda_{ij}$是其他值就不一定了。

朴素贝叶斯法将实例分到后验概率最大的类中。这等价于期望风险最小化。例如选择 0-1 损失函数即$\lambda_{ij}$:

$$
\lambda_{ij}= \begin{cases}1, & i \neq j  \\\\ 0, & i=j\end{cases}
$$

这时, 总体期望风险函数为

$$
R(h)=E_x[R(h(x)\mid x)]
$$

贝叶斯判定准则：为最小化总体条件风险，只需在每个样本上选择那个能使条件风险$R(c\mid x)$最小的类别标记，即：

$$
h^{*}(x) = \underbrace{\arg \min }\_{c\in \gamma} R(c\mid x)
$$

此时，$h^* (x)$称为贝叶斯最优分类器，与之对应的总体风险$R(h^* )$称为贝叶斯风险。$1-R(h*)$反映了分类器所能达到的最好性能。

如果使用上面的0-1损失函数，此时的条件风险：

$$
R(c\mid x) = 1-P(c\mid x)
$$

于是此时的最优分类器变为了

$$
h^* (x) = \arg \min(1-P(c\mid x)) = \arg \maxP(c\mid x)
$$

这就是后验概率最大化的含义。

## 极大似然估计
在朴素贝叶斯法中, 学习意味着估计 $P\left(Y=c_{k}\right)$ 和 $P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)$ 。可以 应用极大似然估计法估计相应的概率。先验概率 $P\left(Y=c_{k}\right)$ 的极大似然估计是

$$
P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, \quad k=1,2, \cdots, K
$$

设第 $j$ 个特征 $x^{(j)}$ 可能取值的集合为 $\left\\{a_{j 1}, a_{j 2}, \cdots, a_{j S_{j}}\right\\\}$, 条件概率 $P\left(X^{(j)}=a_{j l} \mid Y=\right.$ $c_{k}$ ) 的极大似然估计是

$$
\begin{aligned}
&P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\\\
&j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K
\end{aligned}
$$

式中, $x_{i}^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征; $a_{j l}$ 是第 $j$ 个特征可能取的第 $l$ 个值; $I$ 为指 示函数。

$S_j$为$x^{(j)}$的可能取值数，$K$为类别数。

## 拉普拉斯平滑

用极大似然估计可能会出现所要估计的概率值为 0 的情况。这时会影响到后验概 率的计算结果, 使分类产生偏差。解决这一问题的方法是采用贝叶斯估计。具体地, 条 件概率的贝叶斯估计是

$$
P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)+\lambda}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+S_{j} \lambda}
$$

式中 $\lambda \geqslant 0$ 。等价于在随机变量各个取值的频数上赋予一个正数 $\lambda>0$ 。当 $\lambda=0$ 时就 是极大似然估计。常取 $\lambda=1$, 这时称为拉普拉斯平滑 (Laplacian smoothing)。显然, 对任何 $l=1,2, \cdots, S_{j}, k=1,2, \cdots, K$, 有

$$
\begin{aligned}
&P_{\lambda}\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)>0 \\\\
&\sum_{l=1}^{S_{j}} P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=1
\end{aligned}
$$

同样, 先验概率的贝叶斯估计是

$$
P_{\lambda}\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+\lambda}{N+K \lambda}
$$

## 文本分类
接下来是朴素贝叶斯在文本分类中的运用，这里以简单的二分类问题，情感分析为例。

### 如何定义几个概率？
$P(y=k)$很容易得到，可以只评估带有标签k的文档比例，即

$$
P(y=k) = \frac{N(y=k)}{\sum_iN(y=i)}
$$

$P(x|y=k)= P(x_1,x_2,\dots, x_n | y=k)$
这里假设文档x被表示为一组特征，例如一组它的词$(x_1,x_2,\dots, x_n)$

这里需要两个假设，其中一个是上面提到的贝叶斯假设，即：
- 条件独立假设：特征在给定类的情况下是独立的
- Bag of Words假设：词序无关紧要

直观地说，假设 每个单词出现在类别为k的文档中的概率不依赖上下文，因此得到：

$$
P(x|y=k) = P(x_1,x_2,\dots,x_n|y=k) = \prod_{t=1}^nP(x_t|y=k)
$$

概率$P(x_i|y=k)$为单词$x_i$出现在标签为k的文档中的频率，即

$$
P(x_i|y=k) = \frac{N(x_i, y=k)}{\sum_{t=1}^{|V|}N(x_t,y=k)}
$$

但是有个问题就是有可能会出现$N(x_i, y=k)=0$的情况
![](image/Pasted%20image%2020220726202224.png)

这时就需要拉普拉斯平滑，即在所有的计数中都加入一个新的参数$\delta$，

$$
P(x_i|y=k)=\frac{ {\delta} +  N(x_i, y=k)
    }{\sum\limits_{t=1}^{|V|}( {\delta} + N(x_t, y=k))} =
    \frac{ {\delta} +  N(x_i, y=k)
    }{ {\delta\cdot |V|}   + \sum\limits_{t=1}^{|V|} N(x_t, y=k)}
    ,
$$

直观地说，朴素贝叶斯期望某些词作为类指示符。例如，对于情感分类标记 awesome、 brilliant、 great 将有更高的概率给定正面类别然后负面类别。 类似地，给定负类比正类 ，标记awful, boring, bad的概率更高。

![](image/Pasted%20image%2020220726202626.png)
在实践中，一般都是取log，单调性不变，变为$\log(x, y=k) = \log P(y=k) \sum \log P(x_i|y=k)$


## 补充：贝叶斯估计

易知$P(\theta \mid D)$称为后验概率，有三种估计$\theta$的方法：

- 使用后验分布的密度函数最大值点作为$\theta$的点估计的最大后验估计（MAP）。
- 使用后验分布的中位数作为$\theta$的点估计的后验中位数估计（不常用）。
- 使用后验分布的均值作为$\theta$的点估计的**后验期望估计**。

其中后验期望估计也就是贝叶斯估计。

贝叶斯估计是在MAP上做进一步拓展，不直接估计参数的值，而是允许参数服从一定的概率密度分布，先求出$\theta$的后验分布$p(\theta \mid x)$，然后求出$\theta$的期望值。

![](image/Pasted%20image%2020221012113733.png)

![](image/Pasted%20image%2020221012113741.png)
![](image/Pasted%20image%2020221012113749.png)
![](image/Pasted%20image%2020221012113755.png)



