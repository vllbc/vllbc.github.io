# Adam算法



# Adam算法

## 背景
作为机器学习的初学者必然会接触梯度下降算法以及SGD，基本上形式如下：

$$
\theta_t = \theta_{t-1} - \alpha \;g(\theta)
$$
其中$\alpha$为学习率，$g(\theta)$为梯度。

简单来说，Adam = Momentum + Adaptive Learning Rate

Momentum实际上就用过去梯度的[moving average](https://www.zhihu.com/search?q=moving+average&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A2576604040%7D)来更新参数。

## moment(矩)
矩在数学中的定义，一阶矩(first moment)就是样本的均值(mean), 二阶矩就是方差（variance）。
## 滑动平均
滑动平均(exponential moving average)，或者叫做指数加权平均(exponentially weighted moving average)，可以用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。在时间序列预测中也常用。

变量 $v$ 在 $t$ 时刻记为 $v_{t} ，\text{可以理解为0到t时刻的平均值} 。\quad \theta_{t}$ 为变量 $v$ 在 $t$ 时刻的取值，即在不使用滑动平均模型时 $v_{t}=\theta_{t}$ ，在使用滑动平均模型后， $v_{t}$ 的更新公式如下:

$$
v_{t}=\beta \cdot v_{t-1}+(1-\beta) \cdot \theta_{t}
$$

上式中， $\beta \in[0,1) ， \beta=0$ 相当于没有使用滑动平均。
这也是RMSProp和Adam等算法里使用的最重要的思想。通过滑动平均来降低梯度的波动值。



## SGD-Momentum
带动量的随机梯度下降方法

它的思路就是计算前面梯度的该变量，每次迭代会考虑前面的计算结果。这样如果在某个维度上波动厉害的特征，会由于“momentum”的影响，而抵消波动的方向（因为波动剧烈的维度每次更新的方向是相反的，momentum能抵消这种波动）。使得梯度下降更加的平滑，得到更快的收敛效率。而后续提出的Adagrad，RMSProp以及结合两者优点的Adam算法都考虑了这种“momentum”的思想。

前面求梯度的过程省略了，后面可以这样写：


$$
\begin{align}
& v_t = \beta v_{t-1} + (1-\beta)g_t \\\\
& \theta = \theta - \alpha v_t
\end{align}
$$

其中$\alpha$为学习率，一般的$\beta$为0.9。v就是动量。

所以，SGD + Momentum可以理解为，利用历史权重梯度矩阵 $W_{i} l(i<t)$ 和当前权重梯度矩 阵 $W_{t} l$ 的加权平均和，来更新权重矩阵 $W$ 。由于 $\beta \in(0,1)$ ，所以随着 $t$ 的增大和 $i$ 的减 小， $\beta^{t-i}$ 会减小，历史权重梯度矩阵 $W_{i} l(i<t)$ 会逐渐减小。通俗来讲，会逐渐遗忘越旧的权重梯度矩阵。

## AdaGrad算法
![](image/Pasted%20image%2020220731171907.png)
AdaGrad直接暴力累加平方梯度，这种做法的缺点就是累加的和会持续增长，会导致学习率变小最终变得无穷小，最后将无法获得额外信息。

## RMSProp算法
![](image/Pasted%20image%2020220731172027.png)
RMSProp和Adagrad算法的最大区别就是在于更新累积梯度值 r 的时候RMSProp考虑加入了一个权重系数 ρ 。
它使用了一个梯度平方的滑动平均。其主要思路就是考虑历史的梯度，对于离得近的梯度重点考虑，而距离比较远的梯度则逐渐忽略。注意图中的是内积。



## Adam
下面看最经典的伪代码：
![](image/Pasted%20image%2020220731173408.png)


adam算法比起adagrad和RMSProp，不仅加入了一阶和二阶moment的计算。而且加入了bias-correction term。以下将展开分析：

### adam的更新率（stepsize)
adam算法中最重要的就是每次迭代的迭代率（step size），他决定了adam算法的效率。根据上 文的算法， step size等于: $\Delta_{t}=\alpha \cdot \widehat{m}_{t} / \sqrt{\hat{v}_{t}}$
1) 当 $\left(1-\beta_{1}\right)>\sqrt{1-\beta_{2}}$ 的时候，它的上界满足不等式:
$\left|\Delta_{t}\right| \leq \alpha \cdot\left(1-\beta_{1}\right) / \sqrt{1-\beta_{2}}$
2) 否则 $\left|\Delta_{t}\right| \leq \alpha$
1）通常发生在数据很稀疏的时候。当数据密集的时候， stepsize会更小。
3) 当 $\left(1-\beta_{1}\right)=\sqrt{1-\beta_{2}}$ 的时候，因为 $\left|\widehat{m}_{t} / \sqrt{\hat{v}_{t}}\right|<1$ 所以，也满足条件 2 的 $\left|\Delta_{t}\right| \leq \alpha$
总结以上3个条件，可以近似得出stepsize 满足 $\left|\Delta_{t}\right| \cong \alpha$
这里的 $\widehat{m}_{t} / \sqrt{\hat{v}_{t}}$ 通常也成为信噪比（Signal-to-noise ratio SNR)，并且满足SND越小， stepsize也越小。

### 初始化偏差矫正项
原算法中的这两行
$$
\begin{aligned}
&\widehat{m}_{t} \leftarrow m_{t} /\left(1-\beta_{1}^{t}\right) \\\\
&\hat{v}_{t} \leftarrow v_{t} /\left(1-\beta_{2}^{t}\right)
\end{aligned}
$$
称为偏差校正项(bias-correction term),他使用了滑动平均值(EMA: exponential moving average)的思想，例如计算二次moment的 $v_{t}=\beta_{2} \cdot v_{t-1}+\left(1-\beta_{2}\right) \cdot g_{t}^{2}$ 可以写成如下的形 式：
$$
v_{t}=\left(1-\beta_{2}\right) \sum_{i=1}^{t} \beta_{2}^{t-i} \cdot g_{i}^{2}
$$
我们的目的是求得 $\mathbb{E}\left[v_{t}\right]$ (EMA) 和二阶moment $\mathbb{E}\left[g_{t}^{2}\right]$ 之间的关系，推导如下:
$$
\begin{aligned}
\mathbb{E}\left[v_{t}\right] &=\mathbb{E}\left[\left(1-\beta_{2}\right) \sum_{i=1}^{t} \beta_{2}^{t-i} \cdot g_{i}^{2}\right] \\\\
&=\mathbb{E}\left[g_{t}^{2}\right] \cdot\left(1-\beta_{2}\right) \sum_{i=1}^{t} \beta_{2}^{t-i}+\zeta \\\\
&=\mathbb{E}\left[g_{t}^{2}\right] \cdot\left(1-\beta_{2}^{t}\right)+\zeta
\end{aligned}
$$
最后得出
$\mathbb{E}\left[g_{t}^{2}\right]=\frac{\mathbb{E}\left[v_{t}\right]-\zeta}{\left(1-\beta_{2}^{t}\right)}$ 通常可以忽略常数 $\zeta$ 。得出

$$
\bar{v_t} = \frac{v_t}{1-\beta_2^t}
$$


**综上所述，Adam 优化器可以根据历史梯度的震荡情况和过滤震荡后的真实历史梯度对变量进行更新**



