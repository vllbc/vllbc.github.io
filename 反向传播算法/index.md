# 反向传播算法



反向传播算法遵循两个法则：梯度下降法则和链式求导法则。

梯度下降法则不用多说，记住一切的目的就是为了减小损失，即朝着局部最小值点移动。链式求导也就是高数中学的那一套。具体地看一个推导就可以了，反向传播需要一步一步来，要搞清楚每一步在做什么。

这里以输出层为sigmoid激活函数的神经网络为例子。

参数$w$和$b$梯度的求解：

$$
\frac{\partial J}{\partial w}=\frac{1}{m}\sum\limits_{i=1}^{m}\frac{\partial J}{\partial a^{(i)}}\frac{\partial a^{(i)}}{\partial  z^{(i)}}\frac{\partial z^{(i)}}{\partial w}
$$

$$
\frac{\partial J}{\partial a^{(i)}}= -\frac{y}{a^{(i)}}+\frac{1-y}{1-a^{(i)}}
$$

$$
\frac{\partial g(z)}{\partial z}=-\frac{1}{(1+e^{-z})^2}(-e^{-z})=\frac{e^{-z}}{1+e^{-z}}=\frac{1}{1+e^{-z}}\times(1-\frac{1}{1+e^{-z}})=g(z)(1-g(z))
$$

所以

$$
\frac{\partial a^{(i)}}{\partial  z^{(i)}}=a^{(i)}(1-a^{(i)})
$$

$$
\frac{\partial z^{(i)}}{\partial w}=x^{(i)}
$$

可得，

$$
\frac{\partial J}{\partial w}=\frac{1}{m}\sum\limits_{i=1}^{m}(a^{(i)}-y)x^{(i)}
$$

求和可以使用numpy的dot函数通过内积计算来实现。

同样地，推导可得，

$$
\frac{\partial J}{\partial b}=\frac{1}{m}\sum\limits_{i=1}^{m}(a^{(i)}-y)
$$

多隐藏层也一样，不过是当前层的激活值作为下一层的输入。慢慢推导即可。
实现的过程不必这么复杂和小心，只需要定义不同层的前向传播和反向传播即可，然后按照积木一样搭建，比如定义Linear层、sigmoid层等，定义前向传播和反向传播，注意输入输出就好，然后搭建起来就是神经网络了。
