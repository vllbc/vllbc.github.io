# Logistic回归



# Logistic回归
## 线性回归
线性回归表达式：

$$
y = w^Tx+b
$$

广义回归模型：

$$
y = g^{-1}(w^Tx+b)
$$

## Sigmoid函数
在分类任务中，需要找到一个联系函数，即g，将线性回归的输出值与实际的标签值联系起来。因此可以使用Sigmoid函数
即：

$$
\delta(z) = \frac{1}{1+e^{-z}}
$$

对数几率其实是一种“sigmoid"函数，它将z值转化为一个接近 0 或 1 的 $y$ 值:

$$
y=\frac{1}{1+e^{-\left(w^{T} x+b\right)}} \rightarrow \operatorname{In} \frac{y}{1-y}=w^{T} x+b
$$

若将y视为样本 $x$ 作为正例的可能性，则1-y是其反例的可能性，两者的比值 $\frac{y}{1-y}$ 称为“几率”，反映了x作为正例的相对可能性，对几率取对 数则得到 $\operatorname{In} \frac{y}{1-y}$ ，可以看出，上式其实是在用线性回归模型的预测结果去逼近真实标记的对数几率。所以该模型也被称作“对数几率回 归”。
## 损失函数
$$
J = -\frac{1}{m}\sum_{i=1}^my_i\log(\hat{y_i})+(1-y_i)\log(1-\hat{y})
$$

实际上可以看作下面交叉熵损失函数形式在二分类问题上的形式：

$$
J = -\frac{1}{m}\sum_{i=1}^my_i\log(\hat{y_i})
$$

这里的$y_i$与$\hat{y_i}$都是向量，其长度就是类别的数量。其中$y_i$代表实际分布，形式上为onehot向量。$\hat{y_i}$是概率分布，为预测的值。

其实这里可以想一下神经网络，对于sigmoid来说，输出层的神经元可以是一个，也可以是两个，如果是一个的话就可以用上面的形式，如果是两个的话可以用下面的这种形式。

也可以这样理解，对于softmax的这种形式，对于二分类我们可以拆分成这样

$$
\begin{cases}
\log \hat{y_i}, \quad y_i=1 \\\\
\log (1-\hat{y_i}), \quad y_i=0
\end{cases}
$$

再结合起来，这样就可以得到逻辑回归的损失函数的结果。
## 与极大似然估计的关系

$$
h(x;\theta) = p(y=1|x;\theta) = \frac{1}{1+e^{-\theta x+b}}
$$


$$
p(y=0|x;\theta) = 1-p(y=1|x;\theta)
$$

则对于单个样本：

$$
p(y|x;\theta) = h(x;\theta)^y(1-h(x;\theta))^{(1-y)}
$$

接下来用极大似然估计估计出参数$\theta$

$$
\begin{aligned}
L(\theta) = \prod_{i=1}^mp(y_i|x_i;\theta)\\\\ =\prod_{i=1}^mh(x_i;\theta)^{y_i}(1-h(x_i;\theta)) ^{1-y_i}
\end{aligned}
$$

则：

$$
l(\theta) = \ln L(\theta) = \sum_{i=1}^my_i\ln (h(x_i;\theta ))+(1-y_i)ln(1-h(x_i|\theta))
$$

极大这个函数，也就是最小化这个函数的负数，也就是上面的损失函数。

## python实现

```python
class LogisticRegression:
    def __init__(self):
        pass
    def sigmoid(self,a):
        res = []
        for x in a:
            if x >= 0:
                res.append(1/(1+np.exp(-x)))
            else:
                res.append(np.exp(x) / (np.exp(x) + 1))
        return np.array(res)
    def train(self, X, y_true, n_iters=100, learning_rate=1):
        """
        根据给定的训练集X和y来训练逻辑回归
        """
        # 第零步：初始化参数
        n_samples, n_features = X.shape
        #👆样本数m和特征量数n分别赋值为X的行数和列数
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # 第一步和第二步：计算输入的特征量和权值的线性组合，使用sigmoid函数
            y_predict = self.sigmoid(np.dot(X,self.weights)+self.bias)
            # 第三步：计算代价值，用于之后计算代价函数值
            cost = (-1/n_samples)*np.sum(y_true*np.log(y_predict+1e-5)+(1-y_true)*(np.log(1-y_predict+1e-5)))
            # 第四步：计算梯度
            dw = (1/n_samples)*np.dot(X.T,(y_predict - y_true))
            db = (1/n_samples)*np.sum(y_predict-y_true)
            # 第五步；更新参数
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            costs.append(cost)
            if i%10 == 0:
                print(f"Cost after iteration {i}:{cost}")

        # return self.weights,self.bias,costs

    def predict(self,X):
        """
        对于测试集X，预测二元分类标签
        """
        y_predict = self.sigmoid(np.dot(X,self.weights)+self.bias)
        return np.array(y_predict)
```
