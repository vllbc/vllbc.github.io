# hinge loss


在机器学习中，**hinge loss**是一种损失函数，它通常用于"maximum-margin"的分类任务中，如支持向量机。数学表达式为：

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020230313232338.png)

其中 $\hat{y}$ 表示预测输出，通常都是软结果（就是说输出不是0，1这种，可能是0.87。）， $y$ 表示正确的类别。
-   如果 $\hat{y}y<1$ ，则损失为： $1-\hat{y}y$
-   如果$\hat{y}y>1$ ，则损失为：0

