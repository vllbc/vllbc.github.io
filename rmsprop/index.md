# RMSProp


![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220731172027.png)
RMSProp 和 Adagrad 算法的最大区别就是在于更新累积梯度值 r 的时候 RMSProp 考虑加入了一个权重系数 ρ 。
它使用了一个梯度平方的滑动平均。其主要思路就是考虑历史的梯度，对于离得近的梯度重点考虑，而距离比较远的梯度则逐渐忽略。注意图中的是内积。
