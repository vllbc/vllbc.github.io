# AdaGrad


![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020220731171907.png)
AdaGrad 直接暴力累加平方梯度，这种做法的缺点就是累加的和会持续增长，会导致学习率变小最终变得无穷小，最后将无法获得额外信息。
