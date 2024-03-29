# 关联规则概念



参考：[https://www.cnblogs.com/bill-h/p/14863262.html](https://www.cnblogs.com/bill-h/p/14863262.html)

大家可能听说过用于宣传数据挖掘的一个案例:啤酒和尿布；据说是沃尔玛超市在分析顾客的购买记录时，发现许多客户购买啤酒的同时也会购买婴儿尿布，于是超市调整了啤酒和尿布的货架摆放，让这两个品类摆放在一起；结果这两个品类的销量都有明显的增长；分析原因是很多刚生小孩的男士在购买的啤酒时，会顺手带一些婴幼儿用品。

不论这个案例是否是真实的，案例中分析顾客购买记录的方式就是关联规则分析法Association Rules。

关联规则分析也被称为购物篮分析，用于分析数据集各项之间的关联关系。

## 项集

item的集合，如集合{牛奶、麦片、糖}是一个3项集，可以认为是购买记录里物品的集合。

## 频繁项集

顾名思义就是频繁出现的item项的集合。如何定义频繁呢？用比例来判定，关联规则中采用支持度和置信度两个概念来计算比例值

## 支持度（support)

共同出现的项在整体项中的比例。以购买记录为例子，购买记录100条，如果商品A和B同时出现50条购买记录（即同时购买A和B的记录有50），那边A和B这个2项集的支持度为50%


$$
Support(A\cap B) = \frac{Freq(A\cap B)}{N}
$$

## 置信度（Confidence）

购买A后再购买B的条件概率，根据贝叶斯公式，可如下表示：


$$
Confidence = \frac{Freq(A\cap B)}{Freq(A)}
$$

## 提升度

为了判断产生规则的实际价值，即使用规则后商品出现的次数是否高于商品单独出现的评率，提升度和衡量购买X对购买Y的概率的提升作用。如下公式可见，如果X和Y相互独立那么提升度为1，提升度越大，说明X->Y的关联性越强

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020221107221749.png)

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020230610195057.png)
