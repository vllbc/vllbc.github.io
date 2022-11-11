# 精确率和召回率


## 精确率和召回率

### 混淆矩阵
- True Positive(真正, TP)：将正类预测为正类数.
- True Negative(真负 , TN)：将负类预测为负类数.
- False Positive(假正, FP)：将负类预测为正类数 
- False Negative(假负 , FN)：将正类预测为负类数 

### 精确率

$$
P = \frac{TP}{TP+FP}
$$





## 准确率


$$
ACC = \frac{TP+TN}{TP+TN+FP+FN}
$$


## 召回率


$$
R = \frac{TP}{TP+FN}
$$


## F1


$$
\frac{2}{F_1} = \frac{1}{P} + \frac{1}{R}
$$


## 区别



**精确率（查准率）：在所有预测为正的样本中（分母），真正为正的有多少（分子）。**

**召回率（查全率）：在所有实际为正的样本中（分母），成功预测出来的有多少（分子）**

![img](https://pica.zhimg.com/80/d701da76199148837cfed83901cea99e_720w.jpg?source=1940ef5c)

![img](https://pica.zhimg.com/80/d701da76199148837cfed83901cea99e_720w.jpg?source=1940ef5c)


