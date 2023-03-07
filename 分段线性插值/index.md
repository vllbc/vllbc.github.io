# 分段线性插值



# 分段线性插值
利用线性函数作插值
每一段的线性函数：

$$
F1 = \frac{x-x_{i+1}}{x_i-x_{i+1}}f(x_i)+\frac{x-x_i}{x_{i+1}-x_i}f(x_{i+1})
$$


## 代码
```python
import numpy as np
import matplotlib.pyplot as plt
 
#分段线性插值闭包
def get_line(xn, yn):
    def line(x):
        index = -1
         
        #找出x所在的区间
        for i in range(1, len(xn)):
            if x <= xn[i]:
                index = i-1
                break
            else:
                i += 1
         
        if index == -1:
            return -100
         
        #插值
        result = (x-xn[index+1])*yn[index]/float((xn[index]-xn[index+1])) + (x-xn[index])*yn[index+1]/float((xn[index+1]-xn[index]))
         
        return result
    return line
 
xn = [i for i in range(-50,50,10)]
yn = [i**2 for i in xn]
 
#分段线性插值函数
lin = get_line(xn, yn)
 
x = [i for i in range(-50, 40)]
y = [lin(i) for i in x]
 
#画图
plt.plot(xn, yn, 'ro')
plt.plot(x, y, 'b-')
plt.show()
```
