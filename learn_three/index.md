# learn_three



# 记录一下实训学到的内容

## 布尔索引

布尔索引不能使用and or not ，只能用& | ~ 因为只能用位操作符

### 花哨索引

```python
arr = np.arange(32).reshape((8, 4))
arr
```
```
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
```
```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
```
```
array([ 4, 23, 29, 10])
```

### 更常用的方式为
```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]] # 行，列重置顺序
```
```
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])
```

## pandas.cut
```python
import pandas as pd
import numpy as np
cars = pd.read_csv("second_cars_info.csv",encoding="gbk")
final_ = [cars.Sec_price.min()] + list(np.linspace(10,100,10)) + [cars.Sec_price.max()]
pd.cut(cars["Sec_price"],bins=final_).value_counts().sort_index() # 对区间进行排序
# labels参数给每个区间贴上标签
```
## .str的用法
可以对行列进行python字符串一样的操作
```python
stud_alcoh[["Mjob",'Fjob']].apply(lambda x:x.str.upper())
# 对Mjob Fjob两列变成大写
# 也可以用applymap
stud_alcoh[["Mjob",'Fjob']].applymap(lambda x:x.upper()) #区别就在于.str 
# 因为applymap只能对dataframe进行操作,apply可以对Dataframe和Series进行操作。
```
## groupby与apply灵活运用
```python
coffee = pd.read_excel("coffee.xlsx")
coffee.groupby("区域").apply(lambda x:x["销售额"].sum()) # 如果是DataFramegroupby对象调用apply，转换函数处理的就是每一个分组（dataframe）
# 等价于 coffee.groupby("区域")['销售额'].sum()
```

### 求各个区域的 销售总和、平均销售额、预计销售额与实际销售额的差异总和
```python
def transfer(x_):# x_类型是dataframe
    res = pd.Series()
    res["销售总和"] = x_["销售额"].sum()
    res["平均销售额"] = x_['销售额'].mean()
    res['差值'] = ( x_["销售额"] - x_['预计销售额']).mean()
    return res # 返回的是一个series
coffee.groupby("区域").apply(transfer)
```
### 求出个区域中，高于该区域平均利润的记录
```python
def transfer(x): # x(dataframe) 
    mean_value = x["利润额"].mean()
    condition = x["利润额"] > mean_value
    return x[condition] # dataframe
c = coffee.groupby("区域").apply(transfer) # 二维索引
# c.loc[("Central",0)]
```
### groupby也可以嵌套
```python
def top_3_data(x):# x 表示DataFrame 每个区域的数据集
    res = x.groupby("产品名称").sum().sort_values("销售额",ascending=False).iloc[:3]
    return res
coffee.groupby("区域").apply(top_3_data)
```
## nlargest nsmallest
求每一列最大的几个多最小的几个，一般配合`groupby`和`apply`使用

## pd.Grouper
`grouper = pd.Grouper(key="订单日期",freq="M")` 按月进行分组 。`freq="Y"`就是按年分组
```python
def top_sale_month(x): #x 表示DataFrame 每个产品的数据集
    grouper = pd.Grouper(key="订单日期",freq="M")
    return x.groupby(grouper).sum().nlargest(1,"销售额")
coffee.groupby("产品名称").apply(top_sale_month).reset_index()
```
`reset_index()`方法就是按照原来的index显示，不然就是按照分组的结果展示
## plt.xticks plt.yticks
可以理解为自定义x轴的坐标
```python
plt.figure(figsize=(12,8))
width = 0.2
plt.bar(np.arange(4)+0,np.random.randint(3,10,(4)),color='r',width=width)
plt.bar(np.arange(4)+width*1,np.random.randint(3,10,(4)),color='g',width=width)
plt.bar(np.arange(4)+width*2,np.random.randint(3,10,(4)),color='b',width=width)
plt.xticks(np.arange(4)+width/2,list("abcd")) # 将0 1 2 3 替换为 a b c d
plt.show()
```

## np.where
`np.where(condition, [x, y])`,这里三个参数,其中必写参数是condition(判断条件),后边的x和y是可选参数.那么这三个参数都有怎样的要求呢?

`condition：array_like，bool `,当为True时，产生x，否则产生y
情况1：
```python
np.where([[True, False], [True, True]],

[[1, 2], [3, 4]],

[[9, 8], [7, 6]])
```
返回：
```
array([[1, 8],

[3, 4]])
```
条件中第0个元素中的第0个元素是true,那么取x中的相对应元素1;

条件中第0个元素中的第1个元素是false,那么取y中的相对应元素8;

条件中第1个元素中的第0个元素是ture,那么取x中相对应的元素3;

条件中第1个元素中的第1个元素是ture,那么取x中相对应的元素4;

所以最后的结果中取出的元素是1,8,3,4.
情况2：
```python
x = np.arange(9.).reshape(3, 3)
np.where(x>5) # 返回的是索引
```
```
(array([2, 2, 2], dtype=int64), array([0, 1, 2], dtype=int64))
```
第一个array是行坐标，第二个array为列坐标。
不想要索引想要具体的数值也很简单

```python
x[np.where(x>5)]
```

```
array([6., 7., 8.])
```

```python
np.where(x < 5, x, -1)
```

```
array([[ 0.,  1.,  2.],
       [ 3.,  4., -1.],
       [-1., -1., -1.]])
```
可见小于五的部分不变，大于5的则变成了-1
np.where常用于pandas的Series中。
