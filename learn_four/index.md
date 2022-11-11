# learn_four


# pandas补充学习
推荐网站：[http://joyfulpandas.datawhale.club/Content/Preface.html](http://joyfulpandas.datawhale.club/Content/Preface.html)

pandas核心操作手册：[https://mp.weixin.qq.com/s/l1V5e726XixI0W3EDHx0Nw](https://mp.weixin.qq.com/s/l1V5e726XixI0W3EDHx0Nw)

## pd.join和pd.merge

可以说merge包含了join操作，merge支持两个df间行方向或列方向的拼接操作，默认列拼接，取交集，而join只是简化了merge的行拼接的操作
pandas的merge方法提供了一种类似于SQL的内存链接操作，官网文档提到它的性能会比其他开源语言的数据操作（例如R）要高效。
如果对于sql比较熟悉的话，merge也比较好理解。
merge的参数

- on：列名，join用来对齐的那一列的名字，用到这个参数的时候一定要保证左表和右表用来对齐的那一列都有相同的列名。

- left_on：左表对齐的列，可以是列名，也可以是和dataframe同样长度的arrays。

- right_on：右表对齐的列，可以是列名，也可以是和dataframe同样长度的arrays。

- left_index/ right_index: 如果是True的haunted以index作为对齐的key

- how：数据融合的方法。

- sort：根据dataframe合并的keys按字典顺序排序，默认是，如果置false可以提高表现。
简单举一个刚刚在比赛中里面用到的例子
```python
data = pd.merge(data1, data2, on="carid", how="inner") # 根据carid合并两个数据集
```
可以用`pd.merge`，也可以用`dataframe.merge`，更多的信息可以查阅官方API。
## pd.concat
也是合并dataframe
用法：
```python
pd.concat([df1, df2]) # 纵向合并
pd.concat([df1, df2], axis=1) # 横向合并
```
参数
- ignore_index=True，重新设置合并后的dataframe对象的index值
- sort=False，列的顺序保持原样
- join : {"inner", "outer"}，默认为outer。
## pd.append
```python
# 语法结构
df.append(self, other, ignore_index=False,
          verify_integrity=False, sort=False)
```

- other 是它要追加的其他 DataFrame 或者类似序列内容
- ignore_index 如果为 True 则重新进行自然索引
- verify_integrity 如果为 True 则遇到重复索引内容时报错
- sort 进行排序

### 同结构
将同结构的数据追加在原数据后面
### 不同结构
没有的列会增加，没有的相应内容为空。
### 可以合并追加多个
```python
result = df1.append([df2, df3])
```
### 追加序列
```python
s2 = pd.Series(['X0', 'X1', 'X2', 'X3'],
               index=['A', 'B', 'C', 'D'])
result = df1.append(s2, ignore_index=True)
```
### 追加字典列表
```python
dicts = [{'A': 1, 'B': 2, 'C': 3, 'X': 4},
         {'A': 5, 'B': 6, 'C': 7, 'Y': 8}]
result = df1.append(dicts, ignore_index=True, sort=False)
```

## pd.rename
```python
DataFrame.rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors=‘ignore’)
```
作用就是修改index或者columns的名字。使用时可以指定mapper，然后指定axis，默认axis=0，即修改index
也可以使用index=xxx,columns=xxx，同时进行修改。

## pd.get_dummies
用于构造离散数据的独热编码
用法
```python
training = pd.get_dummies(train_data, columns=["xxx", "xx", "x"])
# 对xxx、xx、x这三列进行onehot编码
```
## pd.melt
直观的看就是将宽数据转化为长数据。转化为variable-value这样的形式。
```python
pandas.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)
```
参数解释：

- frame:要处理的数据集。
- id_vars:不需要被转换的列名。
- value_vars:需要转换的列名，如果剩下的列全部都要转换，就不用写了。
- var_name和value_name是自定义设置对应的列名。
- col_level :如果列是MultiIndex，则使用此级别。

常常与seaborn的FacetGrid一起进行操作，示例：
```python
f = pd.melt(train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False) # col_warp限制一行只有两个，sharx、sharey默认为True,与matplotlib.pyplot.subplots相反。
g = g.map(sns.distplot, "value") # 第一个参数可以自定义函数。
```

### sns.pairplot
seaborn一般与pandas的数据结合，因此不分开做笔记了。
根据官方文档，sns.pairplot是在数据集中绘制成对关系。用来展现变量两两之间的关系，比如线性、非线性、相关等等。
hue参数可以指定分类。与其它plot的hue参数一致。
一般都会使用参数` diag_kind="kde"`，因为对角线上的变量x与y都是一样的，探究关系没有意义，因此展示核密度分布。
```python
seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='hist', markers=None, size=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None)¶

```
数据指定：
> vars : 与data使用，否则使用data的全部变量。参数类型：numeric类型的变量list。
{x, y}_vars : 与data使用，否则使用data的全部变量。参数类型：numeric类型的变量list。
dropna : 是否剔除缺失值。参数类型：boolean, optional

特殊参数：
> kind : {‘scatter’, ‘reg’}, optional Kind of plot for the non-identity relationships.
diag_kind : {‘hist’, ‘kde’}, optional。Kind of plot for the diagonal subplots.

基本参数：
> size : 默认 6，图的尺度大小（正方形）。参数类型：numeric
hue : 使用指定变量为分类变量画图。参数类型：string (变量名)
hue_order : list of strings Order for the levels of the hue variable in the palette
palette : 调色板颜色
markers : 使用不同的形状。参数类型：list
aspect : scalar, optional。Aspect * size gives the width (in inches) of each facet.
{plot, diag, grid}_kws : 指定其他参数。参数类型：dicts



