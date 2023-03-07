# learn_one



## 导入库


```python
import pandas as pd
import numpy as np
```

## 一维数组


```python
arr = [0, 1, 2, 3, 4]
s1 = pd.Series(arr)  # 如果不指定索引，则默认从 0 开始
s1
```




    0    0
    1    1
    2    2
    3    3
    4    4
    dtype: int64




```python
n = np.random.randn(5)  # 创建一个随机 Ndarray 数组

index = ['a', 'b', 'c', 'd', 'e']
s2 = pd.Series(n, index=index)
s2
```




    a    0.647546
    b    0.197186
    c    0.590904
    d   -0.422565
    e   -0.122490
    dtype: float64



## 用字典创建一维数组


```python
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}  # 定义示例字典
s3 = pd.Series(d)
s3
```




    a    1
    b    2
    c    3
    d    4
    e    5
    dtype: int64



## Series 基本操作


```python
print(s1)  # 以 s1 为例

s1.index = ['A', 'B', 'C', 'D', 'E']  # 修改后的索引
s1
```

    0    0
    1    1
    2    2
    3    3
    4    4
    dtype: int64





    A    0
    B    1
    C    2
    D    3
    E    4
    dtype: int64




```python
s4 = s3.append(s1)  # 将 s1 拼接到 s3
s4
```




    a    1
    b    2
    c    3
    d    4
    e    5
    A    0
    B    1
    C    2
    D    3
    E    4
    dtype: int64




```python
print(s4)
s4 = s4.drop('e')  # 删除索引为 e 的值
s4
```

    a    1
    b    2
    c    3
    d    4
    e    5
    A    0
    B    1
    C    2
    D    3
    E    4
    dtype: int64





    a    1
    b    2
    c    3
    d    4
    A    0
    B    1
    C    2
    D    3
    E    4
    dtype: int64




```python
s4['A'] = 6  # 修改索引为 A 的值 = 6
s4
```




    a    1
    b    2
    c    3
    d    4
    A    6
    B    1
    C    2
    D    3
    E    4
    dtype: int64




```python
s4.add(s3)#如果索引不同则为NAN sub为减，mul为乘，div为除
```




    A    NaN
    B    NaN
    C    NaN
    D    NaN
    E    NaN
    a    2.0
    b    4.0
    c    6.0
    d    8.0
    e    NaN
    dtype: float64




```python
print(s4.median(),#中位数
s4.sum(),#求和
s4.max(),#求最大值
s4.min())#求最小值
```

    3.0 26 6 1


## DataFrame


```python
dates = pd.date_range('today', periods=6)  # 定义时间序列作为 index
num_arr = np.random.randn(6, 4)  # 传入 numpy 随机数组
columns = ['A', 'B', 'C', 'D']  # 将列表作为列名
df1 = pd.DataFrame(num_arr, index=dates, columns=columns)
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-13 20:54:24.760775</th>
      <td>-0.403545</td>
      <td>0.643881</td>
      <td>-1.697286</td>
      <td>-0.616257</td>
    </tr>
    <tr>
      <th>2020-11-14 20:54:24.760775</th>
      <td>0.102470</td>
      <td>1.451324</td>
      <td>-0.269714</td>
      <td>-0.316194</td>
    </tr>
    <tr>
      <th>2020-11-15 20:54:24.760775</th>
      <td>0.342680</td>
      <td>-0.137238</td>
      <td>-0.785406</td>
      <td>-0.441022</td>
    </tr>
    <tr>
      <th>2020-11-16 20:54:24.760775</th>
      <td>0.130079</td>
      <td>1.929569</td>
      <td>-0.756832</td>
      <td>-2.490272</td>
    </tr>
    <tr>
      <th>2020-11-17 20:54:24.760775</th>
      <td>1.774664</td>
      <td>1.037605</td>
      <td>0.275989</td>
      <td>-0.982924</td>
    </tr>
    <tr>
      <th>2020-11-18 20:54:24.760775</th>
      <td>0.952462</td>
      <td>1.666130</td>
      <td>-0.920394</td>
      <td>-1.358411</td>
    </tr>
  </tbody>
</table>

</div>




```python
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(data, index=labels)
df2#df2.head()查看前几行 df2.tail()查看后几行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>f</th>
      <td>2.0</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df2.columns,df2.index,df2.values #查看索引.列名和值
```




    (Index(['age', 'animal', 'priority', 'visits'], dtype='object'),
     Index(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype='object'),
     array([[2.5, 'cat', 'yes', 1],
            [3.0, 'cat', 'yes', 3],
            [0.5, 'snake', 'no', 2],
            [nan, 'dog', 'yes', 3],
            [5.0, 'dog', 'no', 2],
            [2.0, 'cat', 'no', 3],
            [4.5, 'snake', 'no', 1],
            [nan, 'cat', 'yes', 1],
            [7.0, 'dog', 'no', 2],
            [3.0, 'dog', 'no', 1]], dtype=object))




```python
df2.describe()#数据处理常用
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.437500</td>
      <td>1.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.007797</td>
      <td>0.875595</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.375000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.625000</td>
      <td>2.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
df2.T #逆
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>i</th>
      <th>j</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>2.5</td>
      <td>3</td>
      <td>0.5</td>
      <td>NaN</td>
      <td>5</td>
      <td>2</td>
      <td>4.5</td>
      <td>NaN</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>animal</th>
      <td>cat</td>
      <td>cat</td>
      <td>snake</td>
      <td>dog</td>
      <td>dog</td>
      <td>cat</td>
      <td>snake</td>
      <td>cat</td>
      <td>dog</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>priority</th>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
    </tr>
    <tr>
      <th>visits</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df2.sort_values(by='age')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>f</th>
      <td>2.0</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
    </tr>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df2['age']#显示age列 == df2.age
```




    a    2.5
    b    3.0
    c    0.5
    d    NaN
    e    5.0
    f    2.0
    g    4.5
    h    NaN
    i    7.0
    j    3.0
    Name: age, dtype: float64




```python
df2[['age','animal']] #多列查询
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>f</th>
      <td>2.0</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
    </tr>
  </tbody>
</table>

</div>




```python
df2.iloc[1:3]#查询2，3行 ==df2[1:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3 = df2.copy()#拷贝一份
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>f</th>
      <td>2.0</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.isnull()#检测是否为空，为空返回True
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>b</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>c</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>d</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>e</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>f</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>g</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>h</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>i</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>j</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

</div>




```python
num = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=df3.index)#建立数据，注意索引和df3一致

df3['No.'] = num  # 添加以 'No.' 为列名的新数据列
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>2.0</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.iat[1,1] = 2 #根据索引改变值，这里相当于二维数组了
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.loc['f', 'age'] = 1.5 #根据标签来修改
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.mean()#求平均值
```




    age       3.375
    visits    1.900
    No.       4.500
    dtype: float64




```python
df3['visits'].sum()#对visits列进行求和
```




    19




```python
df4 = df3.copy()
print(df4)
df4.fillna(value=3)#如果是NAN补充为3
```

       age animal priority  visits  No.
    a  2.5    cat      yes       1    0
    b  3.0      2      yes       3    1
    c  0.5  snake       no       2    2
    d  NaN    dog      yes       3    3
    e  5.0    dog       no       2    4
    f  1.5    cat       no       3    5
    g  4.5  snake       no       1    6
    h  NaN    cat      yes       1    7
    i  7.0    dog       no       2    8
    j  3.0    dog       no       1    9





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>3.0</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>h</th>
      <td>3.0</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
df5 = df3.copy()
print(df5)
df5.dropna(how='any')  # 任何存在 NaN 的行都将被删除
```

       age animal priority  visits  No.
    a  2.5    cat      yes       1    0
    b  3.0      2      yes       3    1
    c  0.5  snake       no       2    2
    d  NaN    dog      yes       3    3
    e  5.0    dog       no       2    4
    f  1.5    cat       no       3    5
    g  4.5  snake       no       1    6
    h  NaN    cat      yes       1    7
    i  7.0    dog       no       2    8
    j  3.0    dog       no       1    9





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})

print(left)
print(right)

# 按照 key 列对齐连接，只存在 foo2 相同，所以最后变成一行
pd.merge(left, right, on='key')
```

        key  one
    0  foo1    1
    1  foo2    2
        key  two
    0  foo2    4
    1  foo3    5





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo2</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.to_csv('animal.csv')
print("写入成功.")
```

    写入成功.



```python
df_animal = pd.read_csv('animal.csv')
df_animal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f</td>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>g</td>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>h</td>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>i</td>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>j</td>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>




```python
df3.to_excel('animal.xlsx', sheet_name='Sheet1')
print("写入成功.")
```

    写入成功.



```python
pd.read_excel('animal.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>animal</th>
      <th>priority</th>
      <th>visits</th>
      <th>No.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.5</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3.0</td>
      <td>2</td>
      <td>yes</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.5</td>
      <td>snake</td>
      <td>no</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>dog</td>
      <td>yes</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>f</th>
      <td>1.5</td>
      <td>cat</td>
      <td>no</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>g</th>
      <td>4.5</td>
      <td>snake</td>
      <td>no</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>h</th>
      <td>NaN</td>
      <td>cat</td>
      <td>yes</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>i</th>
      <td>7.0</td>
      <td>dog</td>
      <td>no</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>j</th>
      <td>3.0</td>
      <td>dog</td>
      <td>no</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

</div>

```

```



## 进阶部分


```python
dti = pd.date_range(start='2018-01-01', end='2018-12-31', freq='D')
s = pd.Series(np.random.rand(len(dti)), index=dti)
s
```




    2018-01-01    0.201676
    2018-01-02    0.502547
    2018-01-03    0.170739
    2018-01-04    0.725599
    2018-01-05    0.550199
    2018-01-06    0.032421
    2018-01-07    0.699385
    2018-01-08    0.486797
    2018-01-09    0.829427
    2018-01-10    0.180882
    2018-01-11    0.038766
    2018-01-12    0.277279
    2018-01-13    0.549932
    2018-01-14    0.053058
    2018-01-15    0.031786
    2018-01-16    0.418983
    2018-01-17    0.118010
    2018-01-18    0.396846
    2018-01-19    0.337507
    2018-01-20    0.357052
    2018-01-21    0.715088
    2018-01-22    0.147573
    2018-01-23    0.444479
    2018-01-24    0.354306
    2018-01-25    0.054420
    2018-01-26    0.024047
    2018-01-27    0.432698
    2018-01-28    0.532165
    2018-01-29    0.262547
    2018-01-30    0.727870
                    ...   
    2018-12-02    0.855610
    2018-12-03    0.682516
    2018-12-04    0.991494
    2018-12-05    0.935902
    2018-12-06    0.893206
    2018-12-07    0.884918
    2018-12-08    0.285479
    2018-12-09    0.317185
    2018-12-10    0.449078
    2018-12-11    0.266593
    2018-12-12    0.432669
    2018-12-13    0.065663
    2018-12-14    0.524504
    2018-12-15    0.164935
    2018-12-16    0.347022
    2018-12-17    0.127325
    2018-12-18    0.406984
    2018-12-19    0.698307
    2018-12-20    0.135198
    2018-12-21    0.153526
    2018-12-22    0.312617
    2018-12-23    0.965893
    2018-12-24    0.957769
    2018-12-25    0.449219
    2018-12-26    0.037703
    2018-12-27    0.640956
    2018-12-28    0.434779
    2018-12-29    0.750819
    2018-12-30    0.471872
    2018-12-31    0.566274
    Freq: D, Length: 365, dtype: float64




```python
# 周一从 0 开始
s[s.index.weekday == 2].sum()#求所有周三的和
```




    26.097402479556862




```python
s.resample('M').mean()# 每个月值的平均值
```




    2018-01-31    0.351720
    2018-02-28    0.555833
    2018-03-31    0.510938
    2018-04-30    0.524954
    2018-05-31    0.477990
    2018-06-30    0.584583
    2018-07-31    0.531724
    2018-08-31    0.494037
    2018-09-30    0.579968
    2018-10-31    0.532052
    2018-11-30    0.471283
    2018-12-31    0.518058
    Freq: M, dtype: float64
