# python_lq


# **python刷题针对蓝桥杯和leetcode**

常用模块：itertools,collections,datetime,random

random:

```python
import random
random.random() #随机生成0-1随机浮点数
random.randint(n,m) #生成n到m的随机一个数
random.uniform(1.1,5.4) #生成1.1-5.4中间一个浮点数
random.choice(['石头','剪刀','布']) #从序列中随机选一个
random.shuffle(<list>) #打乱序列
random.sample('abcde12345ABCDE',5) #从序列中随机选5个
```

datetime:

```python
import datetime
datetime.date.today() #今天日期，格式为YY-MM-DD
d = datetime.date(2020,12,10) #生成一个日期对象 d.year d.month d.day获取年月日
d.__format__("%Y-%m-%d")
```

collections:

```python
from collections import Counter #计数器
s = "hello-python-hello-world"
a = Counter(s)
print(a) #返回的是一个字典
```





还有python cookbook里面的一些技巧，可以提高效率


