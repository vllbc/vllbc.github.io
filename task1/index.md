# task1



# **pandas实践1**

**在读取数据之前，我修改了表格里面的表头，以便程序的编写。** 

**先从 excel 读取数据,然后看看 shape 了解行数列数,然后调用 info 方法， 看看有没有缺失值，发现并没有缺失值，但题目里说了可能有重复或者格式 不对的数据，因为最主要的是学号,一般学号的长度都是 12 个数字，所以筛 选出不是 12 位数的** 

**`data[data['studentid'].apply(lambda x:len(x)!=12)]`** 

**考虑到可能出现中文的情况，先尝试转化为整数试试** 

**data[‘studentid’] = data[‘studentid’].astype(“int64”)** 



**发现报错了，然后就看见了那个学号是’忘记了’的 最后修改成了** 

**`data[data['studentid'].apply(lambda x:len(x)!=12 or x=='忘记了')]`** 

**将这些数据删除**

 **`data = data.drop(data[data['studentid'].apply(lambda x:len(x)!=12 or x=='忘记了')].index)`** 

**考虑到有重复，重复的两个因素就是姓名和学号，因此进行去重处理** 

**`data.drop_duplicates(subset=['name','studentid'],keep='first',inplace=Tru e)`**

 **此外，对专业的处理，将无用的 xx-x 去掉即可，这里考虑到了正则表达式** 

**`data['class'] = data['class'].apply(lambda s:re.sub(r"[\s*\d*\-*\—*\ － *\–*\/*]?",'',s))`** 

**因为各种各样的-负号千奇百怪，我只能一次次修改后然后统计一下即调用 `data[‘class’].value_counts()` 有没有没有处理到的，然后把那个-符号加进去 还发现了有/号。**

**最后就成了那样，写到这里我有了更好的想法，和下面的 某两个个例有关系。** 

**然后就是那个 maps 表，都简化为简称，对称呼进行统一，用了 apply 方 法 再统计一下，发现了两个专业后面带名字的学长学姐，因为就两个，就把他 们加到 maps 里面了，其实也可以判断名字是否在专业里面，如果在就替换 为空吧。 之后就差不多可以了，数据预处理完毕，按照要求保存即可。**

```python
#数据预处理文件

import pandas as pd
import re

data = pd.read_excel("附件1.xlsx")

#去除错误数据
data = data.drop(data[data['studentid'].apply(lambda x:len(x)!=12 or x=='忘记了')].index)

#去重
data.drop_duplicates(subset=['name','studentid'], keep='first', inplace=True)

data['class'] = data['class'].apply(lambda s:re.sub(r"[\s*\d*\-*\—*\－*\–*\/*]?", '', s))

maps = {
    '智能科学':'智科',
    '云计算':'云计',
    '应用统计学':'统计',
    '信息与计算科学':'信计',
    '智能科学与技术':'智科',
    '应用统计':'统计',
    '软件工程':'软工',
    '信息与计算科学（云计算）':'信计',
    '光电信息与科学':'光电',
    '信计（云计算）':'信计',
    '光电信息科学与工程':'光电',
    '数据科学':'大数据',
    '智科科学':'智科',
    '信计学长':'信计',
    '信计学姐':'信计',
    '统计学':'统计',
    '信息计算与科学':'信计',
    '信计与计算科学':'信计'
}

def replaces(clas):
    if clas in maps.keys():
        return maps[clas]
    else:
        return clas

data['class'] = data['class'].apply(replaces)

res = pd.DataFrame()
res['账号'] = '21aidc' + data['studentid']
res['姓名'] = data['name']
res['密码'] = res['账号']
res['专业'] = data['class']
res.to_excel("result.xlsx", index=False,encoding='utf-8')
```
