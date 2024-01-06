# 合并区间



# 合并区间

## 题目：

​	[https://leetcode-cn.com/problems/merge-intervals/](https://leetcode-cn.com/problems/merge-intervals/)

## 思路：
​		一开始思路想的是，根据每一个区间的left排序后，然后比较每一个数，再向前更新，然后写了半天，一直WA，感觉这个思路不太行了

## 代码：

​	先贴上错误的代码：

```python
class Solution:
    def merge(self, res: List[List[int]]) -> List[List[int]]:
        
        if not res:
            return []
        res.sort(key=lambda i:i[0])
        n = len(res) - 1
        for i in range(0,n):
            if res[i][1] < res[i+1][0]:
                continue
            else:
                res[i+1] = [res[i][0],max(res[i][1],res[i+1][1])]
                res[i] = res[i-1]
        ress = []
        for i in res:
            if i not in ress:
                ress.append(i)
        return ress
```

在`[[1,4],[0,2],[3,5]]`

​	出错了

输出：

`[[3,5],[0,5]]`

预期结果：

`[[0,5]]`

应该是思路的错误

后来觉得不应该在原数组上操作

又改了如下，终于过了

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda i:i[0])
        res = []
        for i in intervals:
            if len(res) == 0 or res[-1][1] < i[0]:
                res.append(i)
            else:
                res[-1][1] = max(res[-1][1],i[1])
        return res
```

这个思路就是先创造一个空数组res

然后如果数组为空或者题设的条件不成立的时候，把原数组的值加进去，要是条件成立的话，则将目前区间的right改为目前区间的right和原数组的right之间的最大值，预防`[[1,4],[2,3]]`这种情况。注意这个也是按left排序的

我上面代码的思路和这个是一样的，看来类似的题目尽量不要在原数组上面操作，除非题目要求
