# 丑数系列


# 丑数系列

## 1.丑数

### 题目：

[https://leetcode-cn.com/problems/ugly-number/](https://leetcode-cn.com/problems/ugly-number/)

### 思路：

就是让这个数字不断地除以2.3.5 如果最后变成了1 就说明是个丑数

### 代码：

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        if num<=-231 or num>=231-1:
            return False
        while num >1:
            if num %2 == 0:
                num=int(num/2)
            elif num %3 ==0:
                num =int(num/3)
            elif num %5 ==0:
                num=int(num/5)
            else:
                break
        if num == 1:
            return True
        else:
            return False
```

## 丑数II

### 题目：

[https://leetcode-cn.com/problems/ugly-number-ii/](https://leetcode-cn.com/problems/ugly-number-ii/)

### 思路：

利用三指针，维护i2 i3 i5三个指针分别指向2 3 5

### 代码：

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        res = [1] # 先初始化为1
        i2 = i3 = i5 = 0 # 初始化为0
        for i in range(1,n):
            mins = min(res[i2]*2,res[i3]*3,res[i5]*5) # 从小到大找
            res.append(mins)
            if res[i] == res[i2]*2:
                i2 += 1
            if res[i] == res[i3]*3:
                i3 += 1
            if res[i] == res[i5]*5:
                i5 += 1
        return res[n-1]
```




