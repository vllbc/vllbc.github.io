# 使括号有效的最少添加



# 使括号有效的最少添加

## 题目：

[https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/)

## 思路：

通过一个值来判断是否匹配

## 代码：

```python
class Solution:
    def minAddToMakeValid(self, S: str) -> int:
        res,temp = 0,0
        for i in S:
            if i == '(':
                temp += 1
            if i == ')':
                temp -= 1
                if temp == -1:
                    temp = 0
                    res += 1
        return res + temp
            
```

如果右括号过多的话，就在左边补一个左括号。这时结果+1

如果一直是左括号的话，res 为0 temp就是应该补的个数

如果都相匹配的话，temp = 0 相应 res也为0
