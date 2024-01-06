# 平衡括号字符串的最少插入次数



# 平衡括号字符串的最少插入次数

## 题目：

[https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/](https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/)

## 思路：

本题和前面的题属于同一系列的，都是平衡括号字符串，不过这个不是1:1 而是1:2

思路还是差不多，不过判断条件需要改变

## 代码：

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        res,temp = 0,0
        for i in s:
            if i == '(':
                temp += 2
                if temp % 2 == 1:
                    res += 1
                    temp -= 1
            if i == ')':
                temp -= 1
                if temp == -1:
                    res += 1
                    temp = 1
        return res + temp
```

开始还是初始化，temp代表需求的右括号的数量

如果有左括号的话，则让右括号的需求+2 因为一个左对应两个右

这里有个难点，如果需求的是奇数的话，则应添加一个右括号，然后让需求减1

如果是右括号，则需求 减1 如果需求的成了-1 的话  则在左边补上左括号 res++ 此时还需要一个右括号，则temp再初始化为1

最后还是输出

Q:为什么最后不是 temp == -2   res += 1   temp=0 呢？

看看这个例子:

`")))))))"`

这是7个右括号，最后减到最后的话，temp是个负数，影响了最后的结果。

所以还是要用原来的那样，-1的时候就进行判断，不用考虑奇偶的问题了
