# 把数字翻译成字符串



# 把数字翻译成字符串

## 题目：

[https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

## 思路：

dp思想，不用管是什么字符，定义dp[i]为长度为i时 有多少个方法

## 代码:

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        if len(s) < 2:
            return 1
        dp = [0] * len(s)
        dp[0] = 1
        dp[1] = 2 if int(s[0] + s[1]) < 26 else 1
        for i in range(2,len(s)):
            dp[i] = dp[i-1] + dp[i-2] if int(s[i-1] + s[i]) < 26 and s[i-1] != '0' else dp[i-1]
        return dp[-1]
```

注意如果长度小于等于1 则直接返回1

如果不是26个英文字母里面的 则dp[i] = dp[i-1] 说明方法次数并不改变

注意有首位为0的情况 所以要int一下
