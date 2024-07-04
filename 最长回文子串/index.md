# 最长回文子串



# 最长回文子串

## 题目：

​	[https://leetcode-cn.com/problems/longest-palindromic-substring/](	https://leetcode-cn.com/problems/longest-palindromic-substring/)

## 思路：

​	一开始暴力解法，比较好想，结果超时了哎，后来看见了标签是动态规划，才知道不能暴力

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        maxs = -float("inf")
        res = collections.defaultdict(list)
        left,right = 0,len(s)-1
        while left < right:
            for i in range(left,right+2):
                if s[left:i] == s[left:i][::-1]:
                    maxs = max(maxs,len(s[left:i]))
                    res[maxs].append(s[left:i])
            left += 1
        return max(res[max(res.keys())],key=len)
```

也用到了双指针，超时在情理之中。

后来用到了动态规划

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:

        if len(s) <= 1:
            return s
        length = len(s)
        dp = [[False for _ in range(length)] for _ in range(length)]
        for i in range(length):
            dp[i][i] = True
        start = 0
        max_len = 1
        for j in range(1, length):
            for i in range(0, j):
                if s[i] == s[j]:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = False

                if dp[i][j]:
                    cur_len = j - i + 1
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
        return s[start:start + max_len]
```
