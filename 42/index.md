# 42


# 至少有k个重复字符的最长字串

## 题目：

[https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

## 思路：

利用递归，如果s中字符c的数目小于k,则以c作分割，分成的字串再次调用函数形成递归，然后从众多结果中找寻最大长度的。

## 代码：

```python
class Solution(object):
    def longestSubstring(self, s, k):
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
        return len(s)
```




