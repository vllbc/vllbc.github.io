# 字符串的排列



# 字符串的排列

## 题目：

[https://leetcode-cn.com/problems/permutation-in-string/](https://leetcode-cn.com/problems/permutation-in-string/)

## 思路：

滑动窗口加字典

## 代码：

```python
class Solution(object):
    def checkInclusion(self, s1, s2):
        counter1 = collections.Counter(s1)
        N = len(s2)
        left = 0
        right = len(s1) - 1
        counter2 = collections.Counter(s2[0:right])
        while right < N:
            counter2[s2[right]] += 1
            if counter1 == counter2:
                return True
            counter2[s2[left]] -= 1
            if counter2[s2[left]] == 0:
                del counter2[s2[left]]
            left += 1
            right += 1
        return False
```
