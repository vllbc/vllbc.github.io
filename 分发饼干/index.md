# 分发饼干



# 分发饼干

[https://leetcode-cn.com/problems/assign-cookies/](https://leetcode-cn.com/problems/assign-cookies/)



```python
class Solution:
    def findContentChildren(g, s) -> int:
        g = sorted(g)
        s = sorted(s)
        n = 0
        for i in range(len(s)):
            if g[n] <= s[i]:
                n += 1
            if n == len(g):
                return n
        return n
```

贪心算法的题目，考虑局部最优
