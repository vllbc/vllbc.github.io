# 14


# **外观数列**

**[https://leetcode-cn.com/problems/count-and-say/](https://leetcode-cn.com/problems/count-and-say/)**

**这题有意思**

**可以打表，不过打表的过程也相当于做出来了**

```python
class Solution:
    def countAndSay(self,n: int) -> str:
        if n == 1:
           return '1'
        s = self.countAndSay(n - 1)
        n,res = 0,''
        for ii,ss in enumerate(s):
            if ss != s[n]:
                res += str(ii-n) + s[n]
                n = ii
        res += str(len(s) - n) + s[-1]
        return res
print(Solution().countAndSay(3))
```

**思路：**

​	**递归，将上一层计算出来的东西作为迭代对象。**


