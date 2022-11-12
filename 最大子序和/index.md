# 最大子序和


# 最大子序和

[https://leetcode-cn.com/problems/maximum-subarray/](https://leetcode-cn.com/problems/maximum-subarray/)

一开始直接暴力，结果tle了最后

```python
class Solution:
    def maxSubArray(nums):
        res = -float('inf')
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                res = max(res,sum(nums[i:j+1]))
        return res

```

这说明在leetcode尽量不要嵌套循环，大概率Tle

```python
class Solution:
    def maxSubArray(nums):
        for i in range(1,len(nums)):
            maxs = max(nums[i-1]+nums[i],nums[i])
            nums[i] = maxs
        return max(nums)
```

最后巧妙地利用了替换的思想，将每次相加的值和当前比较，并将当前替换为较大的那个值，最后求整个列表的最大值。


