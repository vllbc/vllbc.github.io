# 最长递增子序列



# 最长递增子序列

## 题目：

[https://leetcode-cn.com/problems/longest-increasing-subsequence/](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

## 思路：

动态规划 定义dp[i]为到nums[i]的最长递增子序列的长度，全部都初始化为1,因为本身就是长度为1的递增子序列

## 代码：

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1 for _ in range(len(nums))]
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```
