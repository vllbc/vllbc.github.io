# 打家劫舍


# 打家劫舍

## 打家劫舍I

### 题目：

[https://leetcode-cn.com/problems/house-robber/](https://leetcode-cn.com/problems/house-robber/)

### 思路:

一个简单题，不过踩了特例的坑。。可以暴力解决 也可以动态规划

### 代码:

暴力解决

```python
class Solution:
    def rob(nums):
        if nums == []:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0],nums[1])
        maxs = [] #max[i]代表到i+1家的最大价钱
        maxs.append(nums[0])
        maxs.append(nums[1])
        for i in range(2,len(nums)):
            maxs.append(max(maxs[:i-1])+nums[i]) #从头到这家前面的第二家最大的价钱加上这一家的价钱
        return max(maxs)
```

动态规划

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [0 for i in range(len(nums)+2)] # dp为从第i个房子开始抢 抢到的钱
        for i in range(len(nums)-1,-1,-1):
            dp[i] = max(dp[i+1],dp[i+2]+nums[i])
        return dp[0]  
```

## 打家劫舍II

### 题目：

[https://leetcode-cn.com/problems/house-robber-ii/](https://leetcode-cn.com/problems/house-robber-ii/)

### 思路：

跟上面的题目非常类似，只是加了一个限制条件，就是第一家和最后一家不能同时打劫。

这里先写一个函数，表示从start 到end 范围里面的最大值，然后在主函数里面进行选择

如果打劫第一家，就不能打劫最后一家以及不打劫第一家去打劫最后一家，这两者之间的最大值

### 代码：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        return max(self.dp(0,len(nums)-2,nums),self.dp(1,len(nums)-1,nums))

    def dp(self,start,end,nums):
        dp = [0 for _ in range(len(nums)+2)]
        for i in range(end,start-1,-1):
            dp[i] = max(dp[i+1],dp[i+2]+nums[i])
        return dp[start]

```




