# 分割等和子集


# 分割等和子集

## 题目：

[https://leetcode-cn.com/problems/partition-equal-subset-sum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china](https://leetcode-cn.com/problems/partition-equal-subset-sum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china)

## 思路：

典型的01背包问题，利用套路框架做即可 

注意做了优化，把原本的二维dp降低了一维

## 代码：

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2:
            return False
        s = sum(nums) // 2
        dp = [False for _ in range(s+1)]
        dp[0] = True
        for i in range(1,len(nums)+1): 
            for j in range(s,nums[i-1]-1,-1): # 容量
                dp[j] = dp[j] or dp[j-nums[i-1]] # 用了or操作符
        return dp[s]
```

更一般的套路，定义二维数组，然后二维dp

```python
# i代表前i个物品,j代表背包容量。
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return False
        if sum(nums) % 2:
            return False
        s = sum(nums) // 2
        dp = [[False for _ in range(s+1)] for _ in range(len(nums)+1)]
        for i in range(len(nums)+1):
            dp[i][0] = True # 背包容量为0时 永远都是满的 所以为true
        for i in range(1,len(nums)+1): # 物品个数
            for j in range(1,s+1): # 背包容量，最大为总和的一半，也就是需要求的
                if j - nums[i-1] < 0: # 如果容量小于当前物品的重量
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
            if dp[i][s]: # 剪枝
                return True
        return dp[len(nums)][s]
'''首先，由于i是从 1 开始的，而数组索引是从 0 开始的，所以第i个物品的重量应该是nums[i-1]，这一点不要搞混。
dp[i - 1][j-nums[i-1]]也很好理解：你如果装了第i个物品，就要看背包的剩余重量j - nums[i-1]限制下是否能够被恰好装满。
换句话说，如果j - nums[i-1]的重量可以被恰好装满，那么只要把第i个物品装进去，也可恰好装满j的重量；否则的话，重量j肯定是装不满的。'''
```




