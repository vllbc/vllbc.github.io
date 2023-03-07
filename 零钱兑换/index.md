# 零钱兑换



# 零钱兑换

[https://leetcode-cn.com/problems/coin-change/](https://leetcode-cn.com/problems/coin-change/)

以我目前的水平做出来有点吃力，看了思路才做出来

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(amount+1):
           for coin in coins: #
               if i >= coin:
                   dp[i] = min(dp[i],dp[i-coin]+1)
        
        return -1 if (dp[-1] == float("inf")) else dp[-1]
```

伪代码如下

```python
# 伪码框架
def coinChange(coins: List[int], amount: int):

    # 定义：要凑出金额 n，至少要 dp(n) 个硬币
    def dp(n):
        # 做选择，选择需要硬币最少的那个结果
        for coin in coins:
            res = min(res, 1 + dp(n - coin))
        return res

    # 题目要求的最终结果是 dp(amount)
    return dp(amount)
```
