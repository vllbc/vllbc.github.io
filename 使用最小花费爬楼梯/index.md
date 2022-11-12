# 使用最小花费爬楼梯




# [ 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

每日一题刷到的。

动态规划类型的题目，重点就是找状态转移方程，因为我不太熟练，对动态规划的题目做的比较少，所以WA了好几次。

```python
class Solution:
    def minCostClimbingStairs(cost):
        res = [] #res[i]就是到第i阶梯时最小的花费
        res.append(cost[0])  #到第一阶梯最小就是0+cost[0]
        res.append(cost[1]) #第二阶梯最小就是0+cost[1]
        #状态转移方程:res[i] = min(res[i-1],res[i-2])+cost[i]
        for i in range(2,len(cost)): 
            res.append(min(res[i-1],res[i-2])+cost[i]) #
        return min(res[-1],res[-2])
```

> 踏上第i级台阶有两种方法：
>
> 先踏上第i-2级台阶（最小总花费`dp[i-2]`），再直接迈两步踏上第i级台阶（花费`cost[i]`），最小总花费`dp[i-2] + cost[i]`；
>
> 先踏上第i-1级台阶（最小总花费`dp[i-1]`），再迈一步踏上第i级台阶（花费`cost[i]`），最小总花费`dp[i-1] + cost[i]`；
>

上述为引用的题解的说明，更加深了对动态规划的理解


