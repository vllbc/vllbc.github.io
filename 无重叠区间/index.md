# 无重叠区间


# 无重叠区间

[https://leetcode-cn.com/problems/non-overlapping-intervals/](https://leetcode-cn.com/problems/non-overlapping-intervals/)

利用了贪心 移除的数目就是总数目减去条件成立的数目

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 0:
            return 0
        res = 0
        mins = -float("inf")
        for i in sorted(intervals,key=lambda i:i[1]):
            if i[0] >= mins:
                res += 1
                mins = i[1]
        return len(intervals) - res
```

注意是根据end进行排序的，引用别人的解释@[HONGYANG](https://leetcode-cn.com/u/hongyang57/)

> 比如你一天要参加几个活动，这个活动开始的多早其实不重要，重要的是你结束的多早，早晨7点就开始了然后一搞搞一天，那你今天也就只能参加这一个活动；但如果这个活动开始的不早，比如9点才开始，但是随便搞搞10点就结束了，那你接下来就还有大半天的时间可以参加其他活动。
>
> 这就是为啥要着眼于end，而不是start。
>

贪心就是考虑当前最优解


