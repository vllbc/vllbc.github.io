# 可获得的最大点数



# 可获得的最大点数

## 题目：

[https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

## 思路：

滑动窗口题目，限定窗口大小然后滑动即可

## 代码：

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        # 滑动窗口大小为 n-k
        windowSize = n - k
        # 选前 n-k 个作为初始值
        s = sum(cardPoints[:windowSize])
        minSum = s
        for i in range(windowSize, n):
            # 滑动窗口每向右移动一格，增加从右侧进入窗口的元素值，并减少从左侧离开窗口的元素值
            s += cardPoints[i] - cardPoints[i - windowSize]
            minSum = min(minSum, s)
        return sum(cardPoints) - minSum
```
