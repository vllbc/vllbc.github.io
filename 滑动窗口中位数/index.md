# 滑动窗口中位数



# 滑动窗口中位数

## 题目：

[https://leetcode-cn.com/problems/sliding-window-median/](https://leetcode-cn.com/problems/sliding-window-median/)

## 思路：

很明显的滑动窗口，首先定义一个求中位数的匿名函数，然后一点一点求出来

## 代码：

```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        median = lambda a: (a[(len(a)-1)//2] + a[len(a)//2]) / 2
        res = []
        for i in range(len(nums)-k+1):
            res.append(median(sorted(nums[i:i+k])))
        return res
```
