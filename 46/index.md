# 46


# 和为s的连续正数序列

## 题目:

[https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

## 思路：

滑动窗口即可，滑动窗口就是选取数组的一部分来进行操作，left 和 right只能向右移动

## 代码：

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res = []
        left,right = 1,2
        while left <= target // 2: # 优化 减少时间复杂度
            if sum(range(left,right+1)) < target: # 小于target 右指针移动
                right += 1
            elif sum(range(left,right+1)) > target: # 大于target 左指针移动
                left += 1
            else:
                res.append(list(range(left,right+1))) # 相等的话 两个指针都移动
                right += 1
                left += 1
        return res
```




