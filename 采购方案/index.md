# 采购方案



# 采购方案

## 题目：

[https://leetcode-cn.com/problems/4xy4Wx/](https://leetcode-cn.com/problems/4xy4Wx/)

## 思路：

题目很简单，思想就是双指针，感觉是个双指针的典型例子就写了下来

先对数组进行从小到大排序，然后双指针从两边移动，如果一直大于target就一直左移right 然后right - left就是所有成立的数目，再移动left 进行筛选

## 代码：

```python
class Solution:
    def purchasePlans(self, nums: List[int], target: int) -> int:
        nums.sort()
        left = 0
        right = len(nums) - 1
        res = 0
        while left < right and left < len(nums):
            while left < right and nums[right] + nums[left] > target:
                right -= 1
            res += right - left
            left += 1
        return res % (1000000007)
```
