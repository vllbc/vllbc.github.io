# 搜索旋转排序数组



# 搜索旋转排序数组

## 题目：

[https://leetcode-cn.com/problems/search-in-rotated-sorted-array/](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

## 思路：

明显的二分查找，不过不是有序数组了，而是部分有序，所以需要有判断

## 代码：

```python
class Solution(object):
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[right]:#右边为升序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid 
            if nums[left] <= nums[mid]:#左边为升序
                if nums[left] <= target < nums[mid]:
                    right = mid 
                else:
                    left = mid + 1
        return -1
```
