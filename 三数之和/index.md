# 三数之和

# 三数之和

## 题目：

[https://leetcode-cn.com/problems/3sum/solution/](https://leetcode-cn.com/problems/3sum/solution/)

## 思路：

​	第一眼看就想到了用双指针，注意重复数值的处理问题，算是一个滑动窗口问题

## 代码：

```python
class Solution:

    def threeSum(self, nums: List[int]) -> List[List[int]]:

        res = []

        if len(nums) < 3:

            return []

        nums.sort()

        for i, num in enumerate(nums):

            if num > 0:

                return res

            if i > 0 and nums[i] == nums[i-1]:

                continue

            left, right = i+1, len(nums) - 1

            while left < right:

                temp = nums[i] + nums[left] + nums[right]

                if temp == 0:

                    res.append([nums[i], nums[left], nums[right]])

                    while left < right and nums[right-1] == nums[right]:

                        right -= 1

                    while left < right and nums[left+1] == nums[left]:

                        left += 1

                    left += 1

                    right -= 1

                if temp > 0:

                    right -=1

                if temp < 0:

                    left += 1

        return res
```
