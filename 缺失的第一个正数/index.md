# 缺失的第一个正数


[41. 缺失的第一个正数 - 力扣（LeetCode）](https://leetcode.cn/problems/first-missing-positive/?envType=study-plan-v2&envId=top-100-liked)
空间复杂度o(n)很好想，但o(1)不好想，还是个408考研真题

注意O(n) == O(2n)，即相较于边遍历边判断，还是遍历两次更加方便且不会有太多损失。类似思想：[73. 矩阵置零 - 力扣（LeetCode）](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```python
class Solution:

    def firstMissingPositive(self, nums: List[int]) -> int:

        # 将原数组当作哈希表使用。

        for i, num in enumerate(nums):

            if num <= 0:  # 先把小于0的先一步处理，以便于使用标记

                nums[i] = len(nums) + 1

        for i, num in enumerate(nums):

            num = abs(num)

            if num >= 1 and num <= len(nums):

                if nums[num-1] > 0: # 不重复添加

                    nums[num-1] = -nums[num-1]

        res = len(nums) + 1

        for i, num in enumerate(nums):

            if num > 0:

                res = i + 1

                break

        return res
```
