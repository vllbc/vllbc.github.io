# 30


# 	**长度最小的子数组**

## **题目：**

**[https://leetcode-cn.com/problems/minimum-size-subarray-sum/](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)**

## **思路：**

​	**一开始想的是直接排序，然后从后面开始遍历，因为要求最小的**

**然后出错了，，，，，**

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        nums.sort()
        end = len(nums) - 1
        while end > 0:
            for i in range(end,-1,-1):
                if sum(nums[i:end+1]) >= s:
                    return len(nums[i:end+1])
            end -= 1
        return 0

```

**在**

**`213`**

**`[12,28,83,4,25,26,25,2,25,25,25,12]`**

**出了错，结果试了一下排序后的列表**

**`[2, 4, 12, 12, 25, 25, 25, 25, 25, 26, 28, 83]`**

**结果居然是对的，说明是我的代码的问题，不应该排序**

**那么该怎么办呢**

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        left = 0
        right = 0
        res = float('inf')
        while right < len(nums):
            while sum(nums[left:right+1]) >= s:
                res = min(res, right-left +1)
                left += 1
            else:
                right += 1
        if res == float('inf'):
            return 0
        return res
```

**思路也差不多，也是用到了双指针。不过必须注意要判断最小的这个条件啊。**


