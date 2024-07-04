# 移除元素



# 移除元素

还是以前刷过的题 [https://leetcode-cn.com/problems/remove-element/](https://leetcode-cn.com/problems/remove-element/)

以前的思路早忘了 然后我重新做了一下，一开始就一行代码

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        return len(list(filter(lambda x:x!=val,nums)))

```

​		然后发现输出和正确输出不一样。于是看了了下面的提示，然后改了改

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        for i in range(nums.count(val)):
            nums.remove(val)
        return len(nums)
```

但这算不上叫算法，利用双指针做法如下：
```python
class Solution:

    def removeElement(self, nums: List[int], val: int) -> int:

        left = 0

        for i in range(len(nums)):

            if nums[i] != val:

                nums[left], nums[i] = nums[i], nums[left]

                left += 1

        return left
```
