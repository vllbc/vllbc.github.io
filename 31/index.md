# 31


# 删除排序数组中的重复项

## 删除排序数组中的重复项1

### 	题目：

​	[https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/](	https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

### 	思路：

​		`双指针，定义 nums[0...i] 为为非重复数列，遍历整个数列不断的维护这个定义`

### 	代码：

​		

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        start = 0
        for i in range(len(nums)):
            if nums[i] != nums[start]:
                start += 1
                nums[i],nums[start] = nums[start],nums[i]
        return start + 1
```

## 	删除排序数组中的重复项2

### 		题目：

​		[https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

### 		思路：

​				也是利用双指针，一个指针用于遍历数组元素，一个指针指向要拷贝赋值的索引位置

### 		代码：

​			

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2: #长度小于等于2时
            return len(nums)
        count = 1 #用于重复的计数 
        j = 1 #指向多余重复的元素
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                count += 1 #重复了就加一
                if count > 2: #如果重复两次以上就pass掉，等着被替换
                    pass
                else:
                    nums[j] = nums[i] 
                    j += 1
            else:
                nums[j] = nums[i] #如果不相等了 把多余重复的那个替换掉了
                count = 1 #重置计数
                j += 1
        return j
```

这是一种思路比较清晰的写法




