# 和为K的子数组

[题目地址](https://leetcode.cn/problems/subarray-sum-equals-k/)
# 思路
通过前缀和+哈希表，并有简单的数学变换。前缀和即 $y[i]=y[i-1]+x[i]$
类比于accumlate函数，注意前缀和思想也可以应用为“前缀积、后缀和、后缀积”等思想。[238. 除自身以外数组的乘积 - 力扣（LeetCode）](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-100-liked)
> 使用前缀和的方法可以解决这个问题，因为我们需要找到和为k的连续子数组的个数。通过计算前缀和，我们可以将问题转化为求解两个前缀和之差等于k的情况。
>假设数组的前缀和数组为prefixSum，其中prefixSum[i]表示从数组起始位置到第i个位置的元素之和。那么对于任意的两个下标i和j（i < j），如果prefixSum[j] - prefixSum[i] = k，即从第i个位置到第j个位置的元素之和等于k，那么说明从第i+1个位置到第j个位置的连续子数组的和为k。
通过遍历数组，计算每个位置的前缀和，并使用一个哈希表来存储每个前缀和出现的次数。在遍历的过程中，我们检查是否存在prefixSum[j] - k的前缀和，如果存在，说明从某个位置到当前位置的连续子数组的和为k，我们将对应的次数累加到结果中。
这样，通过遍历一次数组，我们可以统计出和为k的连续子数组的个数，并且时间复杂度为O(n)，其中n为数组的长度。
# 代码
```python
class Solution:

    def subarraySum(self, nums: List[int], k: int) -> int:

        from collections import defaultdict

        d = defaultdict(int)

        d[0] = 1

        prefix = 0

        res = 0

        for num in nums:

            prefix += num

            temp = prefix - k

            if temp in d:

                res += d[temp]

            d[prefix] += 1

        return res
```
