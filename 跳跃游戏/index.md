# 跳跃游戏

# 跳跃游戏
> Problem: 

  

## 思路

  

> 讲述看到这一题的思路

  

## 解题方法

  

> 描述你的解题方法

  

## 复杂度

  

时间复杂度:

> 添加时间复杂度, 示例： $O(n)$

  

空间复杂度:

> 添加空间复杂度, 示例： $O(n)$

  
  
  

## Code

```Python3 []



```

# 跳跃游戏ii

# 划分字母区间

[763. 划分字母区间 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-labels/description/?envType=study-plan-v2&envId=top-100-liked)
本题预处理完毕后思路和跳跃游戏2类似，当然也可以使用合并区间的思路来，都是贪心算法。
## Code
```python
class Solution:

    def partitionLabels(self, s: str) -> List[int]:

        from collections import defaultdict

        d = defaultdict(list)

        for i, char in enumerate(s):

            d[char].append(i)

        # 也可以考虑合并区间做了，下面的解法类似跳跃游戏2

        res = []

        start = 0

        max_jump = 0

        for i, char in enumerate(s):

            max_jump = max(max_jump, d[char][-1])

            if i == max_jump:

                res.append(i - start + 1)

                start = i + 1

                max_jump = 0

        return res
```
