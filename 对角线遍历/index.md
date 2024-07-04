# 对角线遍历



# 对角线遍历

## 题目：

​	 	[https://leetcode-cn.com/problems/diagonal-traverse/](https://leetcode-cn.com/problems/diagonal-traverse/)

## 思路：

​		每个对角线的两索引之和是一样的

## 代码：

  	

```python
class Solution:
    def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: 
            return []
        hashs = collections.defaultdict(list)
        row, col = len(matrix), len(matrix[0])

        for i in range(row):
            for j in range(col):
                hashs[j + i].append(matrix[i][j])
        res = []
        flag = True
        for k, v in sorted(hashs.items()):
            if flag:
                res.extend(v[::-1])
            else:
                res.extend(v)
            flag = not flag
        return res
```

注意flag的作用
