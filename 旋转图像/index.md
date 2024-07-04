# 旋转图像



# 旋转图像

[https://leetcode-cn.com/problems/rotate-image/](https://leetcode-cn.com/problems/rotate-image/)

没难度的中等题，这方法很python

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in list(map(list,map(reversed,zip(*matrix)))):
            matrix.append(i)
        del matrix[:n]
```
