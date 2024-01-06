# 有效的数独



# 有效的数独

[https://leetcode-cn.com/problems/valid-sudoku/](https://leetcode-cn.com/problems/valid-sudoku/)

```python
#有效的数独 难点在将3*3里的数取出来


class Solution:
    def isValidSudoku(board) -> bool:
        for line1,line2 in zip(board,zip(*board)): #行列
            for n1,n2 in zip(line1,line2):
                if (n1 != '.' and line1.count(n1) > 1) or (n2!='.' and line2.count(n2) >1):
                    return False
        pal = [[board[i+m][j+n] for m in range(3) for n in range(3) if board[i+m][j+n] != '.'] for i in (0, 3, 6) for j in (0, 3, 6)]
        for line in pal:
            if len(set(line)) != len(line):
                return False
        return True
```
