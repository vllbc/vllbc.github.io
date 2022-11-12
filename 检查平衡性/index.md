# 检查平衡性


# 检查平衡性

## 题目：

[https://leetcode-cn.com/problems/check-balance-lcci/](https://leetcode-cn.com/problems/check-balance-lcci/)

## 思路：

算深度，然后作差是否大于1

## 代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if self.maxdepth(root) < 1:
            return True
        if abs(self.maxdepth(root.left) - self.maxdepth(root.right)) > 1:
            return False
        return self.isBalanced(root.right) and self.isBalanced(root.left)


    def maxdepth(self,root):
        if not root:
            return 0
        return 1 + max(self.maxdepth(root.right),self.maxdepth(root.left))  
```




