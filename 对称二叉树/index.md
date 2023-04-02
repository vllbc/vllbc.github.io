# 对称二叉树



# 对称二叉树

## 题目：

[https://leetcode-cn.com/problems/symmetric-tree/](https://leetcode-cn.com/problems/symmetric-tree/)

## 思路：

利用双向队列，每次把对称的两个对应的节点放入队列中，然后取出来比较，如果值不相等则返回false,如果一边为空 一边不为空也返回false 符合条件的话就继续搜索

## 代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        from collections import deque
        d = deque()
        d.append((root,root))
        while d:
            left,right = d.popleft()
            if not left and not right:
                continue
            elif not left or not right:
                return False
            elif left.val != right.val:
                return False
            else:
                d.append((left.left,right.right))
                d.append((left.right,right.left))
        return True
```
