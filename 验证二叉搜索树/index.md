# 验证二叉搜索树



# 验证二叉搜索树

[https://leetcode-cn.com/problems/validate-binary-search-tree/](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.search(root,-(232),232)
    def search(self,root,mins,maxs):
        if root == None:
            return True
        if root.val > mins and root.val < maxs:
            pass
        else:
            return False
     	return all([self.search(root.left,mins,root.val),self.search(root.right,root.val,maxs)])
```

最后用了个all 也是简洁了代码
