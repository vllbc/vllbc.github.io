# 37


# 两两交换链表中的节点

## 题目：

[https://leetcode-cn.com/problems/swap-nodes-in-pairs/](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

## 思路:

先把第二位储存起来，然后将后面的递归操作后，再把第二位指向第一位，完成换位

## 代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution: #假设为[1,2,3,4]
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next: #递归出口
            return head
        newnode = head.next #储存第二位2
        head.next = self.swapPairs(head.next.next) #此时为[1,4,3]
        newnode.next = head #[2,1,4,3]
        return newnode
```




