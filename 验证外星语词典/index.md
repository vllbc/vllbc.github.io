# 验证外星语词典

leetcode地址：[953. 验证外星语词典 - 力扣（LeetCode）](https://leetcode.cn/problems/verifying-an-alien-dictionary/description/)

## 简单方法
python列表之间也可以进行比较（太灵活了），比如`[1, 2, 3] < [2, 2, 3]`成立，即按照字典序进行比较，与其是一样的比较规则。因此对于本题可以利用python的特性轻松解决。
好久没写python了，变得很生疏，一开始写的很蠢：
```python
class Solution:

    def isAlienSorted(self, words: List[str], order: str) -> bool:
    
        d = dict(zip(order, range(len(order))))

        words = list(map(lambda s: [d[i] for i in s], words))

        print([1, 2, 3] < [2, 2, 3])

        return words == sorted(words)
```

后来想起来了sorted中还有个key参数，并且列表还有个index方法（我基本上没用过），于是改成了一行
```python
class Solution:

    def isAlienSorted(self, words: List[str], order: str) -> bool:

         return words == sorted(words, key=lambda w:[order.index(x) for x in w])
```


