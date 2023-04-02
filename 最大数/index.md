# 最大数



# 最大数

## 题目：

[https://leetcode-cn.com/problems/largest-number/](https://leetcode-cn.com/problems/largest-number/)

## 思路：

一开始直接暴力搜索，把所有的情况都列举然后比较，结果超时了，最后利用了自定义排序的方法

## 代码：

```python
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        class Comapre(str):
            def __lt__(self,other):
                return int(self+other) > int(other+self)
        nums.sort(key=Comapre)
        return str(int(''.join(map(str,nums))))
```

注意的是这里利用了自定义的比较类型，继承了str，也可以从functools里导入cmp_to_key方法来实现比较

> python3之后不支持cmp，所用key函数并不直接比较任意两个原始元素，而是通过key函数把那些元素转换成一个个新的可比较对象，也就是元素的key，然后用元素的key代替元素去参与比较。如果原始元素本来就是可比较对象，比如数字、字符串，那么不考虑性能优化可以直接sort(key=lambda e: e)。不过这种基于key函数的设计倾向于每个元素的大小有个绝对标准，但有时却会出现单个元素并没有一个绝对的大小的情况，此时可以使用 functools.cmp_to_key构建基于多个元素的比较函数。
>
