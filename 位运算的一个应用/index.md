# 位运算的一个应用


# 位运算的一个应用

翻了翻以前用python刷leetcode的记录，最后刷的一道题是这样的

[https://leetcode-cn.com/problems/single-number/](https://leetcode-cn.com/problems/single-number/)

叫只出现一次的数字，当时看题感觉非常简单啊！直接搞就行了

当时一开始我的做法是这样的

```python
class Solution:
    def singleNumber(self, nums):
        for i in set(nums):
            if nums.count(i) ==1:
                return i
```

信心满满的提交，结果发现TLE了。。

然后看超时案例的输入，没有一个数字重复，也就是说我的set跟没有一样，所以说肯定不能这么做。

然后又想到了哈希表

```python
class Solution:
    def singleNumber(self, nums):
        dic={}
        for num in nums:
            if num in dic.keys():
                dic[num]+=1
            else:
                dic[num]=1
        for i in dic.keys():
            if dic[i] ==1:
                return i
```

这样也算是AC了。本以为这个题就这么结束了，结果无意中看到了别的题解震惊了

代码数量比我短的多得多。

然后就认识到了位运算的魔力。。

先上代码：

```python
class Solution:
    def singleNumber(self, nums):
        a = 0
        for i in nums:
            a^=i
        return a
```

简单的一个异或运算就达到了目的

真是太神奇了！

找到相关资料

1. 交换律：a ^ b ^ c <=> a ^ c ^ b
2. 任何数于0异或为任何数 0 ^ n => n
3. 相同的数异或为0: n ^ n => 0

也就是说相同的数就异或为0了，达到了去重的目的。




