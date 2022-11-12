# 最小公众前缀


# 最小公众前缀

leetcode上的简单题，[最小公众前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

有三种解法，一种常规，两种巧妙解法



```python
# 最小公共前缀

#解1：常规解法 思路就是一个一个判断 先判断所有字符串第一个是否相同，不相同就返回，否则然后依次往后判断
def longestCommonPrefix1(strs):
        if len(strs) == 0:
            return ''
        if len(strs) == 1:
            return strs[0]
        minl=min([len(x) for x in strs])  #求最小长度
        end = 0
        while end < minl:   #判断是否到最小长度
            for i in range(1,len(strs)):  #以第一个字符串为基准
                if strs[i][end] != strs[i-1][end]:  #如果到end这里不再相等 则返回到end这里的字符串即最小公共前缀
                    return strs[0][:end]
            end+=1
        return strs[0][:end]
#常规方法容易想到 但是缺点是运行速度慢，从每次判断都要遍历所有字符串就可以看出

#解2: 通过ascii码来判断
#Python里字符串是可以比较的，按照ascII值排
def longestCommonPrefix2(strs):
        if not strs:
            return 0
        s1 = max(strs) 
        s2 = min(strs)
        #找出s1 s2的最小公共前缀即为整个列表的最小公共前缀
        for i,s in enumerate(s2):
            if s1[i] != s:
                return s1[:i]
        return s2
#通过max 和 min 函数来找到列表里面最大最小的两个字符串 然后找到这两个字符串的最小公共前缀。


#解3：通过python语法糖 将每个字符串的每个对应字符串存为一组，用zip函数，比如说所有的字符串第一个存在一起，然后用set去重，如果留下了一个，则说明都重复了，则就是相同的
def longestCommonPrefix3(strs):
        if not strs:
            return 0
        cc = list(map(set,zip(*strs)))  #为什么用map呢 因为要对zip压缩后的每一个序列去重
        res = ''  #结果
        for i,s in enumerate(cc):
            x = list(s)
            if len(x) > 1: #如果长度大于1 说明有不一样的 则直接退出
                break
            res += x[0]
        return res

```

如上！


