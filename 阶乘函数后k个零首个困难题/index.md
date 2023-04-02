# 阶乘函数后K个零(首个困难题)



# 阶乘函数后K个零(首个困难题)

## 题目：

[https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function/](https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function/)

## 思路：

首先先写个判断阶乘后有多少个零的函数，思路就是找所有相乘的数中因数有5的个数。

然后再用二分查找，找到有K个0的左界和右界，然后相减即可，就是要找的数目

```python
class Solution:
    def preimageSizeFZF(self, K: int) -> int:
       return self.findright(K) - self.findleft(K)
    def whatzero(self,n):
        dis = 5
        res = 0
        while dis <= n:
            res += n // dis 
            dis *= 5
        return res
    def findleft(self,K):
        mins,maxs = 0,sys.maxsize
        while (mins < maxs):
            mid = mins + (maxs-mins) // 2
            if self.whatzero(mid) < K:
                mins = mid + 1
            elif self.whatzero(mid) > K:
                maxs = mid
            else:
                maxs = mid
        return mins
    def findright(self,K):
        mins,maxs = 0,sys.maxsize
        while (mins < maxs):
            mid = mins + (maxs-mins) // 2
            if self.whatzero(mid) < K:
                mins = mid + 1
            elif self.whatzero(mid) > K:
                maxs = mid
            else:
                mins = mid + 1
        return maxs
```

注意这里的最大值要初始化为sys库里的maxsize 用float("inf")会返回nan值
