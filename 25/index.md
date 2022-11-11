# 25


# **种花问题**（新年快乐!2021第一题）

**新年快乐！2021年第一题，每日一题！希望2021年LC和github可以全绿！加油！**

**[https://leetcode-cn.com/problems/can-place-flowers/](https://leetcode-cn.com/problems/can-place-flowers/)**

**代码如下：**

**** 

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        flowerbed = [0] + flowerbed + [0]
        for i in range(1,len(flowerbed)-1):
            if flowerbed[i-1] == 0 and flowerbed[i] == 0 and flowerbed[i+1] == 0:
                n -= 1
                flowerbed[i] = 1
        return n <= 0
```

**思路很暴力，就是三个0在一起就可以插进去。。**

**主要是边界问题，这里构造了两个边界**

**新的一年开开心心，完成自己的目标，让自己更优秀！**




