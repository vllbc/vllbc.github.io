# 计数质数



# 计数质数

[https://leetcode-cn.com/problems/count-primes/](https://leetcode-cn.com/problems/count-primes/)

一开始直接暴力，隐约感觉会超时，果然不出我所料

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        counts = 0
        for i in range(n):
            if self.isprime(i):
                counts += 1
        return counts
                
    def isprime(self,n):
        from itertools import count
        if n <=1:
            return False
        for i in count(2):
            if i* i > n:
                return True
            if n % i == 0:
                return False
```

我还特意用了itertools库，没想到还是超时了

## 埃氏筛

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        res = [1] * n
        count = 0
        for i in range(2, n):
            if res[i]:
                count += 1
            for j in range(i*i, n, i):
                res[j] = 0
        return count
```

埃氏筛的原理：从 2 开始，将每个质数的倍数都标记为合数。同样的，标记到 
根号n停止。

假设一个数 i 为质数时，那么此时大于 i 且是 i 的倍数的数一定不是质数，例如 2i，3i...。那么我们将这些不是质数的数进行标记。

这里需要注意，标记应该从 i * i 开始，而不是 2 * i 开始。因为对于每个数 i 来说，枚举是从小到大的，此时前面数字的倍数都已经进行了标记。对于 i 而言，2∗i 也肯定会被在枚举数字 2 时进行标记，[2, i) 区间的数同理。

## 欧拉筛（线性筛）

具体的证明不说了，背板子就行
```python
class Solution:

    def countPrimes(self, n: int) -> int:

        if n <= 1:

            return 0

        is_prime = [True] * n

        is_prime[0] = is_prime[1] = False

        res = []

        for i in range(2, n):

            if is_prime[i]:

                res.append(i)

            j = 0

            while j < len(res) and (tmp:=i*res[j]) < n:

                is_prime[tmp] = False

                if i % res[j] == 0: # 如果res[j]为当前数的约数，则i*res[j+1]等后面的数必然会在后面的计算得到。就结束循环。这减少了重复计算。

                    break

                j += 1

        return len(res)
```
