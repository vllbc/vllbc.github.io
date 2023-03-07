# 打印零和奇偶数



# 打印零和奇偶数

多线程问题，利用线程锁可以轻松解决

[https://leetcode-cn.com/problems/print-zero-even-odd/](https://leetcode-cn.com/problems/print-zero-even-odd/)

```python
class ZeroEvenOdd:
    def __init__(self, n):
        from threading import Lock
        self.n = n
        self.zerolock = Lock()
        self.oddlock = Lock()
        self.evenlock = Lock()
        self.oddlock.acquire()
        self.evenlock.acquire()
	# printNumber(x) outputs "x", where x is an integer.
    def zero(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1,self.n+1):
            self.zerolock.acquire()
            printNumber(0)
            if i % 2 == 0:
                self.evenlock.release()
            else:
                self.oddlock.release()
    def even(self, printNumber: 'Callable[[int], None]') -> None: #偶
        for i in range(1,self.n+1):
            if i % 2 == 0:
                self.evenlock.acquire()
                printNumber(i)
                self.zerolock.release()

    def odd(self, printNumber: 'Callable[[int], None]') -> None: #奇
        for i in range(1,self.n + 1):
            if i % 2 != 0:
                self.oddlock.acquire()
                printNumber(i)
                self.zerolock.release()
```
