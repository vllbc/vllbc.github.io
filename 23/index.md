# 23


# **交替打印字符串**

**[https://leetcode-cn.com/problems/fizz-buzz-multithreaded/](https://leetcode-cn.com/problems/fizz-buzz-multithreaded/)**

**一个多线程的题目，类似于交替打印数字那个题目**

```python
class FizzBuzz:
    def __init__(self, n: int):
        self.n = n
        from threading import Lock
        self.fizzlock = Lock()
        self.buzzlock = Lock()
        self.fzlock = Lock()
        self.nofzlock = Lock()

        self.fizzlock.acquire()
        self.buzzlock.acquire()
        self.fzlock.acquire()
    # printFizz() outputs "fizz"
    def fizz(self, printFizz: 'Callable[[], None]') -> None:
    	for i in range(1,self.n+1):
            if i % 3 == 0 and i % 5 != 0:
                self.fizzlock.acquire()
                printFizz()
                self.nofzlock.release()
    # printBuzz() outputs "buzz"
    def buzz(self, printBuzz: 'Callable[[], None]') -> None:
    	for i in range(1,self.n+1):
            if i % 5 == 0 and i % 3 != 0:
                self.buzzlock.acquire()
                printBuzz()
                self.nofzlock.release()
    # printFizzBuzz() outputs "fizzbuzz"
    def fizzbuzz(self, printFizzBuzz: 'Callable[[], None]') -> None:
        for i in range(1,self.n+1):
            if i % 15 == 0:
                self.fzlock.acquire()
                printFizzBuzz()
                self.nofzlock.release()
                
    # printNumber(x) outputs "x", where x is an integer.
    def number(self, printNumber: 'Callable[[int], None]') -> None:
        for i in range(1,self.n+1):
            self.nofzlock.acquire()
            if i % 3 != 0 and i % 5 != 0:
                printNumber(i)
                self.nofzlock.release()
            elif i % 3 != 0 and i % 5 == 0:
                self.buzzlock.release()
            elif i % 3 == 0 and i % 5 != 0:
                self.fizzlock.release()
            elif i % 15 == 0:
                self.fzlock.release()
```

**思路一摸一样就是利用了锁的阻塞机制**


