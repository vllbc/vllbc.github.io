# 17


# 按序打印

[https://leetcode-cn.com/problems/print-in-order/](https://leetcode-cn.com/problems/print-in-order/)



今天来点不一样的，来个多线程的题目

题目很简单啊，主要思路就是用Queue的特性

当队列为空时，get()是阻塞的

```python
    class Foo:
        def __init__(self):
            from queue import Queue
            self.q1 = Queue()
            self.q2 = Queue()
        def first(self, printFirst: 'Callable[[], None]') -> None:
            # printFirst() outputs "first". Do not change or remove this line.
            printFirst()
            self.q1.put(0)
        def second(self, printSecond: 'Callable[[], None]') -> None:
            self.q1.get()
            # printSecond() outputs "second". Do not change or remove this line.
            printSecond()
            self.q2.put(0)
        def third(self, printThird: 'Callable[[], None]') -> None:
            self.q2.get()
            # printThird() outputs "third". Do not change or remove this line.
            printThird()
```




