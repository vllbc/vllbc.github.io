# asyncio



```python
import threading
import time
```

## 多线程例子


```python
def spider():
    #.....
    time.sleep(0.02)
def main1():
    for i in range(100):
        spider()
def main2():
    thread_list = []
    for i in range(100):
        thread = threading.Thread(target = spider)
        thread.start()
        thread_list.append(thread)
    for t in thread_list:
        t.join()
if __name__ == "__main__":
    start = time.time()
    main1()
    end = time.time()
    print("time1 :{:.4f}".format(end-start))
    
    start = time.time()
    main2()
    end = time.time()
    print("time2 :{:4f}".format(end-start))
```

    time1 :2.0523
    time2 :0.037929


## yield


```python
def fib(n):
    a,b = 0,1
    while b<n: 
        a,b = b,a+b
        yield a
print(fib(100))
for i in fib(100):
    print(i)
```

    <generator object fib at 0x000002B1A7AA1E60>
    1
    1
    2
    3
    5
    8
    13
    21
    34
    55
    89


## 协程

**GEN_CREATED 创建完成，等待执行**
**GEN_RUNNING 解释器正在执行**
**GEN_SUSPENDED 在 yield 表达式处暂停**
**GEN_CLOSE 执行结束，生成器停止**


```python
import inspect
def generator():
    i = "激活生成器"
    while True:
        try:
            value = yield i
        except ValueError:
            print("OVER")
        i = value

```


```python
g = generator()
print(inspect.getgeneratorstate(g)) #查看状态
next(g) # next(g)相当于g.send(None) 可以用后面的语句来预缴携程
```

    GEN_CREATED





    '激活生成器'




```python
inspect.getgeneratorstate(g) #查看生成器状态
```




    'GEN_SUSPENDED'




```python
g.send("hello world")
```




    'hello world'



**暂停状态的生成器可以使用 send 方法发送数据，此方法的参数就是 yield 表达式的值，也就是 yield 表达式等号前面的 value 变量的值变成 'Hello Shiyanlou'，继续向下执行完一次 while 循环，变量 i 被赋值，继续运行下一次循环，yield 表达式弹出变量 i**


```python
g.throw(ValueError) #抛出异常 结束
```

    OVER





    'hello world'




```python
g.close()
inspect.getgeneratorstate(g) #关闭了
```




    'GEN_CLOSED'



## 预激协程


```python
from functools import wraps
def corcut(func):
    @wraps(func)
    def wrapper(*args,**kw):
        g = func(*args,**kw)
        next(g)
        return g
    return wrapper

@corcut #装饰器 
def generator():
    i = "激活生成器"
    while True:
        try:
            value = yield i
        except ValueError:
            print("OVER")
        i = value
g = generator()
print(inspect.getgeneratorstate(g)) #此时已经用装饰器将生成器激活了

```

    GEN_SUSPENDED



```python
@corcut
def generator():
    l = []
    while True:
        value = yield
        if value == "CLOSE":
            break
        l.append(value)
    return l
g = generator()
for i in ['a','b','CLOSE']:
    try:
        g.send(i)
    except StopIteration as e:
        value = e.value
    
value
        
```




    ['a', 'b']



## yield from用法


```python
from itertools import chain
c = chain({'one','two','three'},list("abc"))
for i in c:
    print(i)
```

    three
    two
    one
    a
    b
    c



```python
def chains1(*args):
    for i in args:
        for n in i:
            yield n
def chains2(*args):
    for i in args:
        yield from i #i为可迭代对象，避免嵌套循环
c1 = chains1({"one","two","three"},list("abc"))
for i in c1:
    print(i)
print("\n")
c2 = chains2({"one","two","three"},list("abc"))
for i in c2:
    print(i)
```

    three
    two
    one
    a
    b
    c


****

    three
    two
    one
    a
    b
    c


## 转移控制权


```python
from functools import wraps
from faker import Faker
import time
```


```python
def corout(func):
    @wraps(func)
    def wapper(*args,**kw):
        g = func(*args,**kw)
        next(g)
        return g
    return wapper

# 子生成器
def generator():
    l = []
    while True:
        i = yield
        if i == "CLOSE":
            break
        l.append(i)
    return sorted(l)
# 委托生成器
@corout
def generator2():
    while True:
        
        l = yield from generator()
        print("排序后的列表",l)
        print("-----------------")

# 客户端
if __name__ == "__main__":
    fake = Faker().country_code
    nest_country = [[fake() for i in range(3)] for j in range(3)]
    for country in nest_country:
        print('国家代号列表：', country)
        c = generator2()
        for i in country:
            c.send(i)
        c.send("CLOSE")
    
```

    国家代号列表： ['AM', 'ZA', 'BG']
    排序后的列表 ['AM', 'BG', 'ZA']
    -----------------
    国家代号列表： ['UG', 'BE', 'SI']
    排序后的列表 ['BE', 'SI', 'UG']
    -----------------
    国家代号列表： ['SC', 'KI', 'KI']
    排序后的列表 ['KI', 'KI', 'SC']
    -----------------

yield显然不只是用来减小循环次数的，引用一下《流畅的python》中关于yield from 的意义：
- 子生成器产出的值都直接传给委派生成器的调用方（即客户端代码）。
- 使用 send() 方法发给委派生成器的值都直接传给子生成器。如果发送的值是 None，那么会调用子生成器的 __next__() 方法。如果发送的值不是 None，那么会调用子生成器的 send() 方法。如果调用的方法抛出 StopIteration 异常，那么委派生成器恢复运行。任何其他异常都会向上冒泡，传给委派生成器。
- 生成器退出时，生成器（或子生成器）中的 return expr 表达式会触发 StopIteration(expr) 异常抛出。
- yield from 表达式的值是子生成器终止时传给 StopIteration异常的第一个参数。
为什么yield可以转移控制权，可以看一下这一段伪代码：
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020230125201438.png)
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020230125202328.png)
![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/Pasted%20image%2020230125202421.png)
注意这里的6是委托生成器向子生成器发送_s，而_s是调用方向委托生成器发送的，发送后得到结果_y，并在下一个循环yield即抛出给调用方。
## asyncio模块


```python
import time
import asyncio

def one():
    start = time.time()
    @asyncio.coroutine #1
    def do_something(): #2
        print("start ------")
        time.sleep(0.1) #3
        print("doing something")
    loop = asyncio.get_event_loop() #4
    coroutine = do_something() #5
    loop.run_until_complete(coroutine) #6
    end = time.time()
    print("消耗时间:{:.4f}".format(end-start))#7
    
one()
```

    start ------
    doing something
    消耗时间:0.1012


**代码说明：**

**1、使用协程装饰器创建协程函数**

**2、协程函数**

**3、模拟 IO 操作**

**4、创建事件循环。每个线程中只能有一个事件循环，get_event_loop 方法会获取当前已经存在的事件循环，如果当前线程中没有，新建一个**

**5、调用协程函数获取协程对象**

**6、将协程对象注入到事件循环，协程的运行由事件循环控制。事件循环的 run_until_complete 方法会阻塞运行，直到任务全部完成。协程对象作为 run_until_complete 方法的参数，loop 会自动将协程对象包装成任务来运行。后面我们会讲到多个任务注入事件循环的情况**

**7、打印程序运行耗时**


```python
import time
import asyncio

def two():
    start = time.time()
    @asyncio.coroutine
    def do_something():
        print("start ------")
        time.sleep(0.1)
        print("doing something")
    loop = asyncio.get_event_loop()
    coroutine = do_something()
    task = loop.create_task(coroutine) #1
    print("task是不是Task的示例？",isinstance(task,asyncio.Task)) #2
    print("task state",task._state) #3
    loop.run_until_complete(task) #4
    print("take state",task._state)
    end = time.time()
    print("消耗时间:{:.4f}".format(end-start))
    
two()
```

    task是不是Task的示例？ True
    task state PENDING
    start ------
    doing something
    take state FINISHED
    消耗时间:0.1013


**1、事件循环的 create_task 方法可以创建任务，另外 asyncio.ensure_future 方法也可以创建任务，参数须为协程对象**

**2、task 是 asyncio.Task 类的实例，为什么要使用协程对象创建任务？因为在这个过程中 asyncio.Task 做了一些工作，包括预激协程、协程运行中遇到某些异常时的处理**

**3、task 对象的 _state 属性保存当前任务的运行状态，任务的运行状态有 PENDING 和 FINISHED 两种**

**4、将任务注入事件循环，阻塞运行**

## async / await


```python
import functools
def three():
    start = time.time()
    #@asyncio.coroutine
    async def do_something():  #1
        print("start doing")
        time.sleep(0.1)
        print("done")
    def callback(name,task): #2
        print("call back:{}".format(name))
        print("call back:{}".format(task._state))
    loop = asyncio.get_event_loop()
    coroutine = do_something()
    task = loop.create_task(coroutine)
    task.add_done_callback(functools.partial(callback, 'vllbc')) #3
    loop.run_until_complete(task)
    end = time.time()
    print("total time {:.4f}".format(end-start))
three()

    
```

    start doing
    done
    call back:vllbc
    call back:FINISHED
    total time 0.1013


**代码说明：**

**1、使用 async 关键字替代 asyncio.coroutine 装饰器创建协程函数**

**2、回调函数，协程终止后需要顺便运行的代码写入这里，回调函数的参数有要求，最后一个位置参数须为 task 对象**

**3、task 对象的 add_done_callback 方法可以添加回调函数，注意参数必须是回调函数，这个方法不能传入回调函数的参数，这一点需要通过 functools 模块的 partial 方法解决，将回调函数和其参数 name 作为 partial 方法的参数，此方法的返回值就是偏函数，偏函数可作为 task.add_done_callback 方法的参数**


```python
def four():
    start = time.time()
    async def do_something(name,t):
        print("start !>>",name)
        await asyncio.sleep(t) #1
        print('Stop coroutine', name)
        return 'Coroutine {} OK'.format(name) #2
    loop = asyncio.get_event_loop()
    coroutine1 = do_something('wlb',3) #3
    coroutine2 = do_something('yyh',1)
    task1 = loop.create_task(coroutine1) #4
    task2 = loop.create_task(coroutine2)
    gather = asyncio.gather(task1,task2) #5
    loop.run_until_complete(gather)
    print("task1",task1.result())
    print("task2",task2.result())
    #result = loop.run_until_complete(gather)
    #这里result就是两个返回值组成的列表 即['task1 Coroutine wlb OK','task2 Coroutine yyh OK']
    end = time.time()
    print("total time:{:.4f}".format(end-start))
four()
        
```

    start !>> wlb
    start !>> yyh
    Stop coroutine yyh
    Stop coroutine wlb
    task1 Coroutine wlb OK
    task2 Coroutine yyh OK
    total time:3.0022


**代码说明：**

**1、await 关键字等同于 Python 3.4 中的 yield from 语句，后面接协程对象。asyncio.sleep 方法的返回值为协程对象，这一步为阻塞运行。asyncio.sleep 与 time.sleep 是不同的，前者阻塞当前协程，即 corowork 函数的运行，而 time.sleep 会阻塞整个线程，所以这里必须用前者，阻塞当前协程，CPU 可以在线程内的其它协程中执行**

**2、协程函数的 return 值可以在协程运行结束后保存到对应的 task 对象的 result 方法中**

**3、创建两个协程对象，在协程内部分别阻塞 3 秒和 1 秒**

**4、创建两个任务对象**

**5、将任务对象作为参数，asyncio.gather 方法创建任务收集器。注意，asyncio.gather 方法中参数的顺序决定了协程的启动顺序**

**6、将任务收集器作为参数传入事件循环的 run_until_complete 方法，阻塞运行，直到全部任务完成**

**7、任务结束后，事件循环停止，打印任务的 result 方法返回值，即协程函数的 return 值**

**到这一步，大家应该可以看得出，上面的代码已经是异步编程的结构了，在事件循环内部，两个协程是交替运行完成的。简单叙述一下程序协程部分的运行过程：**

**-> 首先运行 task1**

**-> 打印 [corowork] Start coroutine ONE**

**-> 遇到 asyncio.sleep 阻塞**

**-> 释放 CPU 转到 task2 中执行**

**-> 打印 [corowork] Start coroutine TWO**

**-> 再次遇到 asyncio.sleep 阻塞**

**-> 这次没有其它协程可以运行了，只能等阻塞结束**

**-> task2 的阻塞时间较短，阻塞 1 秒后先结束，打印 [corowork] Stop coroutine TWO**

**-> 又过了 2 秒，阻塞 3 秒的 task1 也结束了阻塞，打印 [corowork] Stop coroutine ONE**

**-> 至此两个任务全部完成，事件循环停止**

**-> 打印两个任务的 result**

**-> 打印程序运行时间**

**-> 程序全部结束**

### await的理解
从上文中也可以看到await其实是从yield from中转变过来的，当在代码中看到await时，可以知道当前协程要去运行await后面的任务，此时控制权回到了event loop手中，去执行其它的任务，当前面的任务完成了以后，则转去执行前面await后面的代码。注意当await直接跟一个coroutline时，此时相当于去yield from，会卡在那里，并不会实现真正的异步，所以要先将coroutline变为task或者future就可以直接await。
## 异步编程

### 一个买土豆的例子
```python
import asyncio

import random

  
  

# potato类

class Potato:

    # 生成土豆

    @classmethod

    def make(cls, num, *args, **kws):

        potatos = []

        for i in range(num):

            potatos.append(cls.__new__(cls, *args, **kws))

        return potatos

all_potatos = Potato.make(5)

  

## 这是一个异步生成器，可以用async for迭代，nums为想买的数量。

async def take_photos(nums):

    count = 0

    while True:

        # 如果没有土豆了，挂起当前任务请求生成土豆任务。

        if len(all_potatos) == 0:

            await askfor_photos()

        else:

            photo = all_potatos.pop()
			# 如果有土豆将土豆抛出去
            yield photo

            count += 1

            if count == nums :

                break

  
  

async def askfor_photos():

    await asyncio.sleep(2)

    all_potatos.append(Potato.make(5))

  

async def buy_photos():

    bucket = []

    async for p in take_photos(50):

        bucket.append(p)

        print(f"Go photo {id(p)}")

  

loop = asyncio.get_event_loop()

loop.run_until_complete(buy_photos())
```
### requests例子

```python
import asyncio

import requests

import time

# 相当于委托生成器
async def result(url):

    res = await request_url(url)

    print(url, res)

# 相当于子生成器
async def request_url(url):

    res = requests.get(url)

    print(url)

    await asyncio.sleep(2)

    print("execute_time:", time.time() - start)

    return res

  
  

url_list = ["https://www.csdn.net/",

            "https://vllbc.top/",

            "https://www.baidu.com/",

            ]

  
# 以下相当于调用方
start = time.time()

print(f"start_time:{start}\n")

  

task = [result(url) for url in url_list]

loop = asyncio.get_event_loop()

loop.run_until_complete(asyncio.wait(task))

  

endtime = time.time() - start

print("\nendtime:", time.time())

print("all_execute_time:", endtime)
```
