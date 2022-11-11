# skill


## **两个数的交换**


```python
# a = 1
# b = 2
# temp = b
# b = a
# a = temp
# print(a,b)

a = 1
b = 2
a,b = b,a
print(a,b)
```

    2 1


## **格式化字符串**


```python
a = 17
name = "wlb"
# print('%s is %d years old' %  (name,a))
# print('{} is {} years old'.format(name,a))
print(f'{name} is {a} years old') #明显这个方法更简单
```

    wlb is 17 years old


## **yield与yield from**


```python
def fib(n):
    a = 0
    b = 1
    for _ in range(n):
        yield  a
        a,b = b,a+b
for i in fib(10):
    print(i)
#注释的内容与yield a效果相同，yield相当于使其成为一个迭代器 yield一个数后会立马传递出去，而return 要等列表都生成完毕后才会传出去
#他的优势在于一些耗时的操作
```
```python
# 通过yield来进行dfs，由于没有实现__next__因此是个可迭代对象而不是一个迭代器
class Node:
    def __init__(self,value) -> None:
        self._value = value
        self._node = []
    
    def __repr__(self) -> str:
        return f'Node({self._value})'
    
    def add_children(self,node:'Node') -> 'Node':
        self._node.append(node)
    
    def __iter__(self):
        return iter(self._node)

    def dfs(self):
        yield self
        for i in self:
            yield from i.dfs()
root = Node(0)
children1 = Node(1)
children2 = Node(2)
root.add_children(children1)
root.add_children(children2)
children1.add_children(Node(3))
children1.add_children(Node(4))
children11 = Node(5)
children2.add_children(children11)
children11.add_children(Node(6))
for c in root.dfs():
    print(c)
```



```python
from typing import Iterable
def test_format(datas: Iterable[str], max_len: int):
    for data in datas:
        if len(data) > max_len:
            yield data[:max_len] + '...'
        else:
            yield data
print(list(test_format(['vllbc', 'test_for_this_function', 'good'],5)))
# 把长度大于5的部分变成省略号
```
```python
#子生成器
def average_gen():
    total = 0
    count = 0
    average = 0
    while True:
        new_num = yield average
        if new_num is None:
            break
        count += 1
        total += new_num
        average = total/count
    return total,count,average
# 委托生成器
def proxy_gen():
    while True:
        total,count,average = yield from average_gen() # yield from后面是一个可迭代对象,此文后面的将多维数组转化为一维数组中flatten函数就用到了yield from，原理就是如果列表中一个元素是列表就yield from这个列表，否则就直接yield这个元素，也利用了递归的方法。如果子生成器退出while循环了，就执行return以获取返回值。
        print(total,count,average)

def main():
    t = proxy_gen()
    next(t)
    print(t.send(10))
    print(t.send(15))
    print(t.send(20))
    t.send(None)
main()
```


## **列表解析式**


```python
lists = [f"http://www.baidu.com/page{n}" for n in range(21)]
lists#此方法在爬虫构造urls中非常常用
# lists = [f"http://www.baidu.com/page{n}" for n in range(21) if n%2==0] page偶数
# alp = "abcdefghigklmnopqrstuvwxyz"
# ALP = [n.upper() for n in alp] 将小写转换为大写
```

## **enumerate**


```python
lists = ['apple','banana','cat','dog']
for index,name in enumerate(lists):
    print(index,name)

# 手动实现一下enumerate
from typing import Iterable
def enumerate_(Iterable:Iterable,start=0):
    yield from zip(range(start,start+len(Iterable)),Iterable)

for i,item in enumerate_([1,2,3,4,5,6],9):
    print(i,item)

```


## **字典的合并**


```python
dic1 = {'qq':1683070754,
        'phone':123456789
       }
dic2 = {
    'height':180,
    'handsome':True
}
dic3 = {**dic1,**dic2}
#合并两个字典 **叫做解包
#或者用dic1.update(dic2) 将dic2合并到dic1 相同键则dic2替代dic1
dic3
```




    {'handsome': True, 'height': 180, 'phone': 123456789, 'qq': 1683070754}



## **序列解包**


```python
name = "wang lingbo"
xing,ming = name.split(" ") #split返回一个序列，分别赋给xing 和ming
print(xing,ming)
#x,*y,z = [1,2,3,4,5]
#x:1 z:5 y:[2,3,4]
```

    wang lingbo


## **匿名函数lambda**


```python
lists = [1,2,3,4,5,6]
maps = map(lambda x:x*x,lists)
print(maps)
print(list(maps))
```

    <map object at 0x000001911C8E03C8>
    [1, 4, 9, 16, 25, 36]


## **装饰器**


```python
def logging(level):
    def wapper(func):
        def inner_wapper(*args, **wbargs):
            print(f'{level}  enter in {func.__name__}()')
            return func(*args, **wbargs) #不写return 也可以
        return inner_wapper
    return wapper
@logging('inner')
def say(a):
    print('hello!  {}'.format(a))
say('wlb')
```

    inner  enter in say()
    hello!  wlb



```python
import time
def print_time(func):
    def wapper(*args,**wbargs):
        print(f'{func.__name__}()调用于{time.asctime(time.localtime(time.time()))}')
        return func(*args,**wbargs) #不写return 也可以
    return wapper
@print_time
def my_name(name):
    print(f'look!{name}')
my_name("wlb")
```

    my_name()调用于Wed Dec  9 21:21:00 2020
    look!wlb


## **map、reduce、filter**


```python
# map
print(list(map(abs,[-1,-2,-3,-4,-5]))) #也可以自己定义函数或者用匿名函数
# reduce
from functools import reduce #python3中需要从内置库导入
print(reduce(lambda x,y:x+y,list(map(int,str(131351412)))))
# filter
a = [1,2,3,4,5,6,7,8,9]
new_a = filter(lambda x:x%2!=0,a) #filter就是筛选
list(new_a)
# 这三个都是函数式编程中常用的函数
```




## **join()**


```python
# lists = ['1','2','3','4','5']
# ''.join(lists)
lists = [1,2,3,4,5]
''.join(list(map(str,lists))) #join只能是字符串列表，所以要map转换一下
```




    '12345'



## **将多维数组转换为一维**


```python
ab = [[1, 2, 3], [5, 8], [7, 8, 9]]


print([i for item in ab for i in item]) #利用列表解析式

print(sum(ab, [])) # 利用sum函数

from functools import reduce
print(reduce(lambda x,y:x+y,ab)) # 利用reduce

from itertools import chain
print(list(chain(*ab))) # 利用chain


def flatten(items,ignore=(str,bytes)):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore):
            yield from flatten(x)
        else:
            yield x
print(list(flatten(ab))) # 利用自己定义的函数
```

    [1, 2, 3, 5, 8, 7, 8, 9]


## **将一个列表倒序**


```python
lists = [2,4,3,2,5,4]
lists[::-1]
# list(reversed(lists))
```

## **随机生成密码**


```python
import random
b = 8
t = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
print(''.join(random.sample(t,b))) # 主要就是用sample这个方法来取多个随机值
```

    0KmtEZSU

## **断言**


```python
assert(True is True) #成功
print('yes')
assert(True is False) #报错
print('no')
```

    yes


## **合并列表**


```python
list1 = [1,2,31,13]
list2 = [5,2,12,32]
# list1.append(list2)
# print(list1) #错误方法
list1.extend(list2)
print(list1) #正确方法
```

    [1, 2, 31, 13, 5, 2, 12, 32]



```python
a = [1,2,3,4,5]
b = ['a','b','c','d','e']
fin = dict()
for k,i in zip(a,b):
    fin[k] = i
print(fin) # {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
# 或者
d = {}
for i,d[i] in zip(a,b):
    pass
print(d) # {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'} 为什么？在WTFpython中有讲
# 或者
fin = dict(zip(a,b))
print(fin) # {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
```



## 对list进行解包

```python
lists = ['dog','cat','you']
print(*lists)
#想对一个列表进行zip操作时，可以这样
print(list(zip(*lists)))
def test(*args):
    print("args:",args)
test(*lists)
```

```
dog cat you
[('d', 'c', 'y'), ('o', 'a', 'o'), ('g', 't', 'u')]
args:('dog','cat','you')
```

## 对类的一些操作

```python
class Test:
    x = 1
    y = 2
print(Test.x,Test.y) #==>print(Test().x,Test().y)
class Test:
    def __init__(self,x,y):
        self.x = x
        self.y = y
test = Test(1,2)
print(test.x,test.y)
```

```
1 2
1 2
```

```python
class Test:
    def __init__(self,maxlen):
        self.maxlen = maxlen
        self.lists = []
    def put(self,*args):
        for i in args:
            if len(self.lists) <= self.maxlen:
                self.lists.append(i)
            else:
                break
    def get(self):
        return self.lists.pop()
    def empty(self):
        if len(self.lists) != 0:
            return False
        else:
            return True
    def __len__(self):
        return len(self.lists)
    def __del__(self):
        print("del this class")
    def printfs(self):
        return self.lists
test = Test(10)
test.put(1,2,3,4,5,6)
print(test.empty())
print(len(test))
print(test.printfs())
test.__del__()   #直接调用test还存在，__del__是析构函数，垃圾回收时就会调用a
print(test)
#del test
#print(test) 这时候就会报错，因为del将test这个对象直接删除了
```

```
False
6
[1, 2, 3, 4, 5, 6]
del this class
<__main__.Test object at 0x0000021B7DF33EB0>
del this class
```

## 一些内置函数

```python
all([True,True,False]) #False
all([True,True,True]) #True
any([True,True,False]) #True
any([True,False,False])#True
any([False,False]) #False
```

```python
import random
for i in iter(lambda:random.randint(1,10),5):
	print(i)
    
#相当于
while True:
    x = random.randint(1,10)
    print(x)
    if x == 5:
        break
```

```markdown
iter(object[, sentinel])
sentinel为可选参数，若不传入，则object必须为可迭代对象，传入则必须为可调用对象,当可调用对象的返回值为sentinel抛出异常，但for循环会处理这个异常，这常常用于IO操作
```

​		

```python
#这是cookbook里面的一个例子
import sys
f = open('xxx/xxx.txt')
for chunk in iter(lambda:f.read(10),''):
	n = sys.stdout.write(chunk)
```

```python
#深入理解一下
import random
class Test:
    def __init__(self):
        self.lists = [1,23,2,4,1,421,412]
    def __call__(self):
        return random.choice(self.lists)
for i in iter(Test(),1):
    print(i)
#这是可以正常输出的，因为实例化Test后是个可调用对象，返回列表的随机值，当返回1时则循环结束，如果把__call__魔法方法去了后，则会报错，如果想要不使用魔法方法的话可以用匿名函数
import random
class Test:
    def __init__(self):
        self.lists = [1,23,2,4,1,421,412]
    # def __call__(self):
    #     return random.choice(self.lists)
for i in iter(lambda:random.choice(Test().lists),1):
    print(i)
#总之，吹爆cookbook
```

## functools.partial

```python
#先看演示
from functools import partial
def add(a,b):
	return a + b
addOne = partial(add,1)
addOne(2) #3
addOne(4) #5
#大概意思就是利用partial将函数的一个参数固定住了
```

```python
def partial(func,*wargs):
    def wapper(*kargs):
        args = list(wargs)
        print(f"args:{args}")
        print(f"kargs:{kargs}")
        args.extend(kargs)
        print(f"last:{args}")
        return func(*args)
    return wapper

def add(a,b,c):
    return a + b + c

addone = partial(add,1,2) #此时addone相当于wapper
print(addone(3)) #调用wrapper 3为传入的kargs
#输出：
args:[1, 2]
kargs:(3,)
last:[1, 2, 3]
6

#上面是partial函数的简化版本
#很明显的闭包操作，很容易就可以理解
#当然也可以转换为装饰器操作
```

```python
from functools import wraps

from functools import wraps,partial

def out_wapper(*wargs):
    def partialout(func):
        return partial(func,*wargs)
        # 这是使用partial原理的
        # @wraps(func)
        # def wrapper(*kargs):
        #     args = list(wargs)
        #     print(f"args:{args}")
        #     print(f"kargs:{kargs}")
        #     args.extend(kargs)
        #     print(f"last:{args}")
        #     return func(*args)
        # return wrapper
    return partialout

@out_wapper(1,2)
def add(a,b,c):
    return a + b + c

print(add(3)) #6
#明显装饰器要麻烦一点实现，不过毕竟是封装好的函数，以后直接用就可以，不过了解这些有助于提高思维水平
```

## @classmethod和@staticmethod

```python
class A(object):
    bar = 1
    def func1(self):  
        print ('foo') 
    @classmethod
    def func2(cls):
        print ('func2')
        print (cls.bar)
        cls().func1()   # 调用 foo 方法
 
A.func2()               # 不需要实例化
```

```
func2
1
foo
```

```python
class A(object):
    
    # 属性默认为类属性（可以给直接被类本身调用）
    num = "类属性"

    # 实例化方法（必须实例化类之后才能被调用）
    def func1(self): # self : 表示实例化类后的地址id
        print("func1")
        print(self)

    # 类方法（不需要实例化类就可以被类本身调用）
    @classmethod
    def func2(cls):  # cls : 表示没用被实例化的类本身
        print("func2")
        print(cls)
        print(cls.num)
        cls().func1()

    # 不传递传递默认self参数的方法（该方法也是可以直接被类调用的，但是这样做不标准）
    def func3():
        print("func3")
        print(A.num) # 属性是可以直接用类本身调用的
    
# A.func1() 这样调用是会报错：因为func1()调用时需要默认传递实例化类后的地址id参数，如果不实例化类是无法调用的
A.func2()
A.func3() 
```

```python
class A(object):
    def foo(self, x):
        print("executing foo(%s,%s)" % (self, x))
        print('self:', self)
    @staticmethod
    def static_foo(x):
        print("executing static_foo(%s)" % x)   
```

```markdown
问题：@staticmethod修饰的方法函数与普通的类外函数，为什么不直接使用普通函数？
@staticmethod是把函数嵌入到类中的一种方式，函数就属于类，同时表明函数不需要访问这个类。通过子类的继承覆盖，能更好的组织代码。
```

```python
from pydantic import BaseModel
from typing import Sequence


class Test(BaseModel):
    text: Sequence[str]

    @classmethod
    def create(cls,text: Sequence[str]) -> "Test": # classmethod常用构造函数
        return cls(text=text)
    
    def to_tuple(self) -> "Test":
        return Test(text=tuple(self.text))
        
    @classmethod
    def join(cls, *Tests):
        return cls.create(sum([i.text for i in Tests],[]))

test = Test.create(list("Hello world"))
t2 = Test.create(list("NIHAO"))
print(Test.join(test, t2))
# text=['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', 'N', 'I', 'H', 'A', 'O']

```



## 用类实现装饰器

```python
#先看这样的代码  类实现装饰器要求类必须是可调用的
import time
import functools


class DelayFunc:
    def __init__(self,  duration, func):
        self.duration = duration
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f'Wait for {self.duration} seconds...')
        time.sleep(self.duration)
        return self.func(*args, **kwargs)

    def eager_call(self, *args, **kwargs):
        print('Call without delay')
        return self.func(*args, **kwargs)


def delay(duration):
    """装饰器：推迟某个函数的执行。同时提供 .eager_call 方法立即执行
    """
    # 此处为了避免定义额外函数，直接使用 functools.partial 帮助构造
    # DelayFunc 实例
    return functools.partial(DelayFunc, duration)
@delay(2)
def add(a,b):
    print(a,b)
add(1,2) #延迟两秒输出3 相当于delay(2)(add)(1,2)
add.eager_call(1,2) #不延迟输出3  相当于delay(2)(add).eager_call(1,2)
```

```python
#额，当然，想更深入理解的话，也可以这么写
def delay(duration):
    def partial(func):
        return DelayFunc(duration,func)
    return partial
	# 上面的就相当于 partial(DelayFunc,duration),缺的func参数就是要修饰的函数
@delay(2)
def add(a,b):
    return a + b
print(add(1,2))
```

与纯函数相比，我觉得使用类实现的装饰器在**特定场景**下有几个优势：

- 实现有状态的装饰器时，操作类属性比操作闭包内变量更符合直觉、不易出错
- 实现为函数扩充接口的装饰器时，使用类包装函数，比直接为函数对象追加属性更易于维护
- 更容易实现一个同时兼容装饰器与上下文管理器协议的对象

## BaseModel

```python
from pydantic import BaseModel,AnyUrl


class Test(BaseModel): # 继承后可以用类属性创建实例
    url: AnyUrl
    data: str

    def __str__(self):
        return self.url + self.data
    

kwargs = {
    'url': 'https://www.baidu.com',
    'data': '/search'
}
print(Test(**kwargs))
```

## python类型注释

```python
from pydantic import BaseModel
from typing import Any


class cout():
    def __init__(self, cls: "Test", text: str) -> None: 
        self.cls = cls
        self.text = text

    def __str__(self):
        return f"{self.cls}  {self.text}"
#程序到cout时 Test类并没有定义，但最后Test在变量空间中，所以加上引号

class Test(BaseModel):
    def __str__(self) -> str:
        return "I am Test Class"


print(cout(cls=Test(), text="hello world!"))
```

## namedtuple

```python
from collections import namedtuple

Test = namedtuple("Test", ['name', 'age', 'sex'])

def test_for_test(name: str, year: int, sex: str) -> "Test":
    return Test(
        name=name.title(),
        age=2021 - year,
        sex=sex
    )

name,age,sex = test_for_test('wlb', 2002, 'male')
print(name, age, sex)
```

## @property

```python
from pydantic import BaseModel

class Test():

    def __init__(self, cls, n):
        self.cls = cls
        self.n = n 
    @property
    def to_string_cls(self):
        return self.cls
    
    @property
    def to_strings(self):
        return self.n


class Test_For(BaseModel):
    num: int

    def __str__(self):
        return str(self.num)
    
    __repr__ = __str__

test = Test(Test_For, 22)
print(test.to_string_cls(num=1)) # 1
print(test.to_strings) # 22
```

## 	在边界处思考

```python
from typing import Iterable
from pydantic import BaseModel,conint,ValidationError


class NumberInput(BaseModel):
    num: conint(ge=0, le=100)


def input_a_number():
    while True:
        n = input("输入一个数")
        try:
            n = NumberInput(num=n)
        except ValidationError as e:
            print(e)
            continue
        n = n.num
        break
    return n

print(input_a_number()) #要求输入一个0-100的数 这样是不是很优雅
```

## super()进阶

今天学习cookbook8-8`子类中扩展property`  先贴一下代码

```python
class Person:
    def __init__(self, name):
        self.name = name # 有意思的是 这里的self.name是@property修饰的 这行代码调用name.setter

    # Getter function
    @property
    def name(self):
        return self._name

    # Setter function
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value

    # Deleter function
    @name.deleter
    def name(self):
        raise AttributeError("Can't delete attribute")
  
  
  
# 子类

class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name

    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)
```

看到super(SubPerson, SubPerson)感到很疑惑，于是搜索资料大致搞明白了

通俗说默认的super(SubPerson,self) (直接写super()也可) 返回的是一个类的实例

> 为了委托给之前定义的setter方法，需要将控制权传递给之前定义的name属性的 `__set__()` 方法。 不过，获取这个方法的唯一途径是使用类变量而不是实例变量来访问它。 这也是为什么我们要使用 `super(SubPerson, SubPerson)` 的原因。

从书中这句话可以看出 super(cls,cls)返回的是一个类 不是一个实例，super()的参数的作用就是用于定位位置

第一个cls必须是第二cls的父类或者二者相同，可以通过`cls.__mro__`查看继承顺序 比如在D里面`super(A,D).__init__(self)`

而`__mro__` 为 `(<class '__main__.D'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.Base'>, <class 'object'>)` 那么就调用从A以后的类的`__init__()`

不过重点不在这里，重点是super(cls,cls)和super(cls,object)的区别

## dataclass

```python
from dataclasses import dataclass
import random

@dataclass(order=True) # 等于实现了各种比较方法例如=、>、<,排序函数都依赖比较两个对象
class A:
    n: int

nums = [A(random.randint(1,10)) for _ in range(10)]
nums = sorted(nums)
print(nums, end='')
x = '''hello'''
print(x)

```
dataclass可以自动添加__rapr__方法，不必自己实现
```@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)```

1. `init`：默认将生成 `__init__` 方法。如果传入 `False`，那么该类将不会有 `__init__` 方法。
2. `repr`：`__repr__` 方法默认生成。如果传入 `False`，那么该类将不会有 `__repr__` 方法。
3. `eq`：默认将生成 `__eq__` 方法。如果传入 `False`，那么 `__eq__` 方法将不会被 `dataclass` 添加，但默认为 `object.__eq__`。
4. `order`：默认将生成 `__gt__`、`__ge__`、`__lt__`、`__le__` 方法。如果传入 `False`，则省略它们。
5. `unsafe_hash`：默认生成__hash__方法，用于构建可hashable的类
```python
from dataclasses import dataclass
@dataclass(unsafe_hash=True)
class VisitRecordDC:
    first_name: str
    last_name: str
    phone_number: str
    # 跳过“访问时间”字段，不作为任何对比条件
    date_visited: str = field(hash=False, compare=False)


def find_potential_customers_v4():
    return set(VisitRecordDC(**r) for r in users_visited_phuket) - \ #求差集
        set(VisitRecordDC(**r) for r in users_visited_nz)
```

## 自定义format

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __format__(self, format_spec):
        if format_spec == 'long':
            return f'{self.name} is {self.age} years old.'
        elif format_spec == 'simple':
            return f'{self.name}({self.age})'
        raise ValueError('invalid format spec')


vllbc = Student('vllbc', '18')
print(f'{vllbc:simple}')
print(f'{vllbc:long}')
```

## 抽象类实践

```python
import collections
from abc import ABC, abstractmethod
from typing import List

customer = collections.namedtuple('customer', ['name', 'points'])


class Goods():
    def __init__(self, name: str, quantity: float, price: float) -> None:
        self.name = name
        self.quantity = quantity
        self.price = price

    def total(self) -> float:
        return self.quantity * self.price


class Order():
    def __init__(self, customer: customer, cart: List[Goods], prom=None) -> None:
        self.customer = customer
        self.cart = cart
        self.prom = prom

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(i.total() for i in self.cart)
        return self.__total

    def due(self):
        if self.prom is None:
            discount = 0
        else:
            discount = self.prom.discount(self)
        return self.total() - discount

    def __repr__(self) -> str:
        return f'<Order total: {self.total():.2f} due: {self.due():.2f}>'


class Prom(ABC): # 抽象类
    @abstractmethod
    def discount(self,order) -> float:
        '''discount'''

class discount1(Prom):
    def discount(self,order) -> float:
       return order.total() * 0.05 if order.customer.points >= 10000 else 0 

john = customer(name='vllbc', points=100000)
carts = [Goods(name='apple', quantity=5, price=10), Goods(
    name='banana', quantity=8, price=5), Goods(name='peach', quantity=4, price=8)]
order = Order(customer=john, cart=carts,prom=discount1())
```

## accumulate

```python
import itertools

test_list = [i for i in range(1, 11)]

for i in itertools.accumulate(test_list):
    print(i, end=",") # 1,3,6,10,15,21,28,36,45,55,
print()
for i in itertools.accumulate(test_list, lambda x, y: x * y):
    print(i, end=',') # 1,2,6,24,120,720,5040,40320,362880,3628800,
```

## 异步装饰器

```python
from functools import wraps
import asyncio

def decorator(func):
    @wraps(func)
    async def hello(*args, **kwargs):
        await asyncio.sleep(2)
        return await func(*args,**kwargs)
    return hello

@decorator
async def test():
    print("hello")
asyncio.run(test())
```

## bisect

```python
import bisect
import time

# BREAKPOINTS 必须是已经排好序的，不然无法进行二分查找
BREAKPOINTS = (1, 60, 3600, 3600 * 24)
TMPLS = (
    # unit, template
    (1, "less than 1 second ago"),
    (1, "{units} seconds ago"),
    (60, "{units} minutes ago"),
    (3600, "{units} hours ago"),
    (3600 * 24, "{units} days ago"),
)


def from_now(ts):
    """接收一个过去的时间戳，返回距离当前时间的相对时间文字描述
    """
    seconds_delta = int(time.time() - ts)
    unit, tmpl = TMPLS[bisect.bisect(BREAKPOINTS, seconds_delta)] # bisect类似于index方法，要是不存在会选择数值最接近的索引
    return tmpl.format(units=seconds_delta // unit)

```

## contextlib

```python
from contextlib import contextmanager,ContextDecorator

# contextmanager可以把一个函数变成一个上下文管理器，不需要自己去实现一个定义了__enter__和__exit__方法的类
@contextmanager
def open_file(filename, methods="r"):
    print(f"打开了文件{filename}")
    res_file = open(filename, mode=methods)  # __enter__方法 这里也可以是自己定义的类

    try:
        yield res_file    # 相当于在__enter__方法里面返回self  yield后面为空的话就不用as了
    except Exception as e:
        print("有错误发生", e)  # __exit__方法里的错误处理
    finally:
        res_file.close()  # __exit__


with open_file("testvim.txt") as fp:
    print(fp)
```

## 读取大文件

```python
from functools import partial

def digits_(file,block_size=1024*8): # 分块读取
    _read = partial(file.read, block_size) # 使用partial,也可以使用lambda:file.read(block_size)
    for line in iter(_read, ""): # 当读取完毕时退出
        for s in line:
            if s.isdigit():
                yield s # 使用yield

def count_digits(fname):
    """计算文件里包含多少个数字字符"""
    count = 0
    with open(fname) as file:
        for _ in digits_(file=file):
            count+=1
    return count
```

## \__exit__、\__enter__

```python
#     def __enter__(self):
#         该方法将在进入上下文时调用
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         该方法将在退出上下文时调用
#         exc_type, exc_val, exc_tb 分别表示该上下文内抛出的异常类型、异常值、错误栈
#         __enter__()：主要执行一些环境准备工作，同时返回一资源对象。如果上下文管理器open("test.txt")的__enter__()函数返回一个文件对象。
#         __exit__()：完整形式为__exit__(type, value, traceback),这三个参数和调用sys.exec_info()函数返回值是一样的，分别为异常类型、异常信息和堆栈。如果*执行体语句*没有引发异常，则这三个参数均被设为None。否则，它们将包含上下文的异常信息。__exit__()方法返回True或False,分别指示被引发的异常有没有被处理，如果返回False，引发的异常将会被传递出上下文。如果__exit__()函数内部引发了异常，则会覆盖掉执行体的中引发的异常。处理异常时，不需要重新抛出异常，只需要返回False，with语句会检测__exit__()返回False来处理异常。
```

## enum

```python
from enum import IntEnum

class Test(IntEnum):
    X = 2
    Y = 1

print(2 == Test.X)
```

## pydantic数据验证

```python
from pydantic import BaseModel, conint, ValidationError 
# pydantic主要功能就是作数据验证
from typing import (
    List, 
    Union,
    Optional,
    Dict
)

class Test(BaseModel):
    name: Optional[str]
    sex: Union[str, List[str]]
    d: Dict[str, int]
    id: conint(ge=1,le=10)
try:
    test = Test(name='wlb', sex='male', d={'dict':1}, id=1)
    print(test.dict(), test.__annotations__)
    # {'name': 'wlb', 'sex': 'male', 'd': {'dict': 1}, 'id': 1} {'name': typing.Union[str, NoneType], 'sex': typing.Union[str, typing.List[str]], 'd': typing.Dict[str, int], 'id': <class '__main__.ConstrainedIntValue'>}
except ValidationError:
    print("数据错误")
```

## islice

```python
from itertools import islice


def test():
    t = 0
    while True:
        yield t
        t += 1
        
for i in islice(test(), 10, 21, 2):
    print(i)
```

## \__iter__、\__next__

```python
class Range7: # 可迭代类型 只需要实现__iter__即可
    def __init__(self,start,end) -> None:
        self.start = start
        self.end = end
    def __iter__(self):
        return Range7iterator(self)
        


class Range7iterator: #这是迭代器,一般的迭代器只能调用一次
    def __init__(self,rangeobj) -> None:
        self.rangeobj = rangeobj
        self.cur = rangeobj.start

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            if self.cur > self.rangeobj.end:
                raise StopIteration
            if self.is_7(self.cur):
                res = self.cur
                self.cur += 1
                return res
            self.cur += 1
    def is_7(self,num):
        if num == 0:
            return False
        return num%7==0 or "7" in str(num)
for i in Range7(1,100):
    print(i,end=" ")

#可迭代对象不一定是迭代器，但迭代器一定是可迭代对象

# 对可迭代对象使用 iter() 会返回迭代器，迭代器则会返回它自身

# 每个迭代器的被迭代过程是一次性的，可迭代对象则不一定

# 可迭代对象只需要实现 __iter__ 方法，而迭代器要额外实现 __next__ 方法
```

## 求字典的最大值

```python
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
print(max(zip(prices.values(),prices.keys())))
print(max(prices.items(),key=lambda x:x[1]))
print(max(prices,key=lambda k:prices[k]))

```

## 注意循环变量

```python
funcs = []
res = []
for x in range(7):
    def func(x=x): # 去掉x=x则出现[6,6,6,6,6,6] 在循环内部定义一个函数时, 如果该函数在其主体中使用了循环变量, 则闭包函数将与循环变量绑定, 而不是它的值. 因此, 所有的函数都是使用最后分配给变量的值来进行计算的.
        return x
    funcs.append(func)
    res.append(func())

func_res = [f() for f in funcs]
print(func_res)
def create_mult():
    res = []
    for i in range(5):
        def func(x, i=i): # 去掉i=i则全输出8，原因和上面一样
            return x * i
        res.append(func)
    return res
for cr in create_mult():
    print(cr(2))

```

## 空对象模式

```python
修改前
import decimal

class CreateAccountError(Exception):
    """Unable to create a account error"""

class Account:
    """一个虚拟的银行账号"""

    def __init__(self, username, balance):
        self.username = username
        self.balance = balance

    @classmethod
    def from_string(cls, s):
        """从字符串初始化一个账号"""
        try:
            username, balance = s.split()
            balance = decimal.Decimal(float(balance))
        except ValueError:
            raise CreateAccountError('input must follow pattern "{ACCOUNT_NAME} {BALANCE}"')

        if balance < 0:
            raise CreateAccountError('balance can not be negative')
        return cls(username=username, balance=balance)


def caculate_total_balance(accounts_data):
    """计算所有账号的总余额
    """
    result = 0
    for account_string in accounts_data:
        try:
            user = Account.from_string(account_string)
        except CreateAccountError:
            pass
        else:
            result += user.balance
    return result


accounts_data = [
    'piglei 96.5',
    'cotton 21',
    'invalid_data',
    'roland $invalid_balance',
    'alfred -3',
]

print(caculate_total_balance(accounts_data))
```

### 空对象模式简介

>  **额外定义一个对象来表示None**
>

### 好处

1. 它可以加强系统的稳固性，能有有效地防止空指针报错对整个系统的影响，使系统更加稳定。
2. 它能够实现对空对象情况的定制化的控制，能够掌握处理空对象的主动权。
3. 它并不依靠Client来保证整个系统的稳定运行。
4. 它通过isNone对==None的替换，显得更加优雅，更加易懂。

```python
import decimal


class Account:
    """一个虚拟的银行账号"""

    def __init__(self, username, balance):
        self.username = username
        self.balance = balance

    @classmethod
    def from_string(cls, s):
        """从字符串初始化一个账号"""
        try:
            username, balance = s.split()
            balance = decimal.Decimal(float(balance))
        except ValueError:
            # raise CreateAccountError('input must follow pattern "{ACCOUNT_NAME} {BALANCE}"')
            return NullAccount()

        if balance < 0:
            return NullAccount()
        return cls(username=username, balance=balance)


def caculate_total_balance(accounts_data):
    """计算所有账号的总余额
    """
    return sum(Account.from_string(s).balance for s in accounts_data)


class NullAccount: # 要返回的空对象
    username = ""  # 当发生错误时username的值
    balance = 0  # 当发生错误时balance的值

    def re_Null():
        return NotImplementedError


accounts_data = [
    'piglei 96.5',
    'cotton 21',
    'invalid_data',
    'roland $invalid_balance',
    'alfred -3',
]

print(caculate_total_balance(accounts_data))
```

## pathlib

```python
from pathlib import Path


# 把txt文件重命名为csv文件
def unify_ext_with_pathlib(path):
    for fpath in Path(path).glob("*.txt"):
        fpath.rename(fpath.with_suffix(".csv"))


print(Path(".") / "test_pathlib.py")  # Path类型可以使用/运算符
print(Path("testvim.txt").read_text())  # 直接读取文件内容

# .resolve() 取绝对路径
# with_name() 修改文件名 with_suffix()修改后缀名
# 把当前目录下的文件批量重命名
# import os
# from pathlib import Path

# p = Path(".")

# for filepath in p.glob("test_*.py"):
#     name = filepath.with_name(str(filepath).replace("test_",""))
#     filepath.rename(name)

```

## 单元测试

```python
def say_hello(name=None):
    if name:
        return f"hello {name}"
    return "hello world"


import unittest
from typing import List


class sayhellotest(unittest.TestCase):
    def setUp(self,nums:List[int] = 0):
        return super().setUp()

    def tearDown(self):
        return super().tearDown()
    
    def test_sayhello(self):
        rv = say_hello()
        self.assertEqual(rv,"hello world")
    
    def test_to_name(self):
        rv = say_hello("wlb")
        self.assertEqual(rv,"hello wlb")
    

if __name__ == '__main__':
    unittest.main()
```

## takewhile和dropwhile

```python
from itertools import dropwhile,takewhile
# 你想遍历一个可迭代对象，但是它开始的某些元素你并不感兴趣，想跳过它们，用dropwhile

with open('testvim.txt','r') as fp:
    for i in dropwhile(lambda i:i.startswith("#"),fp): # 跳过前面#号开头的
        print(i)

with open("testvim.txt","r") as fp:
    for i in takewhile(lambda i:i.startswith("#"),fp): # 遍历带#号开头的，遇到不是#号开头的就退出循环，可以当做break使用
        # 相当于 if not i.startwith("#"): break
        print(i)
```

## 装饰器可以装饰方法

```python
import random
import wrapt # 为第三方库

def provide_number(min_num, max_num):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        # 参数含义：
        #
        # - wrapped：被装饰的函数或类方法
        # - instance：
        #   - 如果被装饰者为普通类方法，该值为类实例
        #   - 如果被装饰者为 classmethod 类方法，该值为类
        #   - 如果被装饰者为类/函数/静态方法，该值为 None
        #
        # - args：调用时的位置参数（注意没有 * 符号）
        # - kwargs：调用时的关键字参数（注意没有 ** 符号）
        #
        num = random.randint(min_num, max_num)
        # 无需关注 wrapped 是类方法或普通函数，直接在头部追加参数
        args = (num,) + args
        return wrapped(*args, **kwargs)
    return wrapper
    


@provide_number(1, 100)
def print_random_number(num):
    print(num)
    
class Foo:
    @provide_number(1, 100)
    def print_random_number(self, num):
        print(num)

Foo().print_random_number()
print_random_number()

# 使用 wrapt 模块编写的装饰器，相比原来拥有下面这些优势：

# 嵌套层级少：使用 @wrapt.decorator 可以将两层嵌套减少为一层
# 更简单：处理位置与关键字参数时，可以忽略类实例等特殊情况
# 更灵活：针对 instance 值进行条件判断后，更容易让装饰器变得通用
```
## _\__getattribute__
__\__getattribute__仅在新式类中可用，重载__getattrbute__方法对类实例的每个属性访问都有效。
```python
class ClassA:

    x = 'a'

    def __init__(self):
        self.y = 'b'

    def __getattribute__(self, item):
        return '__getattribute__'


if __name__ == '__main__':
    a = ClassA()
    # 使用实例直接访问存在的类属性时,会调用__getattribute__方法
    # 输出结果 __getattribute__
    print(a.x)
    # 使用实例直接访问实例存在的实例属性时,会调用__getattribute__方法
    # 输出结果 __getattribute__
    print(a.y)
    # 使用实例直接访问实例不存在的实例属性时,也会调用__getattribute__方法
    # 输出结果 __getattribute__
    print(a.z)
```

由于__getattr__只针对未定义属性的调用，所以它可以在自己的代码中自由地获取其他属性，而__getattribute__针对所有的属性运行，因此要十分注意避免在访问其他属性时，再次调用自身的递归循环。

当在__getattribute__代码块中，再次执行属性的获取操作时，会再次触发__getattribute__方法的调用，代码将会陷入无限递归，直到Python递归深度限制（重载__setter__方法也会有这个问题）。

示例代码（无限递归）：
```python
class ClassA:

    x = 'a'

    def __getattribute__(self, item):
        print('__getattribute__')
        return self.item


if __name__ == '__main__':
    a = ClassA()
    a.x
```
运行结果引发异常。

同时，也没办法通过从__dict__取值的方式来避免无限递归

```python
class ClassA:

    x = 'a'

    def __getattribute__(self, name):
        return self.__dict__[name]


if __name__ == '__main__':
    a = ClassA()
    # 无限递归
    a.x
```

为了避免无限递归，应该把获取属性的方法指向一个更高的超类，例如object（因为__getattribute__只在新式类中可用，而新式类所有的类都显式或隐式地继承自object，所以对于新式类来说，object是所有新式类的超类）。

修改代码（避免无限递归循环）：
```python
class ClassA:

    x = 'a'

    def __getattribute__(self, item):
        print('__getattribute__')
        return super().__getattribute__(self, item)


if __name__ == '__main__':
    a = ClassA()
    print(a.x)
```
结果：
```
__getattribute__
a
```
## _\__getattr__、\__setattr__
区别 __getattribute__ 和 __getattr__，前者是任何通过 x.y 访问实例的属性时都会调用的特殊方法，而后者则是在正常访问形式下无法找到的情况下才会被调用。

```python
class Chain(object):
    
    def __init__(self, path=''):
        self._path = path

    def __getattr__(self, path):
        return Chain('%s/%s' % (self._path, path))

    def __str__(self):
        return self._path
    
    def users(self,name):
        return Chain(f"{self._path}/users/{name}")

    __repr__ = __str__

chain = Chain("vllbc")
print(chain.x.x.x.x.x) # out: vllbc/x/x/x/x/x
```
另外，当同时定义__getattribute__和__getattr__时，__getattr__方法不会再被调用，除非显示调用__getattr__方法或引发AttributeError异常。可以在__getattribute__中抛出异常来调用__getattr__。

## \__getitem__


## 元类

```python
'''
    元类就是控制类的创建的类
'''


class ModelMetaclass(type):

    def __new__(cls, name, bases, attrs):
        if name == 'Model':
            return type.__new__(cls, name, bases, attrs)
        print(f"found model {name}")
        maps = dict()
        for k, v in attrs.items():
            if isinstance(v, Field):
                print(f"Found mapping {k} ==> {v}")
                maps[k] = v
        for k, v in maps.items():
            attrs.pop(k)
        attrs['__mappings__'] = maps
        attrs['__table__'] = name
        return type.__new__(cls, name, bases, attrs)


class Field(object):

    def __init__(self, name, column_type):
        self.name = name
        self.column_type = column_type

    def __str__(self):
        return '<%s:%s>' % (self.__class__.__name__, self.name)


class StringField(Field):
    def __init__(self, name, column_type='TXT'):
        super().__init__(name, column_type)


class IntegerField(Field):
    def __init__(self, name, column_type='INT'):
        super().__init__(name, column_type)


class Model(dict, metaclass=ModelMetaclass):

    def __init__(self, **kw):
        super(Model, self).__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Model' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

    def save(self):
        fields = []
        params = []
        args = []
        for k, v in self.__mappings__.items():
            fields.append(k)
            params.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into %s (%s) values (%s)' % (
            self.__table__, ','.join(fields), ','.join(params))
        print('SQL: %s' % sql)
        print('ARGS: %s' % str(args))


class User(Model):
    # 定义类的属性到列的映射：
    id = IntegerField('id')
    name = StringField('username')
    email = StringField('email')
    password = StringField('password')


# 创建一个实例：
u = User(id=12345, name='Michael', email='test@orm.org', password='my-pwd')
# 保存到数据库：
u.save()
```

## 关于hash
```python
# set中的元素要求必须是可哈希的，但set本身是不可哈希的
a = set([1,2,3,4])
a.add([1,2,3]) # 会报错，因为列表是不可哈希的
hash(a)  # 会报错，因为set本身是不可哈希的
b = tuple([1,2,3,4]) # 元组本身是可哈希的
c = tuple([1,2,3,[4,5]]) # 如果元组里面有不可哈希的元素 那么整个元组也是不可哈希的了
# 在类中定义__hash__方法就可以变成可哈希的类，注意避免返回可能重复的hash值
class My_Hash:
    def __hash__(self) -> int:
        return 111
# 还有简便的方法就是使用dataclass类，可以省时省力，本博客也有介绍。
```
## 处理列表越界
假如你请求的不是某一个元素，而是一段范围的切片。那么无论你指定的范围是否有效，程序都只会返回一个空列表 []，而不会抛出任何错误。
了解了这点后，你会发现像下面这种边界处理代码根本没有必要：
```python
def sum_list(l, limit):
    """对列表的前 limit 个元素求和
    """
    # 如果 limit 过大，设置为数组长度避免越界
    if limit > len(l):
        limit = len(l)
    return sum(l[:limit])
```
因为做切片不会抛出任何错误，所以不需要判断 limit 是否超出范围，直接做 sum 操作即可：

```python
def sum_list(l, limit):
    return sum(l[:limit])
```
## or操作符
在很多场景下，我们可以利用 or 的特点来简化一些边界处理逻辑。看看下面这个例子：
```python
context = {}
# 仅当 extra_context 不为 None 时，将其追加进 context 中
if extra_context:
    context.update(extra_context)
# 等同于
context.update(extra_context or {})
```
因为 a or b or c or ... 这样的表达式，会返回这些变量里第一个布尔值为真的值，直到最后一个为止。
含义为当extra_context为None时，会返回{}
## 字典的键
在python里面，Python 字典通过检查键值是否相等和比较哈希值来确定两个键是否相同.具有相同值的不可变对象(可哈希)在Python中始终具有相同的哈希值。
**注意:** 具有不同值的对象也可能具有相同的哈希值（哈希冲突）。
下面这个例子
```python
some_dict = {}
some_dict[5.5] = "Ruby"
some_dict[5.0] = "JavaScript"
some_dict[5] = "Python"
```
```python
>>> some_dict[5.5]
"Ruby"
>>> some_dict[5.0]
"Python"
>>> some_dict[5]
"Python"
```
因为Python将 `5` 和 `5.0` 识别为 `some_dict` 的同一个键, 所以已有值 "JavaScript" 就被 "Python" 覆盖了.


