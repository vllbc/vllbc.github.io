# WTF



## 可变与不可变

“-=”操作符会调用__isub__函数，而"-"操作符会调用__sub__函数，一般对于可变对象来说“-=”操作符会直接改变self自身。

```python
import torch

x1 = 1
x2 = 2
params = [x1, x2]
for p in params:
    print(id(p), id(x1), id(x2))
    p -= 4
    print(id(p), id(x1), id(x2))
print(params)

x1 = torch.Tensor([1])
x2 = torch.Tensor([2])
params = [x1, x2]
for p in params:
    print(id(p), id(x1), id(x2))
    p -= 4
    print(id(p), id(x1), id(x2))
print(params)
```

```
9784896 9784896 9784928
9784768 9784896 9784928
9784928 9784896 9784928
9784800 9784896 9784928
[1, 2]
139752445458112 139752445458112 139752445458176
139752445458112 139752445458112 139752445458176
139752445458176 139752445458112 139752445458176
139752445458176 139752445458112 139752445458176
[tensor([-3.]), tensor([-2.])]
```

可以看到对于int类型，地址变换了，而torch类型，地址却没有变化。
p -= 4等价于p.sub_(4)。这个可变对象改变了自身。写成p = p - 4则会调用构造函数，并返回一个新的变量，也就不可能作用到原先的“可变对象”。
int类没有发生就地变化是因为它是一个不可变对象。

这是python "-" 与"-="的一个坑

#### 微妙的字符串


```python
a = 'some_thing'
b = 'some'+'_'+'thing'
id(a),id(b)
```




    (1957716471920, 1957716471920)




```python
a = 'wtf'
b = 'wtf'
a is b
```




    True




```python
a = 'wtf!'
b = 'wtf!'
a is b
```




    False




```python
a,b = 'wtf!','wtf!'
a is b
```




    True




```python
'a'*20 is 'aaaaaaaaaaaaaaaaaaaa','a'*21 is 'aaaaaaaaaaaaaaaaaaaaa'
```




    (True, False)



 Cpython 在编译优化时, 某些情况下会尝试使用已经存在的不可变对象,成为字符串驻留
 发生驻留之后, 许多变量可能指向内存中的相同字符串对象
 所有长度为 0 和长度为 1 的字符串都被驻留.

字符串在编译时被实现 ('wtf' 将被驻留, 但是 ''.join(['w', 't', 'f'] 将不会被驻留)
字符串中只包含字母，数字或下划线时将会驻留. 所以 'wtf!' 由于包含 ! 而未被驻留。

当在同一行将 a 和 b 的值设置为 "wtf!" 的时候, Python 解释器会创建一个新对象, 然后同时引用第二个变量.

常量折叠(constant folding) 是 Python 中的一种 窥孔优化(peephole optimization) 技术. 这意味着在编译时表达式 'a'*20 会被替换为 'aaaaaaaaaaaaaaaaaaaa' 以减少运行时的时钟周期. 只有长度小于 20 的字符串才会发生常量折叠. 


```python
a = 1
b = 1
a is b,id(a) == id(b)
```




    (True, True)



is 是比较对象是否相同(is 表示对象标识符即 object identity)，即用 id() 函数查看的地址是否相同，如果相同则返回 True，如果不同则返回 False。is 不能被重载。

== 是比较两个对象的值是否相等，此操作符内部调用的是 ___eq__() 方法。所以 a==b 等效于a.___eq__(b)，所以 = 可以被重载

## 是时候来点蛋糕了!


```python
some_dict = {}
some_dict[5.5] = 'ruby'
some_dict[5.0] = 'javascript'
some_dict[5] = 'python'
print(some_dict[5.0])
```

    python



```python
5 == 5.0,hash(5) == hash(5.0)
```




    (True, True)



Python 字典通过检查键值是否相等和比较哈希值来确定两个键是否相同.
具有相同值的不可变对象在Python中始终具有相同的哈希值

## 本质上,我们都一样


```python
class WTF:
    pass
print(WTF() == WTF(),WTF() is WTF())
print(hash(WTF()) == hash(WTF()))
print(id(WTF()) == id(WTF()))
```

    False False
    True
    True


当调用 id 函数时, Python 创建了一个 WTF 类的对象并传给 id 函数. 然后 id 函数获取其id值 (也就是内存地址), 然后丢弃该对象. 该对象就被销毁了.

当我们连续两次进行这个操作时, Python会将相同的内存地址分配给第二个对象. 因为 (在CPython中) id 函数使用对象的内存地址作为对象的id值, 所以两个对象的id值是相同的.


```python
print(id(id(WTF())) == id(id(WTF()))) ##无论多少个ID都是True 原因就在上面
##虽然id(id(WTF())) == id(id(WTF())) 但是id(WTF()) is id(WTF()) 返回True
##原因就是id这个函数调用的过程特殊性
print(id(WTF()) is id(WTF())) 
```

    True
    False



```python
class WTF(object):
    def __init__(self): 
        print("I")
    def __del__(self): 
        print("D")
```


```python
WTF() is WTF() ##这时是两个对象一起创建，然后一起销毁，所以id不一样
```

    I
    I
    D
    D





    False




```python
id(WTF()) == id(WTF()) ##这时候先创建一个销毁，然后再创建。对象销毁的顺序是造成所有不同之处的原因.
```

    I
    D
    I
    D





    True



## 为什么？


```python
some_string = "wtf"
some_dict = {}
for i, some_dict[i] in enumerate(some_string):
    pass
```


```python
some_dict
```

[Python 语法](https://docs.python.org/3/reference/grammar.html) 中对 `for` 的定义是:


    {0: 'w', 1: 't', 2: 'f'}

```
for_stmt: 'for' exprlist 'in' testlist ':' suite ['else' ':' suite]
```

其中 `exprlist` 指分配目标. 这意味着对可迭代对象中的**每一项都会执行**类似 `{exprlist} = {next_value}` 的操作.


```python
for i in range(4):
    print(i)
    i = 10
```

    0
    1
    2
    3


## 列表副本


```python
list1 = [1,2,3,4,5]
list2 = list1
list2[0] = 6
print(list1,list2)
```

    [6, 2, 3, 4, 5] [6, 2, 3, 4, 5]



```python
list1 = [1,2,3,4,5]
list2 = list1[:]
list2[0] = 6
print(list1,list2)
```

    [1, 2, 3, 4, 5] [6, 2, 3, 4, 5]


## 执行时机差异


```python
array = [1, 8, 15]
g = (x for x in array if array.count(x) > 0) ##这时候x为[1,8,15]的解包
##而后面的array变成了下面的
array = [2, 8, 22]
print(list(g))
```

    [8]


在生成器表达式中, in 子句在声明时执行, 而条件子句则是在运行时执行.
所以在运行前, array 已经被重新赋值为 [2, 8, 22], 因此对于之前的 1, 8 和 15, 只有 count(8) 的结果是大于 0 的, 所以生成器只会生成 8.


```python
array_1 = [1,2,3,4]
g1 = (x for x in array_1)
array_1 = [1,2,3,4,5]

array_2 = [1,2,3,4]
g2 = (x for x in array_2)
array_2[:] = [1,2,3,4,5]
print(list(g1))
print(list(g2))
```

    [1, 2, 3, 4]
    [1, 2, 3, 4, 5]


第二部分中 g1 和 g2 的输出差异则是由于变量 array_1 和 array_2 被重新赋值的方式导致的.

在第一种情况下, array_1 被绑定到新对象 [1,2,3,4,5], 因为 in 子句是在声明时被执行的， 所以它仍然引用旧对象 [1,2,3,4].

在第二种情况下, 对 array_2 的切片赋值将相同的旧对象 [1,2,3,4] 原地更新为 [1,2,3,4,5]. 因此 g2 和 array_2 仍然引用同一个对象(这个对象现在已经更新为 [1,2,3,4,5]).

## 出人意料的is


```python
a = 256
b = 256
a is b
```




    True




```python
a = 257 
b = 257  ##256 是一个已经存在的对象, 而 257 不是
##当你启动Python 的时候, -5 到 256 的数值就已经被分配好了.
##这些数字因为经常使用所以适合被提前准备好
a is b
```




    False




```python
a,b = 257,257 ##当 a 和 b 在同一行中使用相同的值初始化时，会指向同一个对象.
print(a is b)
print(id(a),id(b))
```

    True
    1957717387056 1957717387056



```python
[] == []
```




    True




```python
[] is [] ##两个空列表位于不同的内存地址
```




    False



## 一蹴即至!


```python
row = [""] * 3
board = [row] * 3
board
```




    [['', '', ''], ['', '', ''], ['', '', '']]




```python
board[0][0] = 'X'
board ##这是因为之前对row做乘法导致的
```




    [['X', '', ''], ['X', '', ''], ['X', '', '']]




```python
##如何避免这种情况？
board = [['']*3 for _ in range(3)]
board[0][0] = 'X'
board
```




    [['X', '', ''], ['', '', ''], ['', '', '']]



## 麻烦的输出


```python
funcs = []
res = []
for x in range(7):
    def func():
        return x
    funcs.append(func)
    res.append(func())
func_res = [func() for func in funcs]
print(func_res,res)
```

    [6, 6, 6, 6, 6, 6, 6] [0, 1, 2, 3, 4, 5, 6]



```python
power_x = [lambda x:x**i for i in range(11)]
print([func(2) for func in power_x])
```

    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]


在循环内部定义一个函数时, 如果该函数在其主体中使用了循环变量, 则闭包函数将与循环变量绑定, 而不是它的值. 因此, 所有的函数都是使用最后分配给变量的值来进行计算的.

## 连Python也知道爱是难言的


```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!



```python
love = this
```


```python
this is love
```




    True




```python
love is True
```




    False




```python
love is False
```




    False




```python
love is not True or False
```




    True




```python
love is not True or False;love is love
```




    True



## 三个引号


```python
print('wtfpython''')
```

    wtfpython



```python
print("wtf" "python")
```

    wtfpython


## 布尔你咋了?


```python
mixed_list = [False, 1.0, "some_string", 3, True, [], False]
integers_found_so_far = 0
booleans_found_so_far = 0

for item in mixed_list:
    if isinstance(item, int):
        integers_found_so_far += 1
    elif isinstance(item, bool):
        booleans_found_so_far += 1
```


```python
integers_found_so_far
```




    4




```python
booleans_found_so_far
```




    0




```python
another_dict = {}
another_dict[True] = "JavaScript"
another_dict[1] = "Ruby"
another_dict[1.0] = "Python"
```


```python
another_dict[True]
```




    'Python'



布尔值是 int 的子类


```python
some_iterable = ('a', 'b')

def some_func(val):
    return "something"
```


```python
[x for x in some_iterable]
```




    ['a', 'b']




```python
[(yield x) for x in some_iterable]
```




    <generator object <listcomp> at 0x000001CC6FFC3888>




```python
list([(yield x) for x in some_iterable])
```




    ['a', 'b']




```python
list(((yield x) for x in some_iterable))
```




    ['a', None, 'b', None]




```python
list(some_func((yield x)) for x in some_iterable)
```




    ['a', 'something', 'b', 'something']



## 消失的外部变量


```python
e = 7
try:
    raise Exception()
except Exception as e:
    pass
```


```python
print(e) ##error!
```

## 从有到无


```python
some_list = [1, 2, 3]
some_dict = {
  "key_1": 1,
  "key_2": 2,
  "key_3": 3
}

some_list = some_list.append(4)
some_dict = some_dict.update({"key_4": 4})
```


```python
some_list
```


```python
some_dict
```

大多数修改序列/映射对象的方法, 比如 list.append, dict.update, list.sort 等等. 都是原地修改对象并返回 None. 这样做的理由是, 如果操作可以原地完成, 就可以避免创建对象的副本来提高性能. 

## 迭代列表时删除元素


```python
list_1 = [1, 2, 3, 4]
list_2 = [1, 2, 3, 4]
list_3 = [1, 2, 3, 4]
list_4 = [1, 2, 3, 4]

for idx, item in enumerate(list_1):
    del item

for idx, item in enumerate(list_2):
    list_2.remove(item)

for idx, item in enumerate(list_3[:]):
    list_3.remove(item)

for idx, item in enumerate(list_4):
    list_4.pop(idx)
```


```python
list_1 ##没有修改list_1
```




    [1, 2, 3, 4]




```python
list_2 ##每一次删除元素后 迭代的list_2也发生改变 比如第一次删除了1 list_2为[2,3,4]这时idx=1 所以下一个删除了3
```




    [2, 4]




```python
list_3 ##迭代副本不会出现上述情况
```




    []




```python
list_4
```




    [2, 4]



## 循环变量泄露



```python
for x in range(7):
    if x == 6:
        print(x, ': for x inside loop')
print(x, ': x in global')
```

    6 : for x inside loop
    6 : x in global



```python
## 这次我们先初始化x
x = -1
for x in range(7):
    if x == 6:
        print(x, ': for x inside loop')
print(x, ': x in global')
```

    6 : for x inside loop
    6 : x in global



```python
x = 1
print([x for x in range(5)])
print(x, ': x in global')
```

    [0, 1, 2, 3, 4]
    1 : x in global


## 当心默认的可变参数


```python
def some_func(default_arg=[]):
    default_arg.append("some_string")
    return default_arg
```


```python
some_func()
```




    ['some_string']




```python
some_func()
```




    ['some_string', 'some_string']




```python
some_func([])
```




    ['some_string']




```python
some_func()
```




    ['some_string', 'some_string', 'some_string']



Python中函数的默认可变参数并不是每次调用该函数时都会被初始化. 相反, 它们会使用最近分配的值作为默认值. 当我们明确的将 [] 作为参数传递给 some_func 的时候, 就不会使用 default_arg 的默认值, 所以函数会返回我们所期望的结果.


```python
some_func.__defaults__
```




    (['some_string', 'some_string', 'some_string'],)



避免可变参数导致的错误的常见做法是将 None 指定为参数的默认值, 然后检查是否有值传给对应的参数. 例:


```python
def some_func(default_arg=None):
    if not default_arg:
        default_arg = []
    default_arg.append("some_string")
    return default_arg
```

## 同人不同命


```python
a = [1, 2, 3, 4]
b = a
a = a + [5, 6, 7, 8]
```


```python
a
```




    [1, 2, 3, 4, 5, 6, 7, 8]




```python
b
```




    [1, 2, 3, 4]




```python
a = [1, 2, 3, 4]
b = a
a += [5, 6, 7, 8]
```


```python
a
```




    [1, 2, 3, 4, 5, 6, 7, 8]




```python
b
```




    [1, 2, 3, 4, 5, 6, 7, 8]



a += b 并不总是与 a = a + b 表现相同. 类实现 op= 运算符的方式 也许 是不同的, 列表就是这样做的.

表达式 a = a + [5,6,7,8] 会生成一个新列表, 并让 a 引用这个新列表, 同时保持 b 不变.

表达式 a += [5,6,7,8] 实际上是使用的是 "extend" 函数, 所以 a 和 b 仍然指向已被修改的同一列表.


```python
a_var = 'global variable'

def a_func():
    print(a_var, '[ a_var inside a_func() ]')

a_func()
print(a_var, '[ a_var outside a_func() ]')
```

    global variable [ a_var inside a_func() ]
    global variable [ a_var outside a_func() ]



```python
a_var = 'global value'

def a_func():
    a_var = 'local value'
    print(a_var, '[ a_var inside a_func() ]')

a_func()
print(a_var, '[ a_var outside a_func() ]')
```

    local value [ a_var inside a_func() ]
    global value [ a_var outside a_func() ]



```python
a_var = 'global value'

def a_func():
    global a_var
    a_var = 'local value'
    print(a_var, '[ a_var inside a_func() ]')

print(a_var, '[ a_var outside a_func() ]')
a_func()
print(a_var, '[ a_var outside a_func() ]')
```

    global value [ a_var outside a_func() ]
    local value [ a_var inside a_func() ]
    local value [ a_var outside a_func() ]



```python
a_var = 'global value'

def outer():
    a_var = 'enclosed value'

    def inner():
        a_var = 'local value'
        print(a_var)

    inner()

outer()
```

    local value



```python
a_var = 'global variable'

def len(in_var):
    print('called my len() function')
    l = 0
    for i in in_var:
        l += 1
    return l

def a_func(in_var):
    len_in_var = len(in_var)
    print('Input variable is of length', len_in_var)

a_func('Hello, World!')
```

    called my len() function
    Input variable is of length 13



```python
a = 'global'

def outer():

    def len(in_var):
        print('called my len() function: ', end="")
        l = 0
        for i in in_var:
            l += 1
        return l

    a = 'local'

    def inner():
        global len
        nonlocal a
        a += ' variable'
    inner()
    print('a is', a)
    print(len(a))


outer()

print(len(a))
print('a is', a)
```

    a is local variable
    called my len() function: 14
    called my len() function
    6
    a is global


## 大海捞针


```python
x, y = (0, 1) if True else None, None
```


```python
x,y
```




    ((0, 1), None)




```python
##正确做法
x,y = (0,1) if True else (None,None)
```


```python
x,y
```




    (0, 1)




```python
t = ('one', 'two')
for i in t:
    print(i)

t = ('one')
for i in t:
    print(i)

t = ()
print(t)
```

    one
    two
    o
    n
    e
    ()



```python
##明显上面的把t = ('one') t当成字符串了，正确做法如下
t = ('one',) ##注意逗号
for i in t:
    print(i)
```

    one
