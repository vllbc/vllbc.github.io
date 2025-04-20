# LEGB

```python
a = 'global'

def outer():

    # def len(in_var):
    #     print('called my len() function: ', end="")
    #     l = 0
    #     for i in in_var:
    #         l += 1
    #     return l

    a = 'local'

    def inner():
        nonlocal a
        a += ' variable'
    inner()
    print('a is', a)
    # print(len(a))


outer()

# print(len(a))
print('a is', a)
```
此时为nonlocal a，会按照local-闭包-global的顺序找到闭包变量a。a的值为local variable

```python
a = 'global'

def outer():

    # def len(in_var):
    #     print('called my len() function: ', end="")
    #     l = 0
    #     for i in in_var:
    #         l += 1
    #     return l

    a = 'local'

    def inner():
        global a
        a += ' variable'
    inner()
    print('a is', a)
    # print(len(a))


outer()

# print(len(a))
print('a is', a)
```
此时为global，会从全局变量中寻找a，a的值为global variable
