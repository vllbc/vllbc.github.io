# apply_along_axis



类似于pandas的apply，就是在某一维上进行定义的函数操作

````python3
apply_along_axis(func1d, axis, arr, *args, **kwargs)
````

官网的例子

```python
def my_func(a):
    return (a[0] + a[-1]) * 0.5
b = np.array([[1,2,3], [4,5,6], [7,8,9]])

np.apply_along_axis(my_func, 0, b)
# 结果
array([ 4.,  5.,  6.])

# 结果

array([ 2.,  5.,  8.])

```
