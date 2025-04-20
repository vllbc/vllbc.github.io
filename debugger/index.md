# debugger

python调试工具，类似于vscode的调试工具，使用命令行进行调试。

## 使用方法
### 插入式
```python
import pdb; pdb.set_trace()
```
或者
```python
breakpoint()
```

### 非插入式
```
python -m pdb [-c command] (-m module | pyfile) [args ...]
```

## 常用命令
### h
即help，可用命令如下
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153050.png)

### p 
p x 即print(x)，用于打印变量。
pp x，使用pprint打印
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323154714.png)

### w
即where，查看当前调用栈。 
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153559.png)

### u和d
u即up，回到上一个frame
d即down，到下一个frame
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153615.png)

### l
即lst
l 查看前后12行代码
ll查看当前函数全部代码
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323154652.png)

### b
即break，进行打断点
b x，在第x行打断点。
b 查看所有断点。
相同的有tbreak，与break的区别是第一次到该断点后会自动移除断点。即temporary breakpoint
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153630.png)

### n
即next，执行下一条语句，但忽视函数调用内部细节
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153732.png)

### s
即step，执行下一条语句，如果有函数调用，则调用新frame，进入函数内部。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153716.png)

### c
即continue，继续程序的执行直到下一个断点

![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153834.png)

### r
即return，直接跳转到当前函数return语句
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153811.png)

### until
until n，使程序继续执行直到执行到行数为n的语句。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153755.png)

### cl
即clear
clear n，清除编号为n的断点
clear，清除所有断点。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153646.png)
### j
即jump，向前或向后跳转，与until区别是，jump不会执行中间的语句，而是忽略他们。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323153957.png)

### display
相当于一个监视器，用于监视变量的变化
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250323155137.png)
### retval
打印当前函数最后一次返回的返回值
### q
即quit，退出pdb调试。
