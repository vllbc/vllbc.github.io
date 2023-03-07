# flip



矩阵的反转，可以按照各个维度很好理解。
例子：
```python3
cs_matrix = np.array([[ 4,  3,  2,  1,  0], [ 8,  7,  6,  5,  1], [11, 10,  9,  6,  2], [13, 12, 10,  7,  3], [14, 13, 11,  8,  4]])
np.flip(cs_matrix, 0)
```

![](image/Pasted%20image%2020221108231234.png)
变成了：
![](image/Pasted%20image%2020221108231249.png)

```python3
np.flip(cs_matrix, 1)
```
变成了：
![](image/Pasted%20image%2020221108231329.png)
