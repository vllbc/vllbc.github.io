# 并查集

```python
def find(x):
	if (p[x] != x):
		p[x] = find(p[x])
	return p[x]

```
上面是y总的模板，实现了路径压缩。

