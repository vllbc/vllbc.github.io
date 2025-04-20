# repeat

>einops.repeat allows reordering elements and repeating them in arbitrary combinations. This operation includes functionality of repeat, tile, and broadcast functions.

repeat是使维度增加，与reduce相反。
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250111210915.png)
![image.png](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/20250111210926.png)
## 应用
比如说repeat_kv函数就可以用einops.repeat很方便的实现
```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```

等价于
```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    einops.repeat(x, 'bs slen kvheads dim-> bs slen (kvheads n_rep) dim', n_rep=n_rep).shape
```
