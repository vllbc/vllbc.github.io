# parse_shape

>Parse a tensor shape to dictionary mapping axes names to their lengths.

```python
# Use underscore to skip the dimension in parsing. 
>>> x = np.zeros([2, 3, 5, 7]) >>> parse_shape(x, 'batch _ h w') {'batch': 2, 'h': 5, 'w': 7} 
# `parse_shape` output can be used to specify axes_lengths for other operations: 
>>> y = np.zeros([700]) 
>>> rearrange(y, '(b c h w) -> b c h w', **parse_shape(x, 'b _ h w')).shape 
(2, 10, 5, 7)
```
也就是把维度的维数映射到对应的命名。与数据无关，只看得到维度。

