# rot90


正为逆时针转，负为顺时针转。
```python3
import numpy as np
mat = np.array([[1,3,5],
                [2,4,6],
                [7,8,9]
                ])
print mat, "# orignal"
mat90 = np.rot90(mat, 1)
print mat90, "# rorate 90 <left> anti-clockwise"
mat90 = np.rot90(mat, -1)
print mat90, "# rorate 90 <right> clockwise"
mat180 = np.rot90(mat, 2)
print mat180, "# rorate 180 <left> anti-clockwise"
mat270 = np.rot90(mat, 3)
print mat270, "# rorate 270 <left> anti-clockwise"

```


直接复制的代码，python2，能看懂就行。


