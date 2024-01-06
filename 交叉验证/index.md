# 交叉验证





​	

```python
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
```


```python
data = datasets.load_iris()
X = data.data
Y = data.target
```


```python
k_scores = []
for k in range(1,31):
    model = KNeighborsClassifier(n_neighbors=k)
    #scores = cross_val_score(model,X,Y,cv=10,scoring="accuracy") # for classification
    loss = -cross_val_score(model,X,Y,cv=10,scoring="neg_mean_squared_error") # for regression
    k_scores.append(loss.mean())
plt.plot(range(1,31),k_scores)
plt.show()
```

![](https://cdn.jsdelivr.net/gh/vllbc/img4blog//image/output1.png)
