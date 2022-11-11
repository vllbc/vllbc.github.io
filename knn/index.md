# KNN


## 导入包


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
```

## 导入数据


```python
data = pd.read_csv("./datasets/Social_Network_Ads.csv")
X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values
# scatter = go.Scatter(x=X[:,0],y=X[:,1],mode='markers',marker={'color':Y})
# fig = go.Figure(scatter)
# fig.show()
```


```python
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
```

## 标准化


```python
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
X_train = sca.fit_transform(X_train)
X_test = sca.transform(X_test)
```

## 训练模型


```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5,p=2)
model.fit(X_train,Y_train)
```




    KNeighborsClassifier()



## 模型得分


```python
model.score(X_test,Y_test)
```




    0.93




