# RFE


## sklearn.feature_selection.RFE
RFE（Recursive feature elimination）：递归特征消除，用来对特征进行重要性评级。主要用于特征选择
RFE阶段
1 初始的特征集为所有可用的特征。
2 使用当前特征集进行建模，然后计算每个特征的重要性。
3 删除最不重要的一个（或多个）特征，更新特征集。
4 跳转到步骤2，直到完成所有特征的重要性评级。

用法如下
```python
model = LogisticRegression()
rfe = RFE(model,8)
rfe = rfe.fit(X,y)
print(f"Selected features {list(X.columns[rfe.support_])}")
```
## sklearn.feature_selection.RFECV
相对RFE加了CV即交叉验证的阶段
CV阶段
1 根据RFE阶段确定的特征重要性，依次选择不同数量的特征。
2 对选定的特征集进行交叉验证。
3 确定平均分最高的特征数量，完成特征选择。
用法如下：
```python
rfecv = RFECV(estimator=LogisticRegression(),step=1,cv=10,scoring='accuracy')
rfecv.fit(X,y)

print("Optimal number of features:",rfecv.n_features_)
print(rfecv.support_)
print("Selecting features:",list(X.columns[rfecv.support_]))
```


