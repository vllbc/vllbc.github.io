# SelectFromModel



# Sklearn.feature_selection.SelectFromModel

```python
class sklearn.feature_selection.SelectFromModel(estimator, 
		*, threshold=None, prefit=False, norm_order=1, 
		max_features=None)[source]
```

参数

```PYTHON
Parameters
-
estimator_：一个估算器
		    用来建立变压器的基本估计器。
			只有当一个不适合的估计器传递给SelectFromModel时，
			才会存储这个值，即当prefit为False时。

threshold_：float
			用于特征选择的阈值。

```

方法:

```python
'fit(self, X[, y])'
	训练SelectFromModel元变压器。

'fit_transform(self, X[, y])'
	训练元变压器，然后对X进行转换。

'get_params(self[, deep])'
	获取此估计量的参数。
	
'get_support(self[, indices])'
	获取所选特征的掩码或整数索引

'inverse_transform(self, X)'
	反向转换操作
	
'partial_fit(self, X[, y])'
	仅将SelectFromModel元变压器训练一次。

'set_params(self, \*\*params)'
	设置此估算器的参数。

'transform(self, X)'
	将X缩小为选定的特征。

```
