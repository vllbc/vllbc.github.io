# learn_two





```python
import pandas as pd
import numpy as np
```

## 读取文件


```python
df = pd.read_csv(
    'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.info() #DataFrame 的一些总体信息。
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3333 entries, 0 to 3332
    Data columns (total 20 columns):
    State                     3333 non-null object
    Account length            3333 non-null int64
    Area code                 3333 non-null int64
    International plan        3333 non-null object
    Voice mail plan           3333 non-null object
    Number vmail messages     3333 non-null int64
    Total day minutes         3333 non-null float64
    Total day calls           3333 non-null int64
    Total day charge          3333 non-null float64
    Total eve minutes         3333 non-null float64
    Total eve calls           3333 non-null int64
    Total eve charge          3333 non-null float64
    Total night minutes       3333 non-null float64
    Total night calls         3333 non-null int64
    Total night charge        3333 non-null float64
    Total intl minutes        3333 non-null float64
    Total intl calls          3333 non-null int64
    Total intl charge         3333 non-null float64
    Customer service calls    3333 non-null int64
    Churn                     3333 non-null bool
    dtypes: bool(1), float64(8), int64(8), object(3)
    memory usage: 498.1+ KB



```python
df.shape #形状大小
```




    (3333, 20)




```python
df.columns #列名
```




    Index(['State', 'Account length', 'Area code', 'International plan',
           'Voice mail plan', 'Number vmail messages', 'Total day minutes',
           'Total day calls', 'Total day charge', 'Total eve minutes',
           'Total eve calls', 'Total eve charge', 'Total night minutes',
           'Total night calls', 'Total night charge', 'Total intl minutes',
           'Total intl calls', 'Total intl charge', 'Customer service calls',
           'Churn'],
          dtype='object')




```python
df['Churn'] = df['Churn'].astype('int64')  #将Churn列修改数据类型
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3333 entries, 0 to 3332
    Data columns (total 20 columns):
    State                     3333 non-null object
    Account length            3333 non-null int64
    Area code                 3333 non-null int64
    International plan        3333 non-null object
    Voice mail plan           3333 non-null object
    Number vmail messages     3333 non-null int64
    Total day minutes         3333 non-null float64
    Total day calls           3333 non-null int64
    Total day charge          3333 non-null float64
    Total eve minutes         3333 non-null float64
    Total eve calls           3333 non-null int64
    Total eve charge          3333 non-null float64
    Total night minutes       3333 non-null float64
    Total night calls         3333 non-null int64
    Total night charge        3333 non-null float64
    Total intl minutes        3333 non-null float64
    Total intl calls          3333 non-null int64
    Total intl charge         3333 non-null float64
    Customer service calls    3333 non-null int64
    Churn                     3333 non-null int64
    dtypes: float64(8), int64(9), object(3)
    memory usage: 520.9+ KB



```python
df.describe() #显示数值特征（int64 和 float64）的基本统计学特性
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Account length</th>
      <th>Area code</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
      <td>3333.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>101.064806</td>
      <td>437.182418</td>
      <td>8.099010</td>
      <td>179.775098</td>
      <td>100.435644</td>
      <td>30.562307</td>
      <td>200.980348</td>
      <td>100.114311</td>
      <td>17.083540</td>
      <td>200.872037</td>
      <td>100.107711</td>
      <td>9.039325</td>
      <td>10.237294</td>
      <td>4.479448</td>
      <td>2.764581</td>
      <td>1.562856</td>
      <td>0.144914</td>
    </tr>
    <tr>
      <th>std</th>
      <td>39.822106</td>
      <td>42.371290</td>
      <td>13.688365</td>
      <td>54.467389</td>
      <td>20.069084</td>
      <td>9.259435</td>
      <td>50.713844</td>
      <td>19.922625</td>
      <td>4.310668</td>
      <td>50.573847</td>
      <td>19.568609</td>
      <td>2.275873</td>
      <td>2.791840</td>
      <td>2.461214</td>
      <td>0.753773</td>
      <td>1.315491</td>
      <td>0.352067</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>408.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>23.200000</td>
      <td>33.000000</td>
      <td>1.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>74.000000</td>
      <td>408.000000</td>
      <td>0.000000</td>
      <td>143.700000</td>
      <td>87.000000</td>
      <td>24.430000</td>
      <td>166.600000</td>
      <td>87.000000</td>
      <td>14.160000</td>
      <td>167.000000</td>
      <td>87.000000</td>
      <td>7.520000</td>
      <td>8.500000</td>
      <td>3.000000</td>
      <td>2.300000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>101.000000</td>
      <td>415.000000</td>
      <td>0.000000</td>
      <td>179.400000</td>
      <td>101.000000</td>
      <td>30.500000</td>
      <td>201.400000</td>
      <td>100.000000</td>
      <td>17.120000</td>
      <td>201.200000</td>
      <td>100.000000</td>
      <td>9.050000</td>
      <td>10.300000</td>
      <td>4.000000</td>
      <td>2.780000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>127.000000</td>
      <td>510.000000</td>
      <td>20.000000</td>
      <td>216.400000</td>
      <td>114.000000</td>
      <td>36.790000</td>
      <td>235.300000</td>
      <td>114.000000</td>
      <td>20.000000</td>
      <td>235.300000</td>
      <td>113.000000</td>
      <td>10.590000</td>
      <td>12.100000</td>
      <td>6.000000</td>
      <td>3.270000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>243.000000</td>
      <td>510.000000</td>
      <td>51.000000</td>
      <td>350.800000</td>
      <td>165.000000</td>
      <td>59.640000</td>
      <td>363.700000</td>
      <td>170.000000</td>
      <td>30.910000</td>
      <td>395.000000</td>
      <td>175.000000</td>
      <td>17.770000</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>5.400000</td>
      <td>9.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.describe(include=['object', 'bool']) #通过 include 参数显式指定包含的数据类型，可以查看非数值特征的统计数据
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3333</td>
      <td>3333</td>
      <td>3333</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>51</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>WV</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>106</td>
      <td>3010</td>
      <td>2411</td>
    </tr>
  </tbody>
</table>

</div>




```python
df['Churn'].value_counts() #如其名
```




    0    2850
    1     483
    Name: Churn, dtype: int64




```python
df['Churn'].value_counts(normalize=True) #传入参数显示比例
```




    0    0.855086
    1    0.144914
    Name: Churn, dtype: float64




```python
df.sort_values(by='Total day charge', ascending=False).head() #根据Total day charge列进行排序 ascending=False为倒序排序
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>365</th>
      <td>CO</td>
      <td>154</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>350.8</td>
      <td>75</td>
      <td>59.64</td>
      <td>216.5</td>
      <td>94</td>
      <td>18.40</td>
      <td>253.9</td>
      <td>100</td>
      <td>11.43</td>
      <td>10.1</td>
      <td>9</td>
      <td>2.73</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>985</th>
      <td>NY</td>
      <td>64</td>
      <td>415</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>346.8</td>
      <td>55</td>
      <td>58.96</td>
      <td>249.5</td>
      <td>79</td>
      <td>21.21</td>
      <td>275.4</td>
      <td>102</td>
      <td>12.39</td>
      <td>13.3</td>
      <td>9</td>
      <td>3.59</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2594</th>
      <td>OH</td>
      <td>115</td>
      <td>510</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
      <td>345.3</td>
      <td>81</td>
      <td>58.70</td>
      <td>203.4</td>
      <td>106</td>
      <td>17.29</td>
      <td>217.5</td>
      <td>107</td>
      <td>9.79</td>
      <td>11.8</td>
      <td>8</td>
      <td>3.19</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>156</th>
      <td>OH</td>
      <td>83</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>337.4</td>
      <td>120</td>
      <td>57.36</td>
      <td>227.4</td>
      <td>116</td>
      <td>19.33</td>
      <td>153.9</td>
      <td>114</td>
      <td>6.93</td>
      <td>15.8</td>
      <td>7</td>
      <td>4.27</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>605</th>
      <td>MO</td>
      <td>112</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>335.5</td>
      <td>77</td>
      <td>57.04</td>
      <td>212.5</td>
      <td>109</td>
      <td>18.06</td>
      <td>265.0</td>
      <td>132</td>
      <td>11.93</td>
      <td>12.7</td>
      <td>8</td>
      <td>3.43</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.sort_values(by=['Churn', 'Total day charge'],
               ascending=[True, False]).head() #先按 Churn 离网率 升序排列，再按 Total day charge 每日总话费 降序排列
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>688</th>
      <td>MN</td>
      <td>13</td>
      <td>510</td>
      <td>No</td>
      <td>Yes</td>
      <td>21</td>
      <td>315.6</td>
      <td>105</td>
      <td>53.65</td>
      <td>208.9</td>
      <td>71</td>
      <td>17.76</td>
      <td>260.1</td>
      <td>123</td>
      <td>11.70</td>
      <td>12.1</td>
      <td>3</td>
      <td>3.27</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2259</th>
      <td>NC</td>
      <td>210</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>31</td>
      <td>313.8</td>
      <td>87</td>
      <td>53.35</td>
      <td>147.7</td>
      <td>103</td>
      <td>12.55</td>
      <td>192.7</td>
      <td>97</td>
      <td>8.67</td>
      <td>10.1</td>
      <td>7</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>534</th>
      <td>LA</td>
      <td>67</td>
      <td>510</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>310.4</td>
      <td>97</td>
      <td>52.77</td>
      <td>66.5</td>
      <td>123</td>
      <td>5.65</td>
      <td>246.5</td>
      <td>99</td>
      <td>11.09</td>
      <td>9.2</td>
      <td>10</td>
      <td>2.48</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>575</th>
      <td>SD</td>
      <td>114</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>36</td>
      <td>309.9</td>
      <td>90</td>
      <td>52.68</td>
      <td>200.3</td>
      <td>89</td>
      <td>17.03</td>
      <td>183.5</td>
      <td>105</td>
      <td>8.26</td>
      <td>14.2</td>
      <td>2</td>
      <td>3.83</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>AL</td>
      <td>141</td>
      <td>510</td>
      <td>No</td>
      <td>Yes</td>
      <td>28</td>
      <td>308.0</td>
      <td>123</td>
      <td>52.36</td>
      <td>247.8</td>
      <td>128</td>
      <td>21.06</td>
      <td>152.9</td>
      <td>103</td>
      <td>6.88</td>
      <td>7.4</td>
      <td>3</td>
      <td>2.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
df[df['Churn'] == 1].mean()
```




    Account length            102.664596
    Area code                 437.817805
    Number vmail messages       5.115942
    Total day minutes         206.914079
    Total day calls           101.335404
    Total day charge           35.175921
    Total eve minutes         212.410145
    Total eve calls           100.561077
    Total eve charge           18.054969
    Total night minutes       205.231677
    Total night calls         100.399586
    Total night charge          9.235528
    Total intl minutes         10.700000
    Total intl calls            4.163561
    Total intl charge           2.889545
    Customer service calls      2.229814
    Churn                       1.000000
    dtype: float64




```python
df[df['Churn'] == 1]['Total day minutes'].mean()
```




    206.91407867494814




```python
df.loc[0:5, 'State':'Area code'] #通过标签来选取
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AL</td>
      <td>118</td>
      <td>510</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.iloc[0:5, 0:3] #通过索引来选取，类似于python的切片操作
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.apply(max) #应用到每一列
```




    State                        WY
    Account length              243
    Area code                   510
    International plan          Yes
    Voice mail plan             Yes
    Number vmail messages        51
    Total day minutes         350.8
    Total day calls             165
    Total day charge          59.64
    Total eve minutes         363.7
    Total eve calls             170
    Total eve charge          30.91
    Total night minutes         395
    Total night calls           175
    Total night charge        17.77
    Total intl minutes           20
    Total intl calls             20
    Total intl charge           5.4
    Customer service calls        9
    Churn                         1
    dtype: object




```python
df[df['State'].apply(lambda state: state[0] == 'W')].head() #获取首字母为W的州 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>WV</td>
      <td>141</td>
      <td>415</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>37</td>
      <td>258.6</td>
      <td>84</td>
      <td>43.96</td>
      <td>222.0</td>
      <td>111</td>
      <td>18.87</td>
      <td>326.4</td>
      <td>97</td>
      <td>14.69</td>
      <td>11.2</td>
      <td>5</td>
      <td>3.02</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>WY</td>
      <td>57</td>
      <td>408</td>
      <td>No</td>
      <td>Yes</td>
      <td>39</td>
      <td>213.0</td>
      <td>115</td>
      <td>36.21</td>
      <td>191.1</td>
      <td>112</td>
      <td>16.24</td>
      <td>182.7</td>
      <td>115</td>
      <td>8.22</td>
      <td>9.5</td>
      <td>3</td>
      <td>2.57</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>WI</td>
      <td>64</td>
      <td>510</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>154.0</td>
      <td>67</td>
      <td>26.18</td>
      <td>225.8</td>
      <td>118</td>
      <td>19.19</td>
      <td>265.3</td>
      <td>86</td>
      <td>11.94</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.95</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>WY</td>
      <td>97</td>
      <td>415</td>
      <td>No</td>
      <td>Yes</td>
      <td>24</td>
      <td>133.2</td>
      <td>135</td>
      <td>22.64</td>
      <td>217.2</td>
      <td>58</td>
      <td>18.46</td>
      <td>70.6</td>
      <td>79</td>
      <td>3.18</td>
      <td>11.0</td>
      <td>3</td>
      <td>2.97</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>WY</td>
      <td>87</td>
      <td>415</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
      <td>151.0</td>
      <td>83</td>
      <td>25.67</td>
      <td>219.7</td>
      <td>116</td>
      <td>18.67</td>
      <td>203.9</td>
      <td>127</td>
      <td>9.18</td>
      <td>9.7</td>
      <td>3</td>
      <td>2.62</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
d = {'No': False, 'Yes': True}
df['International plan'] = df['International plan'].map(d) #将No转换为False Yes转换为True
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>False</td>
      <td>Yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>False</td>
      <td>Yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>False</td>
      <td>No</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>True</td>
      <td>No</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>True</td>
      <td>No</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
df = df.replace({'Voice mail plan': d}) #用replace也可以达到相同的目的
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>Total eve calls</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>



## 汇总表


```python
pd.crosstab(df['Churn'], df['International plan'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>International plan</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>Churn</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2664</td>
      <td>186</td>
    </tr>
    <tr>
      <th>1</th>
      <td>346</td>
      <td>137</td>
    </tr>
  </tbody>
</table>

</div>




```python
pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Voice mail plan</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>Churn</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.602460</td>
      <td>0.252625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.120912</td>
      <td>0.024002</td>
    </tr>
  </tbody>
</table>

</div>




```python
total_calls = df['Total day calls'] + df['Total eve calls'] + \
    df['Total night calls'] + df['Total intl calls']
# loc 参数是插入 Series 对象后选择的列数
# 设置为 len(df.columns)以便将计算后的 Total calls 粘贴到最后一列
df.insert(loc=len(df.columns), column='Total calls', value=total_calls)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>...</th>
      <th>Total eve charge</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
      <th>Total calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>...</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
      <td>303</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>...</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
      <td>332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>...</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
      <td>333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>...</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
      <td>255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>...</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
      <td>359</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>

</div>




```python
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + \
    df['Total night charge'] + df['Total intl charge'] #不创造实例的情况下直接插入
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Account length</th>
      <th>Area code</th>
      <th>International plan</th>
      <th>Voice mail plan</th>
      <th>Number vmail messages</th>
      <th>Total day minutes</th>
      <th>Total day calls</th>
      <th>Total day charge</th>
      <th>Total eve minutes</th>
      <th>...</th>
      <th>Total night minutes</th>
      <th>Total night calls</th>
      <th>Total night charge</th>
      <th>Total intl minutes</th>
      <th>Total intl calls</th>
      <th>Total intl charge</th>
      <th>Customer service calls</th>
      <th>Churn</th>
      <th>Total calls</th>
      <th>Total charge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>...</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
      <td>303</td>
      <td>75.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>False</td>
      <td>True</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>...</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
      <td>332</td>
      <td>59.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>...</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
      <td>333</td>
      <td>62.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>...</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
      <td>255</td>
      <td>66.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>...</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
      <td>359</td>
      <td>52.09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>

</div>
