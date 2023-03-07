# task2





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>



## 数据集有多少男性和女性？


```python
data['sex'].value_counts()
```




    Male      21790
    Female    10771
    Name: sex, dtype: int64



## 数据集女性的平均年龄


```python
data[data['sex'] == 'Female']['age'].mean()
```




    36.85823043357163



## 数据集中德国公民的比例是多少？


```python
data['native-country'].value_counts(normalize=True)['Germany']
```




    0.004207487485028101



## 年收入超过 50K 和低于 50K 人群年龄的平均值和标准差是多少？


```python
salary1 = data[data['salary'] == '>50K']['age']
salary2 = data[data['salary'] == '<=50K']['age']
print(salary1.mean(),salary1.std())
print(salary2.mean(),salary2.std())
```

    44.24984058155847 10.51902771985177
    36.78373786407767 14.020088490824813


## 年收入超过 50K 的人群是否都接受过高中以上教育？


```python
data[data['salary'] == ">50K"]['education'].unique()
```




    array(['HS-grad', 'Masters', 'Bachelors', 'Some-college', 'Assoc-voc',
           'Doctorate', 'Prof-school', 'Assoc-acdm', '7th-8th', '12th',
           '10th', '11th', '9th', '5th-6th', '1st-4th'], dtype=object)



## 使用 groupby 和 describe 统计不同种族和性别人群的年龄分布数据。


```python
for (race,sex),mini_data in data.groupby(['race','sex']):
    print(race,sex)
    print(mini_data['age'].describe())
```

    Amer-Indian-Eskimo Female
    count    119.000000
    mean      37.117647
    std       13.114991
    min       17.000000
    25%       27.000000
    50%       36.000000
    75%       46.000000
    max       80.000000
    Name: age, dtype: float64
    Amer-Indian-Eskimo Male
    count    192.000000
    mean      37.208333
    std       12.049563
    min       17.000000
    25%       28.000000
    50%       35.000000
    75%       45.000000
    max       82.000000
    Name: age, dtype: float64
    Asian-Pac-Islander Female
    count    346.000000
    mean      35.089595
    std       12.300845
    min       17.000000
    25%       25.000000
    50%       33.000000
    75%       43.750000
    max       75.000000
    Name: age, dtype: float64
    Asian-Pac-Islander Male
    count    693.000000
    mean      39.073593
    std       12.883944
    min       18.000000
    25%       29.000000
    50%       37.000000
    75%       46.000000
    max       90.000000
    Name: age, dtype: float64
    Black Female
    count    1555.000000
    mean       37.854019
    std        12.637197
    min        17.000000
    25%        28.000000
    50%        37.000000
    75%        46.000000
    max        90.000000
    Name: age, dtype: float64
    Black Male
    count    1569.000000
    mean       37.682600
    std        12.882612
    min        17.000000
    25%        27.000000
    50%        36.000000
    75%        46.000000
    max        90.000000
    Name: age, dtype: float64
    Other Female
    count    109.000000
    mean      31.678899
    std       11.631599
    min       17.000000
    25%       23.000000
    50%       29.000000
    75%       39.000000
    max       74.000000
    Name: age, dtype: float64
    Other Male
    count    162.000000
    mean      34.654321
    std       11.355531
    min       17.000000
    25%       26.000000
    50%       32.000000
    75%       42.000000
    max       77.000000
    Name: age, dtype: float64
    White Female
    count    8642.000000
    mean       36.811618
    std        14.329093
    min        17.000000
    25%        25.000000
    50%        35.000000
    75%        46.000000
    max        90.000000
    Name: age, dtype: float64
    White Male
    count    19174.000000
    mean        39.652498
    std         13.436029
    min         17.000000
    25%         29.000000
    50%         38.000000
    75%         49.000000
    max         90.000000
    Name: age, dtype: float64


## 统计男性高收入人群中已婚和未婚（包含离婚和分居）人群各自所占数量。


```python
# 未婚
data[(data['sex'] == 'Male') &
     (data['marital-status'].isin(['Never-married',
                                   'Separated', 'Divorced']))]['salary'].value_counts()
```




    <=50K    7423
    >50K      658
    Name: salary, dtype: int64




```python
# 已婚
data[(data['sex'] == 'Male') &
     (data['marital-status'].str.startswith('Married'))]['salary'].value_counts()
```




    <=50K    7576
    >50K     5965
    Name: salary, dtype: int64



## 计算各国超过和低于 50K 人群各自的平均周工作时长。


```python
for (country, salary), sub_df in data.groupby(['native-country', 'salary']):
    print(country, salary, round(sub_df['hours-per-week'].mean(), 2))
```

    ? <=50K 40.16
    ? >50K 45.55
    Cambodia <=50K 41.42
    Cambodia >50K 40.0
    Canada <=50K 37.91
    Canada >50K 45.64
    China <=50K 37.38
    China >50K 38.9
    Columbia <=50K 38.68
    Columbia >50K 50.0
    Cuba <=50K 37.99
    Cuba >50K 42.44
    Dominican-Republic <=50K 42.34
    Dominican-Republic >50K 47.0
    Ecuador <=50K 38.04
    Ecuador >50K 48.75
    El-Salvador <=50K 36.03
    El-Salvador >50K 45.0
    England <=50K 40.48
    England >50K 44.53
    France <=50K 41.06
    France >50K 50.75
    Germany <=50K 39.14
    Germany >50K 44.98
    Greece <=50K 41.81
    Greece >50K 50.62
    Guatemala <=50K 39.36
    Guatemala >50K 36.67
    Haiti <=50K 36.33
    Haiti >50K 42.75
    Holand-Netherlands <=50K 40.0
    Honduras <=50K 34.33
    Honduras >50K 60.0
    Hong <=50K 39.14
    Hong >50K 45.0
    Hungary <=50K 31.3
    Hungary >50K 50.0
    India <=50K 38.23
    India >50K 46.48
    Iran <=50K 41.44
    Iran >50K 47.5
    Ireland <=50K 40.95
    Ireland >50K 48.0
    Italy <=50K 39.62
    Italy >50K 45.4
    Jamaica <=50K 38.24
    Jamaica >50K 41.1
    Japan <=50K 41.0
    Japan >50K 47.96
    Laos <=50K 40.38
    Laos >50K 40.0
    Mexico <=50K 40.0
    Mexico >50K 46.58
    Nicaragua <=50K 36.09
    Nicaragua >50K 37.5
    Outlying-US(Guam-USVI-etc) <=50K 41.86
    Peru <=50K 35.07
    Peru >50K 40.0
    Philippines <=50K 38.07
    Philippines >50K 43.03
    Poland <=50K 38.17
    Poland >50K 39.0
    Portugal <=50K 41.94
    Portugal >50K 41.5
    Puerto-Rico <=50K 38.47
    Puerto-Rico >50K 39.42
    Scotland <=50K 39.44
    Scotland >50K 46.67
    South <=50K 40.16
    South >50K 51.44
    Taiwan <=50K 33.77
    Taiwan >50K 46.8
    Thailand <=50K 42.87
    Thailand >50K 58.33
    Trinadad&Tobago <=50K 37.06
    Trinadad&Tobago >50K 40.0
    United-States <=50K 38.8
    United-States >50K 45.51
    Vietnam <=50K 37.19
    Vietnam >50K 39.2
    Yugoslavia <=50K 41.6
    Yugoslavia >50K 49.5



```python
# 交叉表
pd.crosstab(data['native-country'], data['salary'],
            values=data['hours-per-week'], aggfunc=np.mean)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>salary</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
    </tr>
    <tr>
      <th>native-country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>?</th>
      <td>40.164760</td>
      <td>45.547945</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>41.416667</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>37.914634</td>
      <td>45.641026</td>
    </tr>
    <tr>
      <th>China</th>
      <td>37.381818</td>
      <td>38.900000</td>
    </tr>
    <tr>
      <th>Columbia</th>
      <td>38.684211</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>Cuba</th>
      <td>37.985714</td>
      <td>42.440000</td>
    </tr>
    <tr>
      <th>Dominican-Republic</th>
      <td>42.338235</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>Ecuador</th>
      <td>38.041667</td>
      <td>48.750000</td>
    </tr>
    <tr>
      <th>El-Salvador</th>
      <td>36.030928</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>England</th>
      <td>40.483333</td>
      <td>44.533333</td>
    </tr>
    <tr>
      <th>France</th>
      <td>41.058824</td>
      <td>50.750000</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>39.139785</td>
      <td>44.977273</td>
    </tr>
    <tr>
      <th>Greece</th>
      <td>41.809524</td>
      <td>50.625000</td>
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>39.360656</td>
      <td>36.666667</td>
    </tr>
    <tr>
      <th>Haiti</th>
      <td>36.325000</td>
      <td>42.750000</td>
    </tr>
    <tr>
      <th>Holand-Netherlands</th>
      <td>40.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Honduras</th>
      <td>34.333333</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>Hong</th>
      <td>39.142857</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>31.300000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>India</th>
      <td>38.233333</td>
      <td>46.475000</td>
    </tr>
    <tr>
      <th>Iran</th>
      <td>41.440000</td>
      <td>47.500000</td>
    </tr>
    <tr>
      <th>Ireland</th>
      <td>40.947368</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>39.625000</td>
      <td>45.400000</td>
    </tr>
    <tr>
      <th>Jamaica</th>
      <td>38.239437</td>
      <td>41.100000</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>41.000000</td>
      <td>47.958333</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>40.375000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>40.003279</td>
      <td>46.575758</td>
    </tr>
    <tr>
      <th>Nicaragua</th>
      <td>36.093750</td>
      <td>37.500000</td>
    </tr>
    <tr>
      <th>Outlying-US(Guam-USVI-etc)</th>
      <td>41.857143</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>35.068966</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>Philippines</th>
      <td>38.065693</td>
      <td>43.032787</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>38.166667</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>41.939394</td>
      <td>41.500000</td>
    </tr>
    <tr>
      <th>Puerto-Rico</th>
      <td>38.470588</td>
      <td>39.416667</td>
    </tr>
    <tr>
      <th>Scotland</th>
      <td>39.444444</td>
      <td>46.666667</td>
    </tr>
    <tr>
      <th>South</th>
      <td>40.156250</td>
      <td>51.437500</td>
    </tr>
    <tr>
      <th>Taiwan</th>
      <td>33.774194</td>
      <td>46.800000</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>42.866667</td>
      <td>58.333333</td>
    </tr>
    <tr>
      <th>Trinadad&amp;Tobago</th>
      <td>37.058824</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>United-States</th>
      <td>38.799127</td>
      <td>45.505369</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>37.193548</td>
      <td>39.200000</td>
    </tr>
    <tr>
      <th>Yugoslavia</th>
      <td>41.600000</td>
      <td>49.500000</td>
    </tr>
  </tbody>
</table>
