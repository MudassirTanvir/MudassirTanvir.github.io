---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.3
  nbformat: 4
  nbformat_minor: 5
---

::: {#fce3938a .cell .code execution_count="1"}
``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

::: {.output .stream .stderr}
    C:\Users\Mudassir\AppData\Roaming\Python\Python37\site-packages\pandas\compat\_optional.py:138: UserWarning: Pandas requires version '2.7.0' or newer of 'numexpr' (version '2.6.9' currently installed).
      warnings.warn(msg, UserWarning)
:::
:::

::: {#61b06215 .cell .code execution_count="2"}
``` python
df = pd.read_csv('insurance.csv')
```
:::

::: {#92a12c67 .cell .code execution_count="3"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB
:::
:::

::: {#254eafd2 .cell .code execution_count="4"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="4"}
```{=html}
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#d1882b3b .cell .code execution_count="5"}
``` python
df.head()
```

::: {.output .execute_result execution_count="5"}
```{=html}
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#4f61b5c8 .cell .code execution_count="6"}
``` python
df.nunique()
```

::: {.output .execute_result execution_count="6"}
    age           47
    sex            2
    bmi          548
    children       6
    smoker         2
    region         4
    charges     1337
    dtype: int64
:::
:::

::: {#e8ca555e .cell .code execution_count="7"}
``` python
df.corr()
```

::: {.output .execute_result execution_count="7"}
```{=html}
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.109272</td>
      <td>0.042469</td>
      <td>0.299008</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.109272</td>
      <td>1.000000</td>
      <td>0.012759</td>
      <td>0.198341</td>
    </tr>
    <tr>
      <th>children</th>
      <td>0.042469</td>
      <td>0.012759</td>
      <td>1.000000</td>
      <td>0.067998</td>
    </tr>
    <tr>
      <th>charges</th>
      <td>0.299008</td>
      <td>0.198341</td>
      <td>0.067998</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#866e37dd .cell .code execution_count="8"}
``` python
sns.heatmap(df.corr(),annot=True)
```

::: {.output .execute_result execution_count="8"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/35697defe859e8bf4420a487c9aad3d616ae9840.png)
:::
:::

::: {#1806b69b .cell .code execution_count="9"}
``` python
plt.pie(df['sex'].value_counts(),autopct='%1.2f%%',labels=['Male','Female'])
plt.title('Gender Distribution')
```

::: {.output .execute_result execution_count="9"}
    Text(0.5, 1.0, 'Gender Distribution')
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/3064d629107455c6db00b3aaf2fb60b6833ee91b.png)
:::
:::

::: {#8b122993 .cell .code execution_count="10"}
``` python
plt.pie(df['smoker'].value_counts(),autopct='%1.2f%%',labels=['no','yes'])
plt.title('Smokers')
```

::: {.output .execute_result execution_count="10"}
    Text(0.5, 1.0, 'Smokers')
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/0ce916a4cd655e2347feee1ed331950134645df6.png)
:::
:::

::: {#0dd9ae34 .cell .code execution_count="11"}
``` python
sns.pairplot(df,hue='smoker')
```

::: {.output .execute_result execution_count="11"}
    <seaborn.axisgrid.PairGrid at 0x2f0c32082e8>
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/22cfbec245408b11381c62c3da452d7aafed72e2.png)
:::
:::

::: {#1249b96d .cell .code execution_count="12"}
``` python
sns.pairplot(df,hue='smoker',kind='scatter',y_vars='charges')
```

::: {.output .execute_result execution_count="12"}
    <seaborn.axisgrid.PairGrid at 0x2f0c5c88438>
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/e2749204289f6d3f93b2328cc0b1aadc7a965020.png)
:::
:::

::: {#19ae10b2 .cell .code execution_count="13"}
``` python
sns.scatterplot(x='bmi',y='charges',data=df,hue='smoker')
```

::: {.output .execute_result execution_count="13"}
    <AxesSubplot:xlabel='bmi', ylabel='charges'>
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/e644fba9e2674836fa81589da4b9218b9047e7ac.png)
:::
:::

::: {#ed5f07d1 .cell .code execution_count="14"}
``` python
sns.scatterplot(x='age',y='charges',data=df,hue='smoker')
```

::: {.output .execute_result execution_count="14"}
    <AxesSubplot:xlabel='age', ylabel='charges'>
:::

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/7e4daf831b4601919d0f605f1ce060bcb26eab95.png)
:::
:::

::: {#4fc2b8c8 .cell .code execution_count="15"}
``` python
df['sex'] = df['sex'].map(lambda x :1 if x=='female' else 0)
df['smoker'] = df['smoker'].map(lambda x :1 if x=='yes' else 0)
```
:::

::: {#3ae2bc34 .cell .code execution_count="16"}
``` python
df.rename(columns = {'sex':'if_female', 'smoker':'if_smoker'}, inplace = True)
```
:::

::: {#ca925699 .cell .code execution_count="17"}
``` python
df.head()
```

::: {.output .execute_result execution_count="17"}
```{=html}
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
      <th>age</th>
      <th>if_female</th>
      <th>bmi</th>
      <th>children</th>
      <th>if_smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
      <td>27.900</td>
      <td>0</td>
      <td>1</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>0</td>
      <td>33.770</td>
      <td>1</td>
      <td>0</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0</td>
      <td>33.000</td>
      <td>3</td>
      <td>0</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>22.705</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>0</td>
      <td>28.880</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#e3f683a8 .cell .code execution_count="18"}
``` python
reg_dummies = pd.get_dummies(df['region'],drop_first=True)
reg_dummies
```

::: {.output .execute_result execution_count="18"}
```{=html}
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
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 3 columns</p>
</div>
```
:::
:::

::: {#b9b26e9d .cell .code execution_count="19"}
``` python
df = pd.concat([df,reg_dummies], axis=1)
```
:::

::: {#a8383599 .cell .code execution_count="20"}
``` python
df.head()
```

::: {.output .execute_result execution_count="20"}
```{=html}
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
      <th>age</th>
      <th>if_female</th>
      <th>bmi</th>
      <th>children</th>
      <th>if_smoker</th>
      <th>region</th>
      <th>charges</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
      <td>27.900</td>
      <td>0</td>
      <td>1</td>
      <td>southwest</td>
      <td>16884.92400</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>0</td>
      <td>33.770</td>
      <td>1</td>
      <td>0</td>
      <td>southeast</td>
      <td>1725.55230</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0</td>
      <td>33.000</td>
      <td>3</td>
      <td>0</td>
      <td>southeast</td>
      <td>4449.46200</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>22.705</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>21984.47061</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>0</td>
      <td>28.880</td>
      <td>0</td>
      <td>0</td>
      <td>northwest</td>
      <td>3866.85520</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#9c310449 .cell .code execution_count="21"}
``` python
df.drop(['region'],axis=1,inplace=True)
df.drop(['children'],axis=1,inplace=True)
```
:::

::: {#081f1caf .cell .code execution_count="22"}
``` python
df.head()
```

::: {.output .execute_result execution_count="22"}
```{=html}
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
      <th>age</th>
      <th>if_female</th>
      <th>bmi</th>
      <th>if_smoker</th>
      <th>charges</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
      <td>27.900</td>
      <td>1</td>
      <td>16884.92400</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>0</td>
      <td>33.770</td>
      <td>0</td>
      <td>1725.55230</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0</td>
      <td>33.000</td>
      <td>0</td>
      <td>4449.46200</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>22.705</td>
      <td>0</td>
      <td>21984.47061</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>0</td>
      <td>28.880</td>
      <td>0</td>
      <td>3866.85520</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#3f8f0c09 .cell .code execution_count="23"}
``` python
plt.figure(figsize=(15,6))
sns.regplot(x='age',y='charges',data=df)
plt.show()
```

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/2d120ea83c8f61f055ad2fb261423430415f3483.png)
:::
:::

::: {#361eb0ff .cell .code execution_count="24"}
``` python
plt.figure(figsize=(15,6))
sns.regplot(x='bmi',y='charges',data=df)
plt.show()
```

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/b0809fccaee859df3aa2c969fcb20d18c4e40d6b.png)
:::
:::

::: {#1b2299a6 .cell .code execution_count="25"}
``` python
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),annot=True)
plt.show()
```

::: {.output .display_data}
![](vertopal_94040f3ad45a40519a8e9aafb2a8d3ab/c7cd7797b96723bf5573bcca7719bf34f2ccef15.png)
:::
:::

::: {#e229dff7 .cell .code}
``` python
```
:::

::: {#1874f827 .cell .markdown}
# Linear Regression
:::

::: {#717b77c6 .cell .code execution_count="26"}
``` python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
:::

::: {#5b815c5c .cell .code execution_count="27"}
``` python
X = df.drop(['charges'],axis=1)
y = df['charges']
```
:::

::: {#ff50b557 .cell .code execution_count="28"}
``` python
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
```
:::

::: {#6dfbe17b .cell .code execution_count="29"}
``` python
lm = LinearRegression()
```
:::

::: {#4f2300f4 .cell .code execution_count="30"}
``` python
lm.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="30"}
    LinearRegression()
:::
:::

::: {#39b4675e .cell .code execution_count="31"}
``` python
y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)
print(lm.score(X_test,y_test))
```

::: {.output .stream .stdout}
    0.793108883437337
:::
:::

::: {#ae15e9b2 .cell .code execution_count="32"}
``` python
results = pd.DataFrame({'Actual':y_test,'Predicted':y_test_pred})
results
```

::: {.output .execute_result execution_count="32"}
```{=html}
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
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>578</th>
      <td>9724.53000</td>
      <td>11240.903073</td>
    </tr>
    <tr>
      <th>610</th>
      <td>8547.69130</td>
      <td>9369.966384</td>
    </tr>
    <tr>
      <th>569</th>
      <td>45702.02235</td>
      <td>38011.713581</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>12950.07120</td>
      <td>16867.167820</td>
    </tr>
    <tr>
      <th>198</th>
      <td>9644.25250</td>
      <td>7500.286394</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>574</th>
      <td>13224.05705</td>
      <td>14648.200711</td>
    </tr>
    <tr>
      <th>1174</th>
      <td>4433.91590</td>
      <td>6681.344859</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>9377.90470</td>
      <td>10634.197301</td>
    </tr>
    <tr>
      <th>817</th>
      <td>3597.59600</td>
      <td>6223.124209</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>29141.36030</td>
      <td>37415.262037</td>
    </tr>
  </tbody>
</table>
<p>335 rows × 2 columns</p>
</div>
```
:::
:::

::: {#ce939194 .cell .code execution_count="33"}
``` python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
:::

::: {#d9b90597 .cell .code execution_count="34"}
``` python
pd.DataFrame(X_train).head()
```

::: {.output .execute_result execution_count="34"}
```{=html}
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.514853</td>
      <td>0.985155</td>
      <td>-0.181331</td>
      <td>-0.503736</td>
      <td>-0.557773</td>
      <td>1.622978</td>
      <td>-0.593087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.548746</td>
      <td>0.985155</td>
      <td>-1.393130</td>
      <td>-0.503736</td>
      <td>-0.557773</td>
      <td>-0.616151</td>
      <td>-0.593087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.439915</td>
      <td>-1.015069</td>
      <td>-0.982242</td>
      <td>-0.503736</td>
      <td>-0.557773</td>
      <td>-0.616151</td>
      <td>1.686094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.368757</td>
      <td>0.985155</td>
      <td>-1.011133</td>
      <td>1.985167</td>
      <td>-0.557773</td>
      <td>1.622978</td>
      <td>-0.593087</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.941805</td>
      <td>0.985155</td>
      <td>-1.362635</td>
      <td>-0.503736</td>
      <td>1.792843</td>
      <td>-0.616151</td>
      <td>-0.593087</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#2dde347a .cell .markdown}
# Linear Regression {#linear-regression}
:::

::: {#0090df21 .cell .code execution_count="35"}
``` python
from sklearn.linear_model import LinearRegression
multiple_linear_reg = LinearRegression(fit_intercept=False)
multiple_linear_reg.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="35"}
    LinearRegression(fit_intercept=False)
:::
:::

::: {#a78c3742 .cell .markdown}
# Descision Tree
:::

::: {#02a133fc .cell .code execution_count="36"}
``` python
from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)
decision_tree_reg.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="36"}
    DecisionTreeRegressor(max_depth=5, random_state=13)
:::
:::

::: {#ef28c023 .cell .markdown}
# Random Forest Regression
:::

::: {#77efb61a .cell .code execution_count="37"}
``` python
from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=13)
random_forest_reg.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="37"}
    RandomForestRegressor(max_depth=5, n_estimators=500, random_state=13)
:::
:::

::: {#505ca602 .cell .code execution_count="38"}
``` python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
```
:::

::: {#250bb826 .cell .markdown}
# Evaluating Multiple Linear Regression
:::

::: {#df370a9d .cell .code execution_count="39"}
``` python
#prediction with training dataset
y_pred_mlr_train = multiple_linear_reg.predict(X_train)

#prediction with testing dataset
y_pred_mlr_test = multiple_linear_reg.predict(X_test)
#training accuracy
accuracy_mlr_train = r2_score(y_train,y_pred_mlr_train)
print('Training accuracy for MLR Model:',accuracy_mlr_train)
#testing accuracy
accuracy_mlr_testing = r2_score(y_test,y_pred_mlr_test)
print('Testing accuracy for MLR Model:',accuracy_mlr_testing)
#prediction with 10-fold CV
y_pred_cv_mlr = cross_val_predict(multiple_linear_reg,X,y,cv=10)
#Accuracy after 10-fold CV
accuracy_cv_mlr = r2_score(y, y_pred_cv_mlr)
print('Accuracy for 10-fold Cross Predicted MLR Model',accuracy_cv_mlr)
```

::: {.output .stream .stdout}
    Training accuracy for MLR Model: -0.49022171476415677
    Testing accuracy for MLR Model: -0.3211306235643383
    Accuracy for 10-fold Cross Predicted MLR Model 0.72007881362976
:::
:::

::: {#d4dff8ad .cell .markdown}
# Evaluating Decision Tree
:::

::: {#3a6fdace .cell .code execution_count="40"}
``` python
y_pred_dtr_train = decision_tree_reg.predict(X_train)
y_pred_dtr_test = decision_tree_reg.predict(X_test)
accuracy_dtr_train = r2_score(y_train,y_pred_dtr_train)
print('Training accuracy for DTR Model:', accuracy_dtr_train)
accuracy_dtr_testing = r2_score(y_test,y_pred_dtr_test)
print('Testing accuracy for DTR Model:',accuracy_dtr_testing)
#prediction with 10-fold CV
y_pred_cv_dtr = cross_val_predict(decision_tree_reg,X,y,cv=10)
#Accuracy after 10-fold CV
accuracy_cv_dtr = r2_score(y,y_pred_cv_dtr)
print('Accuracy for 10-fold Cross Predicted DTR Model',accuracy_cv_dtr)
```

::: {.output .stream .stdout}
    Training accuracy for DTR Model: 0.8665597738233303
    Testing accuracy for DTR Model: 0.8646418661352426
    Accuracy for 10-fold Cross Predicted DTR Model 0.8414618512168393
:::
:::

::: {#9d9445e6 .cell .markdown}
# Evaluating Random Forest Regression Model
:::

::: {#ccd5a2fb .cell .code execution_count="41"}
``` python
y_pred_rfr_train = random_forest_reg.predict(X_train)
y_pred_rfr_test = random_forest_reg.predict(X_test)
accuracy_rfr_train = r2_score(y_train,y_pred_rfr_train)
print('Training accuracy for RFR Model:', accuracy_rfr_train)
accuracy_rfr_testing = r2_score(y_test,y_pred_rfr_test)
print('Testing accuracy for RFR Model:',accuracy_rfr_testing)
#prediction with 10-fold CV
y_pred_cv_rfr = cross_val_predict(random_forest_reg,X,y,cv=10)
#accuracy after 10-fold CV
accuracy_cv_rfr = r2_score(y,y_pred_cv_rfr)
print('Accuracy for 10-fold Cross Predicted RFR Model',accuracy_cv_rfr)
```

::: {.output .stream .stdout}
    Training accuracy for RFR Model: 0.8749203691287142
    Testing accuracy for RFR Model: 0.8927239358398791
    Accuracy for 10-fold Cross Predicted RFR Model 0.8541852708064853
:::
:::

::: {#e765f604 .cell .markdown}
# Testing our best regression on new data
:::

::: {#8e93fd16 .cell .code execution_count="42"}
``` python
training_accuracy = [accuracy_mlr_train, accuracy_dtr_train, accuracy_rfr_train]
testing_accuracy = [accuracy_mlr_testing, accuracy_dtr_testing, accuracy_rfr_testing]
cv_accuracy = [accuracy_cv_mlr, accuracy_cv_dtr, accuracy_cv_rfr]

table_data = {"Training Accuracy": training_accuracy, "Testing Accuracy": testing_accuracy, 
               "10-Fold Score": cv_accuracy}
model_names = ["Multiple Linear Regression", "Decision Tree Regression", "Random Forest Regression"]

table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe
```

::: {.output .execute_result execution_count="42"}
```{=html}
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
      <th>Training Accuracy</th>
      <th>Testing Accuracy</th>
      <th>10-Fold Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Multiple Linear Regression</th>
      <td>-0.490222</td>
      <td>-0.321131</td>
      <td>0.720079</td>
    </tr>
    <tr>
      <th>Decision Tree Regression</th>
      <td>0.866560</td>
      <td>0.864642</td>
      <td>0.841462</td>
    </tr>
    <tr>
      <th>Random Forest Regression</th>
      <td>0.874920</td>
      <td>0.892724</td>
      <td>0.854185</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#f3a225d7 .cell .code execution_count="43"}
``` python
df.head()
```

::: {.output .execute_result execution_count="43"}
```{=html}
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
      <th>age</th>
      <th>if_female</th>
      <th>bmi</th>
      <th>if_smoker</th>
      <th>charges</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>1</td>
      <td>27.900</td>
      <td>1</td>
      <td>16884.92400</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>0</td>
      <td>33.770</td>
      <td>0</td>
      <td>1725.55230</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0</td>
      <td>33.000</td>
      <td>0</td>
      <td>4449.46200</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>0</td>
      <td>22.705</td>
      <td>0</td>
      <td>21984.47061</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>0</td>
      <td>28.880</td>
      <td>0</td>
      <td>3866.85520</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#cfc69082 .cell .markdown}
# Predicting charges using sample data
:::

::: {#c3f7a1b3 .cell .code execution_count="59"}
``` python
inp_data = {'age':[62],
            'sex':['female'],
            'bmi':[32.965],
            'smoker':['no'],
            'northwest':[1],
            'southeast':[0],
            'southwest':[0]
            }
inp_data = pd.DataFrame(inp_data)
inp_data
```

::: {.output .execute_result execution_count="59"}
```{=html}
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>smoker</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62</td>
      <td>female</td>
      <td>32.965</td>
      <td>no</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#3f1446b8 .cell .code execution_count="60"}
``` python
inp_data['sex'] = inp_data['sex'].map(lambda x :1 if x=='female' else 0 )
inp_data['smoker'] = inp_data['smoker'].map(lambda x :1 if x=='yes' else 0)
inp_data
```

::: {.output .execute_result execution_count="60"}
```{=html}
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>smoker</th>
      <th>northwest</th>
      <th>southeast</th>
      <th>southwest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62</td>
      <td>1</td>
      <td>32.965</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#95246723 .cell .code execution_count="61"}
``` python
inp_data = sc.transform(inp_data)
inp_data
```

::: {.output .execute_result execution_count="61"}
    array([[ 1.61990478,  0.98515504,  0.36036735, -0.50373604,  1.79284291,
            -0.61615125, -0.59308686]])
:::
:::

::: {#19a62006 .cell .code execution_count="62"}
``` python
random_forest_reg.predict(inp_data)
```

::: {.output .execute_result execution_count="62"}
    array([15177.58494173])
:::
:::

::: {#2d095f65 .cell .code}
``` python
```
:::
