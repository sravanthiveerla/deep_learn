```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
```


```python
data = pd.read_csv(r"C:\Users\sravanth\Desktop\Deep_Learning\iris_modified.csv")
```


```python
df = data.copy()
```


```python
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
      <th>Unnamed: 0</th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
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
      <th>Unnamed: 0</th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>95</td>
      <td>5.7</td>
      <td>3.0</td>
      <td>4.2</td>
      <td>1.2</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <th>96</th>
      <td>96</td>
      <td>5.7</td>
      <td>2.9</td>
      <td>4.2</td>
      <td>1.3</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <th>97</th>
      <td>97</td>
      <td>6.2</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <th>98</th>
      <td>98</td>
      <td>5.1</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>5.7</td>
      <td>2.8</td>
      <td>4.1</td>
      <td>1.3</td>
      <td>versicolor</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop("Unnamed: 0",axis=1,inplace=True)
```


```python
df.shape
```




    (100, 5)




```python
df.isnull().sum()
```




    sepal_length    0
    sepal_width     0
    petal_length    0
    petal_width     0
    species         0
    dtype: int64




```python
df.describe()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.471000</td>
      <td>3.099000</td>
      <td>2.861000</td>
      <td>0.786000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.641698</td>
      <td>0.478739</td>
      <td>1.449549</td>
      <td>0.565153</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>2.800000</td>
      <td>1.500000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.400000</td>
      <td>3.050000</td>
      <td>2.450000</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.900000</td>
      <td>3.400000</td>
      <td>4.325000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>4.400000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.duplicated().sum()
```




    0




```python
df["species"].value_counts()
```




    species
    setosa        50
    versicolor    50
    Name: count, dtype: int64




```python
num = ["sepal_length","sepal_width","petal_length","petal_width"]
for i in num:
    sns.kdeplot(x = i ,hue = "species",data=df)
    plt.show()
```

    C:\Users\sravanth\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_11_1.png)
    


    C:\Users\sravanth\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_11_3.png)
    


    C:\Users\sravanth\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_11_5.png)
    


    C:\Users\sravanth\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](output_11_7.png)
    



```python
num_col1 = ["sepal_length","sepal_width","petal_length","petal_width"]
for i in num_col1:
    for j in num_col1:
        if i!=j:
            sns.scatterplot(x=i,y=j,data=df,hue="species" )
            plt.show()
```


    
![png](output_12_0.png)
    



    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



    
![png](output_12_7.png)
    



    
![png](output_12_8.png)
    



    
![png](output_12_9.png)
    



    
![png](output_12_10.png)
    



    
![png](output_12_11.png)
    



```python
class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(num_features + 1)

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Splitting data into features and target
X = df.iloc[:, :-1].values
y = np.where(df['species'] == 'Iris-setosa', 1, 0)

# Feature scaling (optional for perceptron)
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Adding bias term to features
X_with_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Training the perceptron
perceptron = Perceptron(num_features=X_with_bias.shape[1])
perceptron.train(X_with_bias, y)

# Step 5: Plot the Separator Line
# Plotting decision boundary
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, marker='o', edgecolors='k')

x_vals = np.array([np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0])])
y_vals = -(perceptron.weights[1] * x_vals + perceptron.weights[0]) / perceptron.weights[2]
plt.plot(x_vals, y_vals, 'r--')

plt.title('Perceptron Separator Line')
plt.xlabel('petal Length (scaled)')
plt.ylabel('petal Width (scaled)')
plt.show()
```


    
![png](output_13_0.png)
    



```python

```


```python

```


```python

```


```python

```
