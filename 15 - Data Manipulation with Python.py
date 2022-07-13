import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% 1
amazon = {'date': ['2013-02-08', '2013-02-11', '2013-02-12', '2013-02-13', '2013-02-14'],
          'close': [261.950, 257.210, 258.700, 269.470, 269.240]}

amazon = pd.DataFrame(amazon)

sns.lineplot(
    x='date',
    y='close',
    data=amazon)

plt.xticks(rotation=45)
plt.show()


#%% 2
employee = {'employee_id': ['1ex5', '73fd', 'ei10', 'b45e'],
            'first_name': ['Linda', 'Steve', 'Henry', 'Sara'],
            'gender': ['female', 'male', 'male', 'female'],
            'salary': [3400, 5000, 12400, 7600]}

employee = pd.DataFrame(employee)

private_employee = employee[['employee_id', 'salary']]

print(private_employee)

#%% 3

from scipy import stats
IQR = stats.iqr(pH)
print(IQR)


#%% 4

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)

print(Q3 - Q1)

#%% 5

df = {'gh owner': ['pandas-dev', 'tidyverse', 'tidyverse', 'has2k1'],
      'language': ['python', 'R', 'R', 'python'],
      'repo': ['pandas', 'dplyr', 'ggplot2', 'plotnine'],
      'stars': [17800, 2800, 3500, 1450]}

df = pd.DataFrame(df)

print(df.stars.apply(lambda x: x / 1000))
print(df.stars.transform(lambda x: x / 1000))


#%% 6

subset_wine = wine[['country', 'price']]
print(subset_wine.head())


#%% 7

x = np.array([2.2, 0.9, 4.4, 6.7, 2.8, 3.2, 1.1, 3.5])
x_var = np.var(x)
x_stdev = np.std(x)
print('Variance: {:4.2f}'.format(x_var))
print('Std Deviation: {:4.2f}'.format(x_stdev))

#%% 8

chess = chess.set_index('Fide id')

print(chess.head())

#%% 9

df = {'day': [0, 1, 2, 3, 4, 5],
      'order': [10, 11, 14, 7, 5, 16]}

df = pd.DataFrame(df)

sns.lineplot(data=df, x='day', y='order')
plt.xticks(rotation=45)
plt.show()

#%% 10

df = {'alignment': ['Good', 'Good', 'Evil', 'Evil'],
      'character': ['Batman', 'Wonder Woman', 'Lex Luthor', 'The Joker'],
      'height': [1.88, 1.78, 1.88, 1.85],
      'location': ['Gotham', 'Themyscira', 'Metropolis', 'Gotham']}

df = pd.DataFrame(df)

print(df[['alignment', 'character']])

#%% 11

employee = {'first_name': ['Linda', 'Steve', 'Henry', 'Sara'],
            'gender': ['female', 'male', 'male', 'female'],
            'salary_usd': [3000, 5000, 12000, 5000]}

employee = pd.DataFrame(employee)

print(employee[employee['salary_usd'] == 5000])

#%% 12

df = {'index': ['zero', 'one', 'two', 'tree'],
      'Month': ['Jan', 'Apr', 'Mar', 'Feb'],
      'Count': [52, 29, 46, 3]}

df = pd.DataFrame(df).set_index('index')

print(df.loc['two':])

#%% 13

# --iris
# species    measurement   value
# 0     setosa     sepal_length  5.1
# 1     setosa     sepal_length  4.9
# ...
# 598   virginica  petal_width   4.6
# 599   virginica  petal_width   5.0

ax = sns.swarmplot(x="measurement",
                   y="value",
                   hue="species",
                   data=iris)
plt.show()

#%% 14

ax = sns.lineplot(x='Overall rank', y='Score', data=happiness)

plt.show()

#%% 15

# age emissions   mpg  value
# 0    15       low  23.1   3047
# 1     8       low  10.4  15428
# 2     4      high  15.4  24973
# 3    10      high  31.0   4638
# 4     1       low  30.2  19303

# import matplotlib.pyplot as plt
# import seaborn as sns

sns.scatterplot(x = "age", y = "value", hue = "emissions", data = valuation)

plt.show()