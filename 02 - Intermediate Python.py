# %% Matplotlib --------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import display

# %% Load gapmainder
gapmainder = pd.read_csv('data/gapminder.csv')
print(gapmainder)

# %% Example

year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# %% Line plot

plt.plot(year, pop)
plt.show()

# %% Scatter plot

plt.scatter(year, pop)
plt.show()

# %% Last observation

print(year[-1])
print(pop[-1])

# %% Var select

gdp_cap = gapmainder["gdp_cap"]
life_exp = gapmainder["life_exp"]
population = gapmainder["population"]

# %% Log plot

plt.scatter(gdp_cap, life_exp)
plt.xscale('log')
plt.show()

# %% No correlation

plt.scatter(population, life_exp)
plt.show()

# %% Histogram

plt.hist(life_exp, bins=15)
plt.show()
plt.clf()
plt.hist(life_exp, bins=20)
plt.show()

# %% Plots Customization

year = [1800, 1850, 1900] + year  # додамо ще даних
pop = [1.0, 1.262, 1.65] + pop

plt.plot(year, pop)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World population')
plt.yticks(range(0, 11, 2),
           ['0', '2B', '4B', '6B', '8B', '10B'])  # зміна шкали

plt.show()

# %% Example

np_pop = np.array(gapmainder.population)
np_pop2 = np_pop / 1000000

plt.scatter(gapmainder['gdp_cap'], gapmainder['life_exp'], s=np_pop2)
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')
plt.grid(True)
plt.show()

# %% Dictionaries --------------------------------------------------------------------------------------------

# Poor way
pop = [30.55, 2.77, 39.21]
countries = ["afg", "alb", "alg"]
ind_alb = countries.index("alb")
print(ind_alb)
print(pop[ind_alb])

# Dictionaries!
world = {'afg': 30.55,
         'alb': 2.77,
         'alg': 39.21}
print(world.keys())
print(world['alb'])
# Ключі - незмінювані об'єкти, так само як строки, логічні значення, цілі числа і числа з комою що плаває.
# Списки - можна змінювати.

# Додамо нове значення у словник
world['sealand'] = 0.000028
world

# Видалимо значення зі словника
del (world['sealand'])
world

print('sealand' in world)

# %% Ще приклад з вкладеним словником
europe = {'spain': {'capital': 'madrid',
                    'population': 46.77},
          'france': {'capital': 'paris',
                     'population': 66.03},
          'germany': {'capital': 'berlin',
                      'population': 80.62},
          'norway': {'capital': 'oslo',
                     'population': 5.084}}

print(europe['france']['capital'])

data = {'capital': 'rome',
        'population': 59.83}

europe['italy'] = data

print(europe)

# %% Pandas --------------------------------------------------------------------------------------------
# DF зі словника
dict = {
    'country': ["Brazil", "India", "China"],
    'capital': ["Brasilia", "New Delhi", "Beijing"],
    'area': [8.516, 3.286, 9.597],
    'population': [200.4, 1252, 1357]
}
brics = pd.DataFrame(dict)
brics

brics.index = ['BR', 'IN', 'CH']

brics_df = pd.read_csv('data/brics.csv')
brics_df

brics_df = pd.read_csv('data/brics.csv', index_col=0)
brics_df

# Стовпчики
brics_df['country']
type(brics_df['country'])

brics_df[['country']]
type(brics_df[['country']])

brics_df[['country', 'capital']]

# Рядки
# poor way
brics_df[1:4]

# .loc - label-based
brics_df.loc['IN']
brics_df.loc[['IN']]
brics_df.loc[['IN', 'BR', 'SA']]

# Стовпчики + рядки
brics_df.loc[:, ['country', 'capital']]
brics_df.loc[
    ['IN', 'BR', 'SA'],
    ['country', 'capital']
]

# .iloc - index-based
brics_df.iloc[2]
brics_df.iloc[[2]]
brics_df.iloc[[2, 4]]
brics_df.iloc[:, [0, 1]]
brics_df.iloc[[2, 4], [0, 1]]

# %% Порівняння --------------------------------------------------------------------------------------------
0 < 3
2 == 3
2 <= 3

x = 2
y = 3
x > 3

'carl' < 'chris'
# 3 < 'carl' - помилка, не можна порівнювати int та str

3 < 4.1

"pyscript" == "PyScript"

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
my_house >= 18
my_house < your_house

# %% Булеві оператори: and or not
x = 12
x > 5 and x < 15
5 < x < 15

y = 5
y < 7 or y > 13

np.logical_and(my_house > 17, my_house < 19)
my_house[np.logical_and(my_house > 17, my_house < 19)]

# %% if, elif, else --------------------------------------------------------------------------------------------
z = 3
if z % 2 == 0:
    print("checking " + str(z))
    print('z is even')
else:
    print("checking " + str(z))
    print('z is odd')

z = 3
if z % 2 == 0:
    print('z ділиться на 2')
elif z % 3 == 0:
    print('z ділиться на 3')
else:
    print('z не ділиться на 2 та 3')

# %% Filter pandas DF --------------------------------------------------------------------------------------------
is_huge = brics_df['area'] > 8
brics_df[is_huge]

brics_df[brics_df['area'] > 8]

np.logical_and(brics_df['area'] > 8, brics_df['area'] < 10)
brics_df[np.logical_and(brics_df['area'] > 8, brics_df['area'] < 10)]

# Приклад
cars = pd.read_csv('data/cars.csv', index_col=0)
cars
cpc = cars['cars_per_cap']
many_cars = cars[cpc > 500]
car_maniac = cars[cars['cars_per_cap'] > 500]

medium = cars[np.logical_and(cars['cars_per_cap'] > 100,
                             cars['cars_per_cap'] < 500)]
medium

# %% Loops --------------------------------------------------------------------------------------------
# %% While loop
error = 50
while error > 1:
    # error = error / 4
    error /= 4
    print(error)

offset = -6
while offset != 0:
    print("correcting...")
    if offset > 0:
        offset -= 1
    else:
        offset += 1
    print(offset)

# %% for loop
fam = [1.73, 1.68, 1.71, 1.89]
print(fam[0])
print(fam[1])
print(fam[2])
print(fam[3])

# Без індексу
for var in fam:
    print(var)

# З індексом
for index, var in enumerate(fam):
    print("index " + str(index) + ": " + str(var))

# Слово по літерам
for c in "family":
    print(c.capitalize())

house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]

# Build a for loop from scratch
# my:
for i, j in enumerate(house):
    print("The " + str(house[i][0]) + " is " + str(house[i][1]) + " sqm")
# correct:
for x in house:
    print("the " + x[0] + " is " + str(x[1]) + " sqm")

# Loop Data Structures Part 1
# 1. loop dictionary
world

for key, value in world.items():
    print(key + " -- " + str(value))

# 2. loop np.array
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
bmi = np_weight / np_height ** 2

for val in bmi:
    print(val)

meas = np.array([np_height, np_weight])

for val in meas:
    print(val)

for val in np.nditer(meas):
    print(val)

# Loop Data Structures Part 2
brics_df

# Друкує назви стовпчиків
for val in brics_df:
    print(val)

# Окремо кожен об'єкт
for lab, row in brics_df.iterrows():
    print(lab)
    print(row)

# Індекс + столиця
for lab, row in brics_df.iterrows():
    print(lab + ": " + row['capital'])

# Новий стовпчик з довжиною назви
for lab, row in brics_df.iterrows():
    brics_df.loc[lab, "name_length"] = len(row['country'])
print(brics_df)

# Новий стовпчик з довжиною назви: apply - так краще!
brics_df['name_length'] = brics_df['country'].apply(len)
print(brics_df)

# %% Random Numbers -----------------------------------------------------------------------------------
np.random.seed(123)
np.random.rand()

np.random.seed(123)
# Генеруємо 0 або 1
coin = np.random.randint(0, 2)
print(coin)

if coin == 0:
    print('heads')
else:
    print('tails')

# Піднімаємось по сходинкам
# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice <= 5 :
    step = step + 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)

# %% Random Walk
# 1.
np.random.seed(123)
outcomes = []
for i in range(10):
    coin = np.random.randint(0, 2)
    if coin == 0:
        outcomes.append("heads")
    else:
        outcomes.append("tails")
print(outcomes)

# 2.
np.random.seed(123)
tails = [0]
for i in range(10):
    coin = np.random.randint(0, 2)
    tails.append(tails[i] + coin)
print(tails)

# 3.
# Initialize random_walk
np.random.seed(123)
random_walk = [0]

for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)
    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

print(random_walk)
plt.plot(random_walk)
plt.show()

# %% Distribution
np.random.seed(123)
final_tails = []

for i in range(10000):
    tails = [0]
    for i in range(10):
        coin = np.random.randint(0, 2)
        tails.append(tails[i] + coin)
    final_tails.append(tails[-1])

plt.hist(final_tails, bins=10)
plt.show()

# 2.
all_walks = []
for i in range(10) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()

# 3.
# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

# 4.
# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()