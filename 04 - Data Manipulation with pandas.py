import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

os.chdir(r'E:\OneDrive - КНЕУ\Університет\Courses\DataCamp - Data Scientist with Python')
print(os.getcwd())

# %% 1. Transforming DataFrames
homelessness = pd.read_csv('data/homelessness.csv')
# Методы
homelessness.head()
homelessness.info()
homelessness.describe()

# Атрибуты
homelessness.shape
homelessness.values
homelessness.columns
homelessness.index

# Sorting and subsetting
homelessness.sort_values('family_members')
homelessness['family_members'].sort_values()
homelessness.sort_values('family_members', ascending=False)

homelessness.sort_values(['family_members', 'state_pop'])
homelessness.sort_values(['family_members', 'state_pop'], ascending=[False, True])

homelessness['region']
homelessness[['region', 'state_pop']]

homelessness['state_pop'] > np.mean(homelessness['state_pop'])
homelessness[homelessness['state_pop'] > np.mean(homelessness['state_pop'])]
homelessness[homelessness['region'] == 'New England']

new_england = homelessness['region'] == 'New England'
state_pop_2kk = homelessness['state_pop'] > 2000000
homelessness[new_england & state_pop_2kk]
homelessness[(homelessness['state_pop'] > 2000000) & (homelessness['region'] == 'New England')]

new_eng_and_mount = homelessness['region'].isin(['New England', 'Mountain'])
homelessness[new_eng_and_mount]
homelessness[homelessness['region'].isin(['New England', 'Mountain'])]

# New columns
homelessness['fam_st_pop'] = (homelessness['family_members'] / homelessness['state_pop']) * 100
homelessness

# %% 2. Aggregating DataFrames

# Summary statistics
sales = pd.read_csv('data/sales_subset.csv', index_col=0)

sales['weekly_sales'].mean()
sales['weekly_sales'].median()
sales['is_holiday'].mode()
sales['weekly_sales'].min()
sales['weekly_sales'].max()
sales['weekly_sales'].var()
sales['weekly_sales'].std()
sales['weekly_sales'].sum()
sales['weekly_sales'].quantile(0.5)

sales['weekly_sales'].sort_values()
sales.sort_values('weekly_sales')


def pct30(column):
    return column.quantile(0.3)


sales['weekly_sales'].agg(pct30)
sales[['weekly_sales', 'temperature_c']].agg(pct30)


def pct40(column):
    return column.quantile(0.4)


sales[['weekly_sales', 'temperature_c']].agg([pct30, pct40])

sales['weekly_sales'].cumsum()
sales['weekly_sales'].cummin()
sales['weekly_sales'].cumprod()

# Counting
sales.drop_duplicates('store') \
    .value_counts('type')
sales.drop_duplicates(['store', 'type']) \
    .value_counts('department', sort=True)
sales.drop_duplicates(['date']) \
    .value_counts('is_holiday', normalize=True)

sales.drop_duplicates(['store', 'type']).value_counts('type', normalize=True)
sales.drop_duplicates(['store', 'department'])['department'].value_counts(sort=True)
sales[sales['is_holiday']].drop_duplicates('date')
sales.drop_duplicates(['store', 'department'])['department'].value_counts(sort=True, normalize=True)

# Grouped summary statistics
sales.groupby('type')['weekly_sales'].mean()
sales.groupby('type')['weekly_sales'].agg([min, max, np.median, np.std])
sales.groupby(['type', 'is_holiday'])['weekly_sales'].mean()
sales.groupby(['type', 'is_holiday'])['weekly_sales'].agg([min, max, np.median, np.std])

sales_all = sales["weekly_sales"].sum()
sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()
sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()
sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)

# Pivot tables
sales.pivot_table('weekly_sales', 'type')
sales.pivot_table('weekly_sales', 'type', aggfunc=np.median)
sales.pivot_table('weekly_sales', 'type', aggfunc=[np.mean, np.median])

sales.pivot_table('weekly_sales', 'type', 'is_holiday')
sales.pivot_table('weekly_sales', 'type', 'is_holiday', aggfunc=[np.mean, np.median])
sales.pivot_table('weekly_sales', 'type', 'is_holiday', aggfunc=[np.mean, np.median], fill_value=0)
sales.pivot_table('weekly_sales', 'type', 'is_holiday', aggfunc=[np.mean, np.median], margins=True)

# %% 3. Slicing and Indexing DataFrames
# Explicit indexes
temperatures = pd.read_csv('data/temperatures.csv', index_col=0)
temperatures_ind = temperatures.set_index('city')

cities = ["Moscow", "Saint Petersburg"]
temperatures[temperatures['city'].isin(cities)]
temperatures_ind.loc[cities]

temperatures_ind = temperatures.set_index(['country', 'city'])
rows_to_keep = [('Brazil', 'Rio De Janeiro'), ('Pakistan', 'Lahore')]
print(temperatures_ind.loc[rows_to_keep])

temperatures_ind.sort_index()
temperatures_ind.sort_index(level='city')
temperatures_ind.sort_index(level=['country', 'city'], ascending=[True, False])

# Slicing and subsetting with .loc and .iloc
temperatures_ind_srt = temperatures_ind.sort_index()
temperatures_ind_srt.loc['Afghanistan':'Albania']
temperatures_ind_srt.loc[('Afghanistan', 'Kabul'): ('China', 'Xian')]
temperatures_ind_srt.loc[:, 'date':'avg_temp_c']
temperatures_ind_srt.loc[('Afghanistan', 'Kabul'):('China', 'Xian'), 'avg_temp_c']
temperatures.set_index('date').sort_index().loc['2013-08-01':'2013-09-01']
temperatures.set_index('date').sort_index().loc['2012':'2013']

temperatures_ind_srt.iloc[:5, 1:4]

temperatures_bool = temperatures[(temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
print(temperatures_bool)
temperatures_ind = temperatures.set_index('date').sort_index()
print(temperatures_ind.loc['2010':'2011'])
print(temperatures_ind.loc['2010-08':'2011-02'])

# Working with pivot tables
temperatures_pivot = temperatures.pivot_table(
    'avg_temp_c', 'country', 'date'
)
temperatures_pivot.loc['Ukraine':'United States']
temperatures_pivot.mean(axis='index')
temperatures_pivot.mean(axis='columns')

temperatures['date'] = pd.to_datetime(temperatures.date)
temperatures['year'] = temperatures['date'].dt.year
temp_by_country_city_vs_year = temperatures.pivot_table(
    values='avg_temp_c', index=['country', 'city'], columns='year'
)
temp_by_country_city_vs_year
temp_by_country_city_vs_year.loc['Egypt':'India']
temp_by_country_city_vs_year.loc[('Egypt', 'Cairo'):('India', 'Delhi')]
temp_by_country_city_vs_year.loc[('Egypt', 'Cairo'):('India', 'Delhi'), 2005:2010]
# same
temp_by_country_city_vs_year.loc["Egypt":"India"]
temp_by_country_city_vs_year.loc[("Egypt", "Cairo"):("India", "Delhi")]
temp_by_country_city_vs_year.loc[("Egypt", "Cairo"):("India", "Delhi"), "2005":"2010"]

mean_temp_by_year = temp_by_country_city_vs_year.mean(axis='index')
mean_temp_by_year[mean_temp_by_year == mean_temp_by_year.max()]
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis='columns')
mean_temp_by_city[mean_temp_by_city == mean_temp_by_city.min()]

# %% 4. Creating and Visualizing DataFrames
# Visualizing your data
temperatures['avg_temp_c'].hist(bins=20)
plt.show()

temperatures.groupby('year')['avg_temp_c'].mean().plot(kind='bar',
                                                       title="Mean temp by year")
plt.show()

temperatures[temperatures['city'] == 'Xian'].plot(
    x='date',
    y='avg_temp_c',
    kind='line',
    rot=45
)
plt.show()

sales.plot('temperature_c', 'weekly_sales', 'scatter')
plt.show()

sales[sales['store'] == 1]['unemployment'].hist(alpha=0.7)
sales[sales['store'] == 2]['unemployment'].hist(alpha=0.7)
plt.legend([1, 2])
plt.show()

# Import
import pickle
with open('data/avoplotto.pkl', 'rb') as f:
    avocados = pickle.load(f)

nb_sold_by_size = avocados.groupby('size')['nb_sold'].sum()
nb_sold_by_size.plot(kind='bar', rot=0)
plt.show()

nb_sold_by_date = avocados.groupby('date')['nb_sold'].sum()
nb_sold_by_date.plot(kind='line', rot=45)
plt.show()

avocados.plot('nb_sold', 'avg_price', kind='scatter',
              title="Number of avocados sold vs. average price")
plt.show()

avocados[avocados['type'] == 'conventional']['avg_price'].hist(alpha=0.5, bins=20)
avocados[avocados['type'] == 'organic']['avg_price'].hist(alpha=0.5, bins=20)
plt.legend(['conventional', 'organic'])
plt.show()

# Missing values
avocados.isna().any()
avocados.isna().sum()

avocados.isna().sum().plot(kind='bar')
plt.show()

avocados.dropna()
avocados.fillna(0)

cols = ["avg_price", "nb_sold"]
avocados[cols].hist()
plt.show()

# Creating DataFrames
my_dict = {
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3',
}

my_dict['key1']

# 1. List of dicts
list_of_dicts = [
    {
        'name': 'Ginger',
        'breed': 'Dach',
        'height': 22,
        'weight': 10
    },
    {
        'name': 'Scout',
        'breed': 'Dalm',
        'height': 59,
        'weight': 25
    }
]
new_df1 = pd.DataFrame(list_of_dicts)

# 2. Dicts of list
dict_of_lists = {
    'name': ['Ginger', 'Scout'],
    'breed': ['Dach', 'Dalm'],
    'height': [22, 59],
    'weight': [10, 25]
}
new_df2 = pd.DataFrame(dict_of_lists)

# Reading and writing CSVs

avocados.sort_values('avg_price', ascending=False)