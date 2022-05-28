import pandas as pd
pd.set_option('display.max_columns', 5)

# 1. Inner join
census = pd.read_pickle('data/06/census.p')
wards = pd.read_pickle('data/06/ward.p')

census.head()
census.shape

wards.head()
wards.shape

wards_census = wards.merge(census, on="ward")
wards_census.head()
wards_census.shape

wards_census.columns