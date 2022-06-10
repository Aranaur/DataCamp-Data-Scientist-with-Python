import pandas as pd
pd.set_option('display.max_columns', 5)

#%%  1. Inner join
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

wards_census = wards.merge(census, on="ward", suffixes=('_ward', '_cen'))
wards_census.head()
wards_census.shape

#%%  1.1 Inner join - taxi
taxi_owners = pd.read_pickle('data/06/taxi_owners.p')
taxi_veh = pd.read_pickle('data/06/taxi_vehicles.p')

taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid')
taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid', suffixes=('_own', '_veh'))

taxi_own_veh['fuel_type'].value_counts()

#%%  One-to-many relationships
