import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 4)

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

#%%  2. One-to-many relationships

licenses = pd.read_pickle('data/06/licenses.p')
ward_licenses = wards.merge(licenses, on='ward', suffixes=('_ward', '_lic'))
ward_licenses.head()
ward_licenses.shape

#%%  2.1 One-to-many relationships

biz_owners = pd.read_pickle('data/06/business_owners.p')

# Merge the licenses and biz_owners table on account
licenses_owners = licenses.merge(biz_owners, on='account')

# Group the results by title then count the number of accounts
counted_df = licenses_owners.groupby('title').agg({'account': 'count'})

# Sort the counted_df in desending order
sorted_df = counted_df.sort_values('account', ascending=False)

# Use .head() method to print the first few rows of sorted_df
print(sorted_df.head())

#%% 3. Merging multiple DataFrames

# grants_licenses_ward = grants.merge(licenses, on=['address', 'zip']) \
#    .merge(wards, on='ward', suffixes=('_bis', '_ward'))
# grants_licenses_ward.head()

# grants_licenses_ward.groupby('ward').agg('sum').plot(kind='bar', y='grant')
# plt.show()

# Merge the ridership, cal, and stations tables
# ridership_cal_stations = ridership.merge(cal, on=['year','month','day']) \
# 							.merge(stations, on='station_id')

# Create a filter to filter ridership_cal_stations
# filter_criteria = ((ridership_cal_stations['month'] == 7)
#                   & (ridership_cal_stations['day_type'] == 'Weekday')
#                   & (ridership_cal_stations['station_name'] == 'Wilson'))

# Use .loc and the filter to select for rides
# print(ridership_cal_stations.loc[filter_criteria, 'rides'].sum())

zip_demo = pd.read_pickle('data/06/zip_demo.p')

# Merge licenses and zip_demo, on zip; and merge the wards on ward
licenses_zip_ward = licenses.merge(zip_demo, on='zip')\
    .merge(wards, on='ward')

# Print the results by alderman and show median income
print(licenses_zip_ward.groupby('alderman').agg({'income':'median'}))

#%% 3.1 One-to-many merge with multiple tables
land_use = pd.read_pickle('data/06/land_use.p')

land_cen_lic = land_use.merge(census, on='ward')\
    .merge(licenses, on='ward', suffixes=('_cen', '_lic'))

pop_vac_lic = land_cen_lic.groupby(['ward', 'pop_2010', 'vacant'],
                                   as_index=False).agg({'account':'count'})

sorted_pop_vac_lic = pop_vac_lic.sort_values(['vacant', 'account', 'pop_2010'],
                                             ascending=[False, True, False])