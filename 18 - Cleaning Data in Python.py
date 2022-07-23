#%% Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import missingno as msno
import fuzzywuzzy
import recordlinkage

# Importing course datasets as DataFrames
ride_sharing = pd.read_csv('data/18/ride_sharing_new.csv', index_col='Unnamed: 0')
airlines = pd.read_csv('data/18/airlines_final.csv',  index_col='Unnamed: 0')
banking = pd.read_csv('data/18/banking_dirty.csv', index_col='Unnamed: 0', parse_dates=['birth_date'])
restaurants = pd.read_csv('data/18/restaurants_L2.csv', index_col='Unnamed: 0')
restaurants_new = pd.read_csv('data/18/restaurants_L2_dirty.csv', index_col='Unnamed: 0')

#%% 1. Common data problems

ride_sharing.dtypes
ride_sharing.info()

ride_sharing['duration'].sum()

ride_sharing['duration'] = ride_sharing['duration'].str.strip(' minutes')
ride_sharing['duration'] = ride_sharing['duration'].astype('int')

assert ride_sharing['duration'].dtype == 'int'

ride_sharing['user_type'].describe()
ride_sharing['user_type'] = ride_sharing['user_type'].astype('category')
ride_sharing['user_type'].describe()

#%%
# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

#%%
# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')

# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'

# Print new summary statistics
print(ride_sharing['user_type_cat'].describe())

#%%
# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration
print(ride_sharing[['duration', 'duration_trim', 'duration_time']])
print(ride_sharing['duration_time'].mean())

#%% 1.1 Data range constraints

banking['Age']

plt.hist(banking['Age'])
plt.title('Age of clients')

today_date = dt.date.today()
banking["birth_date"] = pd.to_datetime(banking["birth_date"]).dt.date
banking[banking['birth_date'] > dt.date.today()]

banking[banking['Age'] > 58]

banking = banking[banking['Age'] <= 58]
banking.drop(banking[banking['Age'] > 58].index, inplace=True)

assert banking['Age'].max() <= 58

banking.loc[banking['Age'] > 58, 'Age'] = 60
assert banking['Age'].max() <= 60

#%%

# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())
#%%

# Convert ride_date to date
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date']).dt.date

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())

#%% 1.2 Uniqueness constraints

restaurants.head()
duplicates = ride_sharing.duplicated()
duplicates.sum()
ride_sharing[duplicates]

duplicates = ride_sharing.duplicated(subset=['duration', 'bike_id', 'user_gender'], keep=False)
ride_sharing[duplicates].sort_values(by='user_birth_year')

ride_sharing.drop_duplicates(inplace=True)

column_names = ['station_A_id', 'bike_id', 'user_gender']
summaries = {'duration': 'max', 'user_birth_year': 'mean'}
ride_sharing = ride_sharing.groupby(by=column_names).agg(summaries).reset_index()

#%%
# Find duplicates
duplicates = ride_sharing.duplicated('ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id', 'duration', 'user_birth_year']])

#%%
# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0

#%% 2. Text and categorical data problems

#%% 2.1 Membership constraints
