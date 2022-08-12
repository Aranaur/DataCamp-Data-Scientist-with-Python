#%% 1. Preparing the data for analysis
import pandas as pd

#%% 1.1 Stanford Open Policing Project dataset
ri = pd.read_csv('data/24/police.csv')
ri.head()

ri.isnull().sum().sort_values(ascending=False)

ri.shape

ri.drop('county_name', axis='columns', inplace=True)

ri.head()

ri.dropna(subset=['stop_date', 'stop_time'], inplace=True)

#%%
# Import the pandas library as pd
import pandas as pd

# Read 'police.csv' into a DataFrame named ri
ri = pd.read_csv('police.csv')

# Examine the head of the DataFrame
print(ri.head())

# Count the number of missing values in each column
print(ri.isnull().sum())

#%%
# Examine the shape of the DataFrame
print(ri.shape)

# Drop the 'county_name' and 'state' columns
ri.drop(['county_name', 'state'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(ri.shape)

#%%
# Count the number of missing values in each column
print(ri.isnull().sum())

# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=['driver_gender'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame
print(ri.shape)

#%% 1.2 Using proper data types
ri.dtypes

#%%
# Examine the head of the 'is_arrested' column
print(ri.is_arrested.head())

# Change the data type of 'is_arrested' to 'bool'
ri['is_arrested'] = ri.is_arrested.astype('bool')

# Check the data type of 'is_arrested'
print(ri.is_arrested.dtypes)

#%% 1.3 Creating a DatetimeIndex
ri.head()
ri.dtypes

ri['stop_date_time'] = pd.to_datetime(ri.stop_date.str.cat(ri.stop_time, sep=' '))
ri.set_index('stop_date_time', inplace=True)
ri.index

#%%
# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = ri.stop_date.str.cat(ri.stop_time, ' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame
print(ri.dtypes)

#%%
# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns
print(ri.columns)

#%% 2. Exploring the relationship between gender and policing

#%% 2.1 Do the genders commit different violations?
