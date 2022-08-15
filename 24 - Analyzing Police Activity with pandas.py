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
ri.stop_outcome.value_counts()
ri.stop_outcome.value_counts().sum()
ri.shape

ri.stop_outcome.value_counts(normalize=True)

ri.driver_race.value_counts()

white = ri[ri.driver_race == 'White']
white.shape
white.stop_outcome.value_counts(normalize=True)

asian = ri[ri.driver_race == 'Asian']
asian.stop_outcome.value_counts(normalize=True)

#%%
# Count the unique values in 'violation'
print(ri.violation.value_counts())

# Express the counts as proportions
print(ri.violation.value_counts(normalize=True))

#%%
# Create a DataFrame of female drivers
female = ri[ri.driver_gender == 'F']

# Create a DataFrame of male drivers
male = ri[ri.driver_gender == 'M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize=True))

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize=True))

#%% 2.2 Does gender affect who gets a ticket for speeding?
female_and_arrested = ri[(ri.driver_gender == 'F') &
                         (ri.is_arrested == True)]
female_and_arrested.shape

female_or_arrested = ri[(ri.driver_gender == 'F') |
                         (ri.is_arrested == True)]
female_or_arrested.shape

#%%
# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender == 'F') & (ri.violation_raw == 'Speeding')]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender == 'M') & (ri.violation_raw == 'Speeding')]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize=True))

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize=True))


#%% 2.3
ri.isnull().sum()

ri.is_arrested.value_counts(normalize=True)
ri.is_arrested.dtype
ri.is_arrested.mean()

ri.district.unique()
ri[ri.district == 'Zone K1'].is_arrested.mean()