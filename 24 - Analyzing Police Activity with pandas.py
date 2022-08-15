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

ri[ri.district == 'Zone K2'].is_arrested.mean()

ri.groupby('district').is_arrested.mean()

ri.groupby(['district', 'driver_gender']).is_arrested.mean()
ri.groupby(['driver_gender', 'district']).is_arrested.mean()

#%%
# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize=True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())

#%%
# Calculate the search rate for female drivers
print(ri[ri.driver_gender == 'F'].search_conducted.mean())

# Calculate the search rate for male drivers
print(ri[ri.driver_gender == 'M'].search_conducted.mean())

# Calculate the search rate for both groups simultaneously
print(ri.groupby('driver_gender').search_conducted.mean())

#%%
# Calculate the search rate for each combination of gender and violation
print(ri.groupby(['driver_gender', 'violation']).search_conducted.mean())

# Reverse the ordering to group by violation before gender
print(ri.groupby(['violation', 'driver_gender']).search_conducted.mean())

#%% 2.4 Does gender affect who is frisked during a search?
ri.search_conducted.value_counts()

ri.search_type.value_counts(dropna=False)

ri.search_type.value_counts()

ri['inventory'] = ri.search_type.str.contains('Inventory', na=False)
ri.inventory.sum()
ri.inventory.mean()

searched = ri[ri.search_conducted == True]
searched.inventory.mean()

#%%
# Count the 'search_type' values
print(ri.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)

# Check the data type of 'frisk'
print(ri.frisk.dtype)

# Take the sum of 'frisk'
print(ri.frisk.sum())

#%%
# Create a DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())

#%% 3. Visual exploratory data analysis
ri.index.month
ri.dtypes

monthly_arrest = ri.groupby(ri.index.month).is_arrested.mean()

import matplotlib.pyplot as plt

monthly_arrest.plot()
plt.xlabel('Month')
plt.ylabel('Arrests')
plt.title('Monthly mean arrests')

#%%
# Calculate the overall arrest rate
print(ri.is_arrested.mean())

# Calculate the hourly arrest rate
print(ri.groupby(ri.index.hour).is_arrested.mean())

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()

#%%
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()

#%% 3.1 Are drug-related stops on the rise?
monthly_arrests = ri.is_arrested.resample('M').sum()
monthly_drugs = ri.drugs_related_stop.resample('M').sum()

monthly = pd.concat([monthly_arrests, monthly_drugs], axis='columns')

monthly.plot()
plt.show()

monthly.plot(subplots=True)
plt.show()

#%%
# Calculate the annual rate of drug-related stops
print(ri.drugs_related_stop.resample('A').mean())

# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()

# Display the plot
plt.show()

#%%
# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate, annual_search_rate], axis='columns')

# Create subplots from 'annual'
annual.plot(subplots=True)

# Display the subplots
plt.show()

#%% 3.2 What violations are caught in each district?
table = pd.crosstab(ri.driver_race,
                    ri.driver_gender)

table = table.loc['Asian':'Hispanic']

table.plot(kind='bar')
plt.show()

table.plot(kind='bar', stacked=True)
plt.show()

#%%
# Create a frequency table of districts and violations
print(pd.crosstab(ri.district, ri.violation))

# Save the frequency table as 'all_zones'
all_zones = pd.crosstab(ri.district, ri.violation)

# Select rows 'Zone K1' through 'Zone K3'
print(all_zones.loc['Zone K1':'Zone K3'])

# Save the smaller table as 'k_zones'
k_zones = all_zones.loc['Zone K1':'Zone K3']

#%%
# Create a bar plot of 'k_zones'
k_zones.plot(kind='bar')

# Display the plot
plt.show()

#%%
# Create a stacked bar plot of 'k_zones'
k_zones.plot(kind='bar', stacked=True)

# Display the plot
plt.show()

#%% 3.3 How long might you be stopped for a violation?
mapping = {'up': True, 'down': False}
apple['is_up'] = apple.change.map(mapping)
apple.is_up.mean()

search_rate = ri.groupby('violation').search_conducted.mean()

search_rate.plot(kind='bar')
plt.show()

search_rate.sort_values().plot(kind='bar')
plt.show()

search_rate.sort_values().plot(kind='barh')
plt.show()

#%%
# Print the unique values in 'stop_duration'
print(ri.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min': 8, '16-30 Min': 23, '30+ Min': 45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(ri.stop_minutes.unique())

#%%
# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(ri.groupby('violation_raw').stop_minutes.mean())

# Save the resulting Series as 'stop_length'
stop_length = ri.groupby('violation_raw').stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind='barh')

# Display the plot
plt.show()

#%% 4. Analyzing the effect of weather on policing

#%% 4.1 Exploring the weather dataset
weather = pd.read_csv('data/24/weather.csv')

weather.head(3)

weather[['AWND', 'WSF2']].head(3)
weather[['AWND', 'WSF2']].describe()

weather[['AWND', 'WSF2']].plot(kind='box')
plt.show()

weather['WDIFF'] = weather.WSF2 - weather.AWND

weather.WDIFF.plot(kind='hist')
plt.show()

weather.WDIFF.plot(kind='hist', bins=20)
plt.show()

#%%
# Read 'weather.csv' into a DataFrame named 'weather'
weather = pd.read_csv('weather.csv')

# Describe the temperature columns
print(weather[['TMIN', 'TAVG', 'TMAX']].describe())

# Create a box plot of the temperature columns
weather[['TMIN', 'TAVG', 'TMAX']].plot(kind='box')

# Display the plot
plt.show()

#%%
# Create a 'TDIFF' column that represents temperature difference
weather['TDIFF'] = weather.TMAX - weather.TMIN

# Describe the 'TDIFF' column
print(weather.TDIFF.describe())

# Create a histogram with 20 bins to visualize 'TDIFF'
weather.TDIFF.plot(kind='hist', bins=20)

# Display the plot
plt.show()

#%% 4.2 Categorizing the weather
weather.shape

weather.columns

temp = weather.loc[:, 'TAVG':'TMAX']
temp.shape
temp.columns

temp.head()
temp.sum()
temp.sum(axis='columns').head()

ri.stop_duration.unique()
mapping = {'0-15 Min': 'short',
           '16-30 Min': 'medium',
           '30+ Min': 'long'}

ri['stop_lenght'] = ri.stop_duration.map(mapping)
ri.stop_lenght.dtype
ri.stop_lenght.unique()
ri.stop_lenght.memory_usage(deep=True)

cats = pd.CategoricalDtype(['short', 'medium', 'long'], ordered=True)

ri['stop_lenght'] = ri.stop_duration.astype(cats)

ri.stop_lenght.head()

ri[ri.stop_lenght > 'short'].shape

ri.groupby('stop_lenght').is_arrested.mean()

#%%
# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[:, 'WT01':'WT22']

# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis='columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Create a histogram to visualize 'bad_conditions'
weather.bad_conditions.plot(kind='hist')

# Display the plot
plt.show()

#%%
# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {0: 'good',
           1: 'bad', 2: 'bad', 3: 'bad', 4: 'bad',
           5: 'worse', 6: 'worse', 7: 'worse', 8: 'worse', 9: 'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping)

# Count the unique values in 'rating'
print(weather.rating.value_counts())

#%%
# Specify the logical order of the weather ratings
cats = pd.CategoricalDtype(['good', 'bad', 'worse'], ordered=True)

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype(cats)

# Examine the head of 'rating'
print(weather.rating.head())

#%%
# Reset the index of 'ri'
ri.reset_index(inplace=True)

# Examine the head of 'ri'
print(ri.head())

# Create a DataFrame from the 'DATE' and 'rating' columns
weather_rating = weather[['DATE', 'rating']]

# Examine the head of 'weather_rating'
print(weather_rating.head())

#%%
# Examine the shape of 'ri'
print(ri.shape)

# Merge 'ri' and 'weather_rating' using a left join
ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

# Examine the shape of 'ri_weather'
print(ri_weather.shape)

# Set 'stop_datetime' as the index of 'ri_weather'
ri_weather.set_index('stop_datetime', inplace=True)

#%% 4.3 Does weather affect the arrest rate?
search_rate = ri.groupby(['violation', 'driver_gender']).search_conducted.mean()
search_rate

search_rate.loc['Equipment']
search_rate.loc['Equipment', 'M']

search_rate.unstack()

ri.pivot_table(index='violation',
               columns='driver_gender',
               values='search_conducted')

#%%
# Calculate the overall arrest rate
print(ri_weather.is_arrested.mean())

# Calculate the arrest rate for each 'rating'
print(ri_weather.groupby('rating').is_arrested.mean())

# Calculate the arrest rate for each 'violation' and 'rating'
print(ri_weather.groupby(['violation', 'rating']).is_arrested.mean())

#%%
# Save the output of the groupby operation from the last exercise
arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

# Print the 'arrest_rate' Series
print(arrest_rate)

# Print the arrest rate for moving violations in bad weather
print(arrest_rate.loc['Moving violation', 'bad'])

# Print the arrest rates for speeding violations in all three weather conditions
print(arrest_rate.loc['Speeding'])

#%%
# Unstack the 'arrest_rate' Series into a DataFrame
print(arrest_rate.unstack())

# Create the same DataFrame using a pivot table
print(ri_weather.pivot_table(index='violation', columns='rating', values='is_arrested'))