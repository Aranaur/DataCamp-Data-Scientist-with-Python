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

inconsistent_categories = set(study_data['blood_type']).difference(categiroes['blood_type'])
print(inconsistent_categories)

inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
study_data[inconsistent_rows]
consistent_data = study_data[~inconsistent_rows]

#%%
# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")

#%%
# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])

#%% 2.2 Categorical variables

airlines['day'].value_counts()
airlines.groupby('day').count()

airlines['day'].str.upper()
airlines['day'].str.lower()
airlines['day'].str.strip()

group_names = ['0-200K', '200-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q=3, labels=group_names)
demographics[['income_group', 'household_income']]

ranges = [0, 200000, 500000, np.inf]
group_names = ['0-200K', '200-500K', '500K+']
demographics['income_group'] = pd.cut(demographics['household_income'], bins=ranges, labels=group_names)
demographics[['income_group', 'household_income']]

mapping = {'Microsoft': 'DesktopOS',
           'MacOS': 'DesktopOS',
           'Linux': 'DesktopOS',
           'IOS': 'MobileOS',
           'Android': 'MobileOS'}
devices['operating_system'] = devices['operating_system'].replace(mapping)
devices['operating_system'].unique()

#%%
# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

#%%

# Create ranges for categories
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'medium', 'long']

# Create wait_type column
airlines['wait_type'] = pd.cut(airlines['wait_min'], bins = label_ranges,
                               labels = label_names)

# Create mappings and replace
mappings = {'Monday':'weekday', 'Tuesday':'weekday', 'Wednesday': 'weekday',
            'Thursday': 'weekday', 'Friday': 'weekday',
            'Saturday': 'weekend', 'Sunday': 'weekend'}

airlines['day_week'] = airlines['day'].replace(mappings)

#%% 2.3 Cleaning text data

phones['Phone number'] = phones['Phone number'].str.replace('+', '00')
phones['Phone number'] = phones['Phone number'].str.replace('-', '')
digits = phones['Phone number'].str.len()
phones.loc[digits < 10, 'Phone number'] = np.nan

assert digits.min() >= 10
assert phones['Phone number'].str.contains('+|-').any() == False

phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')  # leave only numbers

ride_sharing['duration'].str.replace(r'\D+', '')

#%%
# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.", "")

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Mr.', '')

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Miss', '')

# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace('Ms.', '')

# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False

#%%
# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()

# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]

# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40

# Print new survey_response column
print(airlines_survey['survey_response'])

#%% 3. Advanced data problems

#%% 3.1 Uniformity

temp_fah = temp.loc[temp['Temp'] > 40, 'Temp']
temp_cels = (temp_fah - 32) * (5/9)
temp.loc[temp['Temp'] > 40, 'Temp'] = temp_cels

accert temp['temp'].max() < 40

birth['Birth'] = pd.to_datetime(birth['Birth'],
                                infer_datetime_format=True,
                                errors='coerce')

birth['Birth'] = birth['Birth'].dt.strftime('%d-%m-%Y')

#%%
# Find values of acct_cur that are equal to 'euro'
acct_eu = banking['acct_cur'] == 'euro'

# Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1

# Unify acct_cur column by changing 'euro' values to 'dollar'
banking.loc[banking['acct_cur'] == 'euro', 'acct_cur'] = 'dollar'

# Assert that only dollar currency remains
assert banking['acct_cur'].unique() == 'dollar'

#%%
# Print the header of account_opend
print(banking['account_opened'].head())

# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                           # Infer datetime format
                                           infer_datetime_format = True,
                                           # Return missing value for error
                                           errors = 'coerce')

# Get year of account opened
banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')

# Print acct_year
print(banking['acct_year'])

#%% 3.2 Cross field validation

sum_classes = flights[['eco', 'buis', 'first']].sum(axis=1)
passengers_equ = sum_classes == flight['total']
inconsistent_pass = flight[~passengers_equ]
consistent_pass = flight[passengers_equ]

users['Birth'] = pd.to_datetime(users['Birth'])
today = dt.date.today()
age_manual = today.year - users['Birth'].dt.year
age_equ = age_manual == users['Age']
inconsistent_age = users[~age_equ]
consistent_age = users[age_equ]

#%%
# Store fund columns to sum against
fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']

# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis=1) == banking['inv_amount']

# Store consistent and inconsistent data
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]

# Store consistent and inconsistent data
print("Number of inconsistent investments: ", inconsistent_inv.shape[0])

#%%
# Store today's date and find ages
today = dt.date.today()
ages_manual = today.year - banking['birth_date'].dt.year

# Find rows where age column == ages_manual
age_equ = ages_manual == banking['age']

# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]

# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])

#%% 3.3 Completeness

air.isna().sum()

msno.matrix(air)
plt.show()

missing = air[air['CO2'].isna()]
missing.describe()
complite = air[~air['CO2'].isna()]
complite.describe()

sorted_air = air.sort_values(by='Temp')
msno.matrix(sorted_air)
plt.show()

# Drop NA
air_drop = air.dropna(subset=['CO2'])
air_drop.head()

# Replace NA
co2_mean = air['CO2'].mean()
air_imputed = air.fillna({'CO2': co2_mean})
air_imputed.head()

#%%
# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]

# Sort banking by age and visualize
banking_sorted = banking.sort_values('age')
msno.matrix(banking_sorted)
plt.show()

#%%
# Drop missing values of cust_id
banking_fullid = banking.dropna(subset = ['cust_id'])

# Compute estimated acct_amount
acct_imp = banking_fullid['inv_amount'] * 5

# Impute missing acct_amount with corresponding acct_imp
banking_imputed = banking_fullid.fillna({'acct_amount':acct_imp})

# Print number of missing values
print(banking_imputed.isna().sum())

#%% 4. Record linkage

#%% 4.1 Comparing strings

# intention
# execution

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

fuzz.WRatio('Reeding', 'Reading')
fuzz.WRatio('Houston Rockets', 'Rockets')
fuzz.WRatio('Houston Rockets vs Los Angeles Lakers', 'Lakers vs Rockets')

string = 'Houston Rockets vs Los Angeles Lakers'
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets',
                     'Huston vs Lakers', 'Heat vs Bulls'])
process.extract(string, choices, limit=2)

print(restaurants_new['city'].unique())
categories = {'city': ['New York', 'Los Angeles']}
categories = pd.DataFrame(categories)
for city in categories['city']:
    matches = process.extract(city, restaurants_new['city'], limit=restaurants_new.shape[0])
    for potential_match in matches:
        if potential_match[1] >= 80:
            restaurants_new.loc[restaurants_new['city'] == potential_match[0], 'city'] = city

#%%
# Import process from fuzzywuzzy
from fuzzywuzzy import process

# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['cuisine_type'].unique()

# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit = len(unique_types)))

# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit = len(unique_types)))

# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit = len(unique_types)))

#%%
# Inspect the unique values of the cuisine_type column
print(restaurants['cuisine_type'].unique())

#%%
# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

# Iterate through the list of matches to italian
for match in matches:
    # Check whether the similarity score is greater than or equal to 80
    if match[1] >= 80:
        # Select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
        restaurants.loc[restaurants['cuisine_type'] == match[0]] = 'italian'

#%%
# Iterate through categories
for cuisine in categories:
    # Create a list of matches, comparing cuisine with the cuisine_type column
    matches = process.extract(cuisine, restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

    # Iterate through the list of matches
    for match in matches:
        # Check whether the similarity score is greater than or equal to 80
        if match[1] >= 80:
            # If it is, select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
            restaurants.loc[restaurants['cuisine_type'] == match[0]] = cuisine

# Inspect the final result
print(restaurants['cuisine_type'].unique())

#%% 4.2 Generating pairs

import recordlinkage

indexer = recordlinkage.Index()
indexer.block('state')
pairs = indexer.index(census_A, census_B)
compare_cl = recordlinkage.Compare()

compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label='state')

compare_cl.exact('surname', 'surname', threshold=0.85, label='surname')
compare_cl.exact('address_1', 'address_1', threshold=0.85, label='address_1')

potential_matches = compare_cl.compute(pairs, census_A, census_B)
potential_matches
potential_matches[potential_matches.sum(asix=1) >= 2]

#%%
# Create an indexer and object and find possible pairs
indexer = recordlinkage.Index()

# Block pairing on cuisine_type
indexer.block('cuisine_type')

# Generate pairs
pairs = indexer.index(restaurants, restaurants_new)

#%%
# Create a comparison object
comp_cl = recordlinkage.Compare()

#%%
# Create a comparison object
comp_cl = recordlinkage.Compare()

# Find exact matches on city, cuisine_types
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('cuisine_type', 'cuisine_type', label='cuisine_type')

# Find similar matches of rest_name
comp_cl.string('rest_name', 'rest_name', label='name', threshold = 0.8)

#%%
# Create a comparison object
comp_cl = recordlinkage.Compare()

# Find exact matches on city, cuisine_types -
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('cuisine_type', 'cuisine_type', label='cuisine_type')

# Find similar matches of rest_name
comp_cl.string('rest_name', 'rest_name', label='name', threshold = 0.8)

# Get potential matches and print
potential_matches = comp_cl.compute(pairs, restaurants, restaurants_new)
print(potential_matches)

#%%

potential_matches[potential_matches.sum(axis = 1) >= 3]

#%% 4.3 Linking DataFrames

matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
matches.index

dublicate_rows = matches.index.get_level_values(1)
census_B_index

census_B_duplicates = census_B[census_B.index.isin(dublicate_rows)]

full_census = sensus_A.append(census_B_new)

#%%
# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis=1) >= 3]

# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)

# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]

# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)