# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing course datasets as DataFrames
bike_share = pd.read_csv('data/13/bike_share.csv')
college_data = pd.read_csv('data/13/college_datav3.csv')
daily_show = pd.read_csv('data/13/daily_show_guests_cleaned.csv')
insurance = pd.read_csv('data/13/insurance_premiums.csv')
grants = pd.read_csv('data/13/schoolimprovement2010grants.csv', index_col=0)

bike_share.head()  # Display the first five rows of this DataFrame

#%% 1. Seaborn Introduction

#%% 1.1 Introduction to Seaborn

fig, ax = plt.subplots()
ax.hist(bike_share['temp'])

bike_share['temp'].plot.hist()

sns.histplot(bike_share['temp'])

sns.displot(bike_share['temp'], kind='kde')

#%%

grant_file = 'https://assets.datacamp.com/production/course_7030/datasets/schoolimprovement2010grants.csv'

# import all modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the DataFrame
df = pd.read_csv(grant_file)

#%%

# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()

# Clear out the pandas histogram
plt.clf()

#%%

# Display a Seaborn displot
sns.displot(df['Award_Amount'])
plt.show()

# Clear the displot
plt.clf()

#%% 1.2 Using the distribution plot

