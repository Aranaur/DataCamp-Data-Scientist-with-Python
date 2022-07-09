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

sns.displot(df['Award_Amount'], kde=True, bins=10)

sns.displot(df['Award_Amount'], kind='kde', rug=True, fill=True)

sns.displot(df['Award_Amount'], kind='ecdf', rug=True)

#%%

# Create a displot
sns.displot(df['Award_Amount'],
            bins=20)

# Display the plot
plt.show()

#%%

# Create a displot of the Award Amount
sns.displot(df['Award_Amount'],
            kind='kde',
            rug=True,
            fill=True)

# Plot the results
plt.show()

#%% 1.3 Regression Plots in Seaborn

sns.regplot(data=bike_share, x='temp', y='atemp')
sns.regplot(data=bike_share, x='temp', y='atemp')

sns.lmplot(data=bike_share, x='temp', y='atemp')
sns.lmplot(data=bike_share, x='temp', y='atemp', hue='mnth')
sns.lmplot(data=bike_share, x='temp', y='atemp', col='holiday')

#%%

# Create a regression plot of premiums vs. insurance_losses
sns.regplot(data=insurance, x='insurance_losses', y='premiums')

# Display the plot
plt.show()

#%%

# Create an lmplot of premiums vs. insurance_losses
sns.lmplot(data=insurance, x='insurance_losses', y='premiums')

# Display the second plot
plt.show()

#%%

# Create a regression plot using hue
sns.lmplot(data=insurance,
           x="insurance_losses",
           y="premiums",
           hue="Region")

# Show the results
plt.show()

#%%

# Create a regression plot with multiple rows
sns.lmplot(data=insurance,
           x="insurance_losses",
           y="premiums",
           row="Region")

# Show the plot
plt.show()

#%% 2. Customizing Seaborn Plots

#%% 2.1 Using Seaborn Styles
