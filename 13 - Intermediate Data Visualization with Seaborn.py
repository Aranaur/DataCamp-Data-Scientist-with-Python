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

sns.set()
insurance['insurance_losses'].plot.hist()

for style in ['white', 'dark', 'whitegrid', 'darkgrid']:
    sns.set_style(style)
    sns.displot(insurance['insurance_losses'])
    plt.show()

sns.set_style('white')
sns.displot(insurance['insurance_losses'])
sns.despine(left=True)

#%%

# Plot the pandas histogram
df['fmr_2'].plot.hist()
plt.show()
plt.clf()

# Set the default seaborn style
sns.set()

# Plot the pandas histogram again
df['fmr_2'].plot.hist()
plt.show()
plt.clf()

#%%

sns.set_style('dark')
sns.displot(df['fmr_2'])
plt.show()
plt.clf()

#%%

sns.set_style('whitegrid')
sns.displot(df['fmr_2'])
plt.show()
plt.clf()

#%%

# Set the style to white
sns.set_style('white')

# Create a regression plot
sns.lmplot(data=df,
           x='pop2010',
           y='fmr_2')

# Remove the spines
sns.despine(top=True, right=True)

# Show the plot and clear the figure
plt.show()
plt.clf()

#%% 2.2 Colors in Seaborn

sns.set(color_codes=True)
sns.displot(insurance['insurance_losses'], color='g')

palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
for p in palettes:
    sns.set_palette(p)
    sns.displot(insurance['insurance_losses'])

palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
for p in palettes:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()

sns.palplot(sns.color_palette('Paired', 10))

sns.palplot(sns.color_palette('Blues', 10))

sns.palplot(sns.color_palette('BrBG', 10))

#%%

# Set style, enable color code, and create a magenta displot
sns.set(color_codes=True)
sns.displot(df['fmr_3'], color='m')

# Show the plot
plt.show()

#%%

# Loop through differences between bright and colorblind palettes
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
    sns.displot(df['fmr_3'])
    plt.show()

    # Clear the plots
    plt.clf()

#%%

sns.palplot(sns.color_palette('Purples', 8))
plt.show()

sns.palplot(sns.color_palette('husl', 10))
plt.show()

sns.palplot(sns.color_palette('coolwarm', 6))
plt.show()

#%% 2.3 Customizing with matplotlib

fig, ax = plt.subplots()
sns.histplot(insurance['insurance_losses'], ax=ax)
ax.set(xlabel='Insurance Losses')

fig, ax = plt.subplots()
sns.histplot(insurance['insurance_losses'], ax=ax)
ax.set(xlabel='Insurance Losses',
       ylabel='Distribution', xlim=(0, 250),
       title='Insurance Losses and Distribution')

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7, 4))
sns.histplot(grants['Award_Amount'], stat='density', ax=ax0)
sns.histplot(grants.query('State == "AK"')['Award_Amount'], stat='density', ax=ax1)
ax1.set(xlabel='Tuition (NM)', xlim=(0, 70000))
ax1.axvline(x=20000, label='My Budget', linestyle='--')
ax1.legend

#%%

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of data
sns.histplot(df['fmr_3'], ax=ax)

# Create a more descriptive x axis label
ax.set(xlabel="3 Bedroom Fair Market Rent")

# Show the plot
plt.show()

#%%

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the distribution of 1 bedroom rents
sns.histplot(df['fmr_1'], ax=ax)

# Modify the properties of the plot
ax.set(xlabel="1 Bedroom Fair Market Rent",
       xlim=(100,1500),
       title="US Rent")

# Display the plot
plt.show()

#%%

# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.histplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=median, color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()

#%%

# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Plot the distribution of 1 bedroom apartments on ax0
sns.histplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500))

# Plot the distribution of 2 bedroom apartments on ax1
sns.histplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100,1500))

# Display the plot
plt.show()

#%% 3. Additional Plot Types

#%% 3.1 Categorical Plot Types
