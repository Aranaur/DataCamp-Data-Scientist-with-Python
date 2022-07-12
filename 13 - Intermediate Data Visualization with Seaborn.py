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

sns.stripplot(data=grants,
              y='Region',
              x='Award_Amount',
              jitter=True)

sns.swarmplot(data=grants,
              y='Region',
              x='Award_Amount')

sns.boxplot(data=grants,
            y='Region',
            x='Award_Amount')

sns.violinplot(data=grants,
            y='Region',
            x='Award_Amount')

sns.boxenplot(data=grants,
               y='Region',
               x='Award_Amount')

sns.barplot(data=grants,
            y='Region',
            x='Award_Amount',
            hue='Model Selected')

sns.pointplot(data=grants,
            y='Region',
            x='Award_Amount',
            hue='Model Selected')

sns.countplot(data=grants,
              y='Region',
              hue='Model Selected')
#%%

# Create the stripplot
sns.stripplot(data=df,
              x='Award_Amount',
              y='Model Selected',
              jitter=True)

plt.show()

#%%

# Create and display a swarmplot with hue set to the Region
sns.swarmplot(data=df,
              x='Award_Amount',
              y='Model Selected',
              hue='Region')

plt.show()

#%%

# Create a boxplot
sns.boxplot(data=df,
            x='Award_Amount',
            y='Model Selected')

plt.show()
plt.clf()

#%%

# Create a violinplot with the husl palette
sns.violinplot(data=df,
               x='Award_Amount',
               y='Model Selected',
               palette='husl')

plt.show()
plt.clf()

#%%

# Create a boxenplot with the Paired palette and the Region column as the hue
sns.boxenplot(data=df,
              x='Award_Amount',
              y='Model Selected',
              palette='Paired',
              hue='Region')

plt.show()
plt.clf()

#%%

# Show a countplot with the number of models used with each region a different color
sns.countplot(data=df,
              y="Model Selected",
              hue="Region")

plt.show()
plt.clf()

#%%

# Create a pointplot and include the capsize in order to show caps on the error bars
sns.pointplot(data=df,
              y='Award_Amount',
              x='Model Selected',
              capsize=.1)

plt.show()
plt.clf()

#%%

# Create a barplot with each Region shown as a different color
sns.barplot(data=df,
            y='Award_Amount',
            x='Model Selected',
            hue='Region')

plt.show()
plt.clf()

#%% 3.2 Regression Plots

sns.regplot(data=bike_share,
            x='temp',
            y='total_rentals',
            marker='+')

sns.residplot(data=bike_share,
              x='temp',
              y='total_rentals')

sns.regplot(data=bike_share,
              x='temp',
              y='total_rentals',
              order=2)

sns.residplot(data=bike_share,
              x='temp',
              y='total_rentals',
              order=2)

sns.regplot(data=bike_share,
              x='mnth',
              y='total_rentals',
              x_jitter=0.1,
              order=2)

sns.regplot(data=bike_share,
            x='mnth',
            y='total_rentals',
            x_estimator=np.mean,
            order=2)

sns.regplot(data=bike_share,
            x='temp',
            y='total_rentals',
            x_bins=4)


#%%
# Display a regression plot for Tuition
sns.regplot(data=df,
            y='Tuition',
            x='SAT_AVG_ALL',
            marker='^',
            color='g')

plt.show()
plt.clf()

#%%

# Display the residual plot
sns.residplot(data=df,
              y='Tuition',
              x='SAT_AVG_ALL',
              color='g')

plt.show()
plt.clf()

#%%
# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df,
            y='Tuition',
            x='PCTPELL')

plt.show()
plt.clf()

#%%
# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df,
            y='Tuition',
            x='PCTPELL',
            x_bins=5)

plt.show()
plt.clf()

#%%
# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df,
            y='Tuition',
            x='PCTPELL',
            x_bins=5,
            order=2)

plt.show()
plt.clf()

#%%

# Create a scatter plot by disabling the regression line
sns.regplot(data=df,
            y='Tuition',
            x='UG',
            fit_reg=False)

plt.show()
plt.clf()

#%%

# Create a scatter plot and bin the data into 5 bins
sns.regplot(data=df,
            y='Tuition',
            x='UG',
            x_bins=5)

plt.show()
plt.clf()

#%%
# Create a regplot and bin the data into 8 bins
sns.regplot(data=df,
            y='Tuition',
            x='UG',
            x_bins=8)

plt.show()
plt.clf()

#%% 3.3 Matrix plots

df_crosstab = pd.crosstab(bike_share['mnth'], bike_share['weekday'],
            values=bike_share['total_rentals'], aggfunc='mean').round(0)

sns.heatmap(df_crosstab)

sns.heatmap(df_crosstab, annot=True, fmt="1", cmap='YlGnBu', cbar=False, linewidths=0.5)

sns.heatmap(df_crosstab, annot=True, fmt="1", cmap='YlGnBu', cbar=True, center=df_crosstab.loc[9, 6])

cols = ['total_rentals', 'temp', 'casual', 'hum', 'windspeed']
sns.heatmap(bike_share[cols].corr(), cmap='YlGnBu')

#%%
# Create a crosstab table of the data
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])
print(pd_crosstab)

# Plot a heatmap of the table
sns.heatmap(pd_crosstab)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

plt.show()

#%%
# Create the crosstab DataFrame
pd_crosstab = pd.crosstab(df["Group"], df["YEAR"])

# Plot a heatmap of the table with no color bar and using the BuGn palette
sns.heatmap(pd_crosstab, cbar=False, cmap="BuGn", linewidths=0.3)

# Rotate tick marks for visibility
plt.yticks(rotation=0)
plt.xticks(rotation=90)

#Show the plot
plt.show()
plt.clf()

#%% 4 Creating Plots on Data Aware Grids

#%% 4.1 Using FacetGrid, catplot and lmplot

g = sns.FacetGrid(college_data, col='HIGHDEG')
g.map(sns.boxplot, 'Tuition', order=['1', '2', '3', '4'])

sns.catplot(x='Tuition', data=college_data, col='HIGHDEG', kind='box')

g = sns.FacetGrid(college_data, col='HIGHDEG')
g.map(plt.scatter, 'Tuition', 'SAT_AVG_ALL')

sns.lmplot(data=college_data, x='Tuition', y='SAT_AVG_ALL', col='HIGHDEG', fit_reg=False)

sns.lmplot(data=college_data, x='Tuition', y='SAT_AVG_ALL', col='HIGHDEG', row='REGION')

#%%
# Create FacetGrid with Degree_Type and specify the order of the rows using row_order
g2 = sns.FacetGrid(df,
                   row="Degree_Type",
                   row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

# Map a pointplot of SAT_AVG_ALL onto the grid
g2.map(sns.pointplot, 'SAT_AVG_ALL')

# Show the plot
plt.show()
plt.clf()

#%%
# Create a factor plot that contains boxplots of Tuition values
sns.catplot(data=df,
            x='Tuition',
            kind='box',
            row='Degree_Type')

plt.show()
plt.clf()

#%%
# Create a facetted pointplot of Average SAT_AVG_ALL scores facetted by Degree Type
sns.catplot(data=df,
            x='SAT_AVG_ALL',
            kind='point',
            row='Degree_Type',
            row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

plt.show()
plt.clf()

#%%
# Create a FacetGrid varying by column and columns ordered with the degree_order variable
g = sns.FacetGrid(df, col="Degree_Type", col_order=degree_ord)

# Map a scatter plot of Undergrad Population compared to PCTPELL
g.map(plt.scatter, 'UG', 'PCTPELL')

plt.show()
plt.clf()

#%%
# Re-create the previous plot as an lmplot
sns.lmplot(data=df,
           x='UG',
           y='PCTPELL',
           col="Degree_Type",
           col_order=degree_ord)

plt.show()
plt.clf()

#%%
# Create an lmplot that has a column for Ownership, a row for Degree_Type and hue based on the WOMENONLY column
sns.lmplot(data=df,
           x='SAT_AVG_ALL',
           y='Tuition',
           col="Ownership",
           row='Degree_Type',
           row_order=['Graduate', 'Bachelors'],
           hue='WOMENONLY',
           col_order=inst_ord)

plt.show()
plt.clf()

#%% 4.2 Using PairGrid and pairplot

g = sns.PairGrid(college_data, vars=['PCTPELL', 'PCTFLOAN'])
g = g.map_diag(sns.histplot)
g = g.map_offdiag(sns.scatterplot)

sns.pairplot(college_data, vars=['PCTPELL', 'PCTFLOAN'], kind='reg', diag_kind='hist')

sns.pairplot(college_data.query("REGION < 3"),
             vars=['PCTPELL', 'PCTFLOAN', 'UG'],
             hue='REGION', palette='husl',
             plot_kws={'alpha': 0.5})

#%%
# Create a PairGrid with a scatter plot for fatal_collisions and premiums
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map(sns.scatterplot)

plt.show()
plt.clf()

#%%
# Create the same PairGrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map_diag(sns.histplot)
g3 = g2.map_offdiag(sns.scatterplot)

plt.show()
plt.clf()

#%%
# Create a pairwise plot of the variables using a scatter plot
sns.pairplot(data=df,
             vars=["fatal_collisions", "premiums"],
             kind='scatter')

plt.show()
plt.clf()

#%%
# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=df,
             vars=["fatal_collisions", "premiums"],
             kind='scatter',
             hue='Region',
             palette='RdBu',
             diag_kws={'alpha':.5})

plt.show()
plt.clf()

#%%
# Build a pairplot with different x and y variables
sns.pairplot(data=df,
             x=["fatal_collisions_speeding", "fatal_collisions_alc"],
             y=['premiums', 'insurance_losses'],
             kind='scatter',
             hue='Region',
             palette='husl')

plt.show()
plt.clf()

#%%
# Build a pairplot with different x and y variables
sns.pairplot(data=df,
             x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],
             y_vars=['premiums', 'insurance_losses'],
             kind='scatter',
             hue='Region',
             palette='husl')

plt.show()
plt.clf()

#%%
# plot relationships between insurance_losses and premiums
sns.pairplot(data=df,
             vars=["insurance_losses", "premiums"],
             kind='reg',
             palette='BrBG',
             diag_kind = 'kde',
             hue='Region')

plt.show()
plt.clf()

#%% 4.2 Using JointGrid and jointplot

g = sns.JointGrid(data=bike_share, x='temp', y='total_rentals', dropna=True)
g.plot(sns.regplot, sns.histplot)

g = sns.JointGrid(data=bike_share, x='temp', y='total_rentals', dropna=True)
g = g.plot_joint(sns.kdeplot)
g = g.plot_marginals(sns.kdeplot, shade=True)

sns.jointplot(data=bike_share, x='temp', y='total_rentals', kind='hex')

g = (sns.jointplot(x='temp',
                   y='total_rentals',
                   kind='scatter',
                   data=bike_share.query("mnth == 12 & total_rentals > 1000"))
     .plot_joint(sns.kdeplot))


#%%
# Build a JointGrid comparing humidity and total_rentals
sns.set_style("whitegrid")
g = sns.JointGrid(x="hum",
                  y="total_rentals",
                  data=df,
                  xlim=(0.1, 1.0))

g.plot(sns.regplot, sns.histplot)

plt.show()
plt.clf()

#%%
# Create a jointplot similar to the JointGrid
sns.jointplot(x="hum",
              y="total_rentals",
              kind='reg',
              data=df)

plt.show()
plt.clf()

#%%
# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp",
              y="total_rentals",
              kind='reg',
              data=df,
              order=2,
              xlim=(0, 1))

plt.show()
plt.clf()

#%%
# Plot a jointplot showing the residuals
sns.jointplot(x="temp",
              y="total_rentals",
              kind='resid',
              data=df,
              order=2)

plt.show()
plt.clf()

#%%
# Create a jointplot of temp vs. casual riders
# Include a kdeplot over the scatter plot
g = sns.jointplot(x="temp",
                  y="casual",
                  kind='scatter',
                  data=df,
                  marginal_kws=dict(bins=10))
g.plot_joint(sns.kdeplot)

plt.show()
plt.clf()

#%%
# Replicate the above plot but only for registered riders
g = sns.jointplot(x="temp",
                  y="registered",
                  kind='scatter',
                  data=df,
                  marginal_kws=dict(bins=10))
g.plot_joint(sns.kdeplot)

plt.show()
plt.clf()

#%% 4.3 Selecting Seaborn Plots
