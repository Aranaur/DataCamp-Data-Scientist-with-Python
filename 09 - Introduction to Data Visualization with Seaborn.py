# %% Importing course packages; you can add more too!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Importing course datasets as DataFrames

country_data = pd.read_csv('data/09/countries-of-the-world.csv', decimal=",")
mpg = pd.read_csv('data/09/mpg.csv')
student_data = pd.read_csv('data/09/student-alcohol-consumption.csv', index_col=0)
survey = pd.read_csv('data/09/young-people-survey-responses.csv', index_col=0)

survey.head()

# %% 1. Introduction to Seaborn

height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]

sns.scatterplot(x=height, y=weight)
plt.show()

gender = ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male']

sns.countplot(x=gender)
plt.show()

# %%
country_data.columns

gdp = country_data['GDP ($ per capita)']
phones = country_data['Phones (per 1000)']
percent_literate = country_data['Literacy (%)']
region = country_data['Region']

sns.scatterplot(x=gdp, y=phones)
plt.show()

sns.scatterplot(x=gdp, y=percent_literate)
plt.show()

sns.countplot(y=region)
plt.show()

# %% 1.1 Using pandas with Seaborn

sns.countplot(y='Region', data=country_data)
plt.show()

# %% 1.2 Adding a third variable with hue

tips = sns.load_dataset('tips')
tips.head()

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='smoker', hue_order=['No', 'Yes'])
plt.show()

hue_colors = {'Yes': 'black',
              'No': 'red'}
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='smoker', palette=hue_colors)
plt.show()

hue_colors = {'Yes': '#808080',
              'No': '#00ff00'}
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='smoker', palette=hue_colors)
plt.show()

sns.countplot(x='smoker',
              data=tips,
              hue='sex')
plt.show()

# %%

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3",
                data=student_data,
                hue="location",
                hue_order=['Rural', 'Urban'])
# Show plot
plt.show()

# %%

# Create a dictionary mapping subgroup values to colors
palette_colors = {'Rural': "green", 'Urban': "blue"}
# Create a count plot of school with location subgroups
sns.countplot(x='school', data=student_data, hue='location', palette=palette_colors)
# Display plot
plt.show()

# %% 2. Introduction to relational plots and subplots

sns.scatterplot(x='total_bill',
                y='tip',
                data=tips)
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            col='smoker')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            row='smoker')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            col='smoker',
            row='time')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            col='day',
            col_wrap=2)
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            col='day',
            col_wrap=2,
            col_order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.show()

# %%

# Change to use relplot() instead of scatterplot()
sns.relplot(x="absences", y="G3",
            data=student_data,
            kind='scatter')
# Show plot
plt.show()

# %%

# Change to make subplots based on study time
sns.relplot(x="absences", y="G3",
            data=student_data,
            kind="scatter",
            col='study_time')
# Show plot
plt.show()

# %%

# Change this scatter plot to arrange the plots in rows instead of columns
sns.relplot(x="absences", y="G3",
            data=student_data,
            kind="scatter",
            row="study_time")
# Show plot
plt.show()

# %%

# Create a scatter plot of G1 vs. G3
sns.relplot(x='G1',
            y='G3',
            data=student_data,
            kind='scatter')
# Show plot
plt.show()

# %%

# Adjust to add subplots based on school support
sns.relplot(x="G1", y="G3",
            data=student_data,
            kind="scatter",
            col='schoolsup',
            col_order=['yes', 'no'])
# Show plot
plt.show()

# %%

# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3",
            data=student_data,
            kind="scatter",
            col="schoolsup",
            col_order=["yes", "no"],
            row='famsup',
            row_order=['yes', 'no'])
# Show plot
plt.show()

# %% 2.1 Customizing scatter plots

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            size='size',
            hue='size')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            hue='smoker',
            style='smoker')
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            alpha=.4)
plt.show()

sns.relplot(x='total_bill',
            y='tip',
            data=tips,
            kind='scatter',
            size='size',
            hue='size',
            style='smoker',
            alpha=.4)
plt.show()

# %%

# Create scatter plot of horsepower vs. mpg
sns.relplot(x='horsepower', y='mpg', data=mpg, kind='scatter', size='cylinders')
# Show plot
plt.show()

# %%

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg",
            data=mpg, kind="scatter",
            size="cylinders",
            hue='cylinders')
# Show plot
plt.show()

# %%

# Create a scatter plot of acceleration vs. mpg
sns.relplot(x='acceleration', y='mpg', data=mpg, kind='scatter', hue='origin', style='origin')
# Show plot
plt.show()

# %% 2.2 Introduction to line plots

air_df_mean = pd.DataFrame({'hour': pd.Series(range(1, 6)),
                            'NO_2_mean': [13.375,
                                          30.041667,
                                          30.666667,
                                          20.416667,
                                          16.958333],
                            'location': ['East', 'North', 'East', 'North', 'East']})

sns.relplot(x='hour',
            y='NO_2_mean',
            data=air_df_mean,
            kind='line',
            style='location',
            hue='location',
            markers=True,
            dashes=False)
plt.show()

sns.relplot(x='hour',
            y='NO_2_mean',
            data=air_df_mean,
            kind='line',
            ci='sd') #None
plt.show()

# %%

# Create line plot
sns.relplot(x='model_year', y='mpg', data=mpg, kind='line')
# Show plot
plt.show()

# %%

# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg",
            data=mpg, kind="line", ci='sd')
# Show plot
plt.show()

# %%

# Create line plot of model year vs. horsepower
sns.relplot(x='model_year', y='horsepower', data=mpg, kind='line', ci=None)
# Show plot
plt.show()

# %%

# Change to create subgroups for country of origin
sns.relplot(x="model_year", y="horsepower",
            data=mpg, kind="line",
            ci=None,
            style='origin',
            hue='origin')
# Show plot
plt.show()

# %%

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower",
            data=mpg, kind="line",
            ci=None, style="origin",
            hue="origin",
            markers=True,
            dashes=False)
# Show plot
plt.show()

# %% 3. Count plots and bar plots