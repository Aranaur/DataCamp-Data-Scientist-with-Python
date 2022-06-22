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