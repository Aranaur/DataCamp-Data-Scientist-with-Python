#%% 1. The most Nobel of Prizes

# Loading in required libraries
import pandas as pd
import seaborn as sns
import numpy as np

# Reading in the Nobel Prize data
nobel = pd.read_csv('data/14/archive.csv')

nobel.columns = nobel.columns.str.lower().str.replace(' ', '_')

# Taking a look at the first several winners
nobel.head(n=6)

#%% 2. So, who gets the Nobel Prize?

# Display the number of (possibly shared) Nobel Prizes handed
# out between 1901 and 2016
display(len(nobel))

# Display the number of prizes won by male and female recipients.
nobel['sex'].value_counts()

# Display the number of prizes won by the top 10 nationalities.
nobel.value_counts('birth_country').head(n=10)

#%% 3. USA dominance

# Calculating the proportion of USA born winners per decade
nobel['usa_born_winner'] = np.where(nobel['birth_country'] == "United States of America", True, False)
nobel['decade'] = np.floor(np.floor(nobel['year'] / 10) * 10).astype(np.int64)
prop_usa_winners = nobel.groupby('decade', as_index=False)['usa_born_winner'].mean()

# Display the proportions of USA born winners per decade
prop_usa_winners

#%% 4. USA dominance, visualized

# Setting the plotting theme
sns.set()
# and setting the size of all plots.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [11, 7]

# Plotting USA born winners
ax = sns.lineplot(data=prop_usa_winners,
                  x='decade',
                  y='usa_born_winner')

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

#%% 5. What is the gender of a typical Nobel Prize winner?¶

# Calculating the proportion of female laureates per decade
nobel['female_winner'] = np.where(nobel['sex'] == 'Female', True, False)
prop_female_winners = nobel.groupby(['decade', 'category'], as_index=False)['female_winner'].mean()

sns.set()
ax = sns.lineplot(data=prop_female_winners,
                  x='decade',
                  y='female_winner',
                  hue='category')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

#%% 6. The first woman to win the Nobel Prize

# Picking out the first woman to win a Nobel Prize
nobel[nobel['sex'] == 'Female'].nsmallest(1, 'year')

#%% 7. Repeat laureates

# Selecting the laureates that have received 2 or more prizes.
nobel.groupby('full_name').filter(lambda x: x.value_counts('year') > 1)

nobel.groupby('full_name').value_counts('year')