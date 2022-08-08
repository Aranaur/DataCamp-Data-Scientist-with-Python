# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.interpolate
import statsmodels.formula.api as smf

# Importing course datasets as DataFrames
brfss = pd.read_hdf('data/23/brfss.hdf5', 'brfss') # Behavioral Risk Factor Surveillance System (BRFSS)
gss = pd.read_hdf('data/23/gss.hdf5', 'gss') # General Social Survey (GSS)
nsfg = pd.read_hdf('data/23/nsfg.hdf5', 'nsfg') # National Survey of Family Growth (NSFG)

brfss.head() # Display the first five rows

#%% 1. Read, clean, and validate


#%% 1.1 DataFrames and Series
type(nsfg)

nsfg.head()
nsfg.shape
nsfg.columns

nsfg.birthwgt_lb1.value_counts()

nsfg['birthwgt_lb1']
type(nsfg['birthwgt_lb1'])

nsfg.birthwgt_lb1
type(nsfg.birthwgt_lb1)

nsfg.birthwgt_oz1.value_counts()

#%%
# Display the number of rows and columns
nsfg.shape

# Display the names of the columns
nsfg.columns

# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']

# Print the first 5 elements of ounces
print(ounces.head())

#%% 1.2 Clean and Validate
pounds = nsfg['birthwgt_lb1']
ounces = nsfg['birthwgt_oz1']

pounds.value_counts().sort_index()

pounds.describe()

pounds = pounds.replace([98, 99], np.nan)
pounds.mean()

ounces.replace([98, 99], np.nan, inplace=True)

birth_weight = pounds + ounces / 16
birth_weight.describe()

#%%
nsfg['outcome'].value_counts().sort_index()

#%%
# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace(8, np.nan, inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())

#%%
# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())

#%% 1.3 Filter and visualize
plt.hist(birth_weight.dropna(), bins=30)
plt.xlabel('Birth weight (lb)')
plt.ylabel('Fraction of births')
plt.show()

preterm = nsfg['prglngth'] < 37
preterm.sum()
preterm.mean()

preterm_weight = birth_weight[preterm]
preterm_weight.sum()
preterm_weight.mean()

full_term_weight = birth_weight[~preterm]
full_term_weight.sum()
full_term_weight.mean()