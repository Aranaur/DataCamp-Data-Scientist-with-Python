# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.interpolate
import statsmodels.formula.api as smf

# Importing course datasets as DataFrames
brfss = pd.read_hdf('data/23/brfss.hdf5', 'brfss')  # Behavioral Risk Factor Surveillance System (BRFSS)
gss = pd.read_hdf('data/23/gss.hdf5', 'gss')  # General Social Survey (GSS)
nsfg = pd.read_hdf('data/23/nsfg.hdf5', 'nsfg')  # National Survey of Family Growth (NSFG)

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

#%%
# Plot the histogram
plt.hist(agecon, bins=20)

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

#%%
# Plot the histogram
plt.hist(agecon, bins=20, histtype='step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()

#%%
# Create a Boolean Series for full-term babies
full_term = nsfg['prglngth'] >= 37

# Select the weights of full-term babies
full_term_weight = birth_weight[full_term]

# Compute the mean weight of full-term babies
print(full_term_weight.mean())

#%%
# Filter full-term babies
full_term = nsfg['prglngth'] >= 37

# Filter single births
single = nsfg['nbrnaliv'] == 1

# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[full_term & single]
print('Single full-term mean:', single_full_term_weight.mean())

# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[full_term & ~single]
print('Multiple full-term mean:', mult_full_term_weight.mean())

#%% 2. Distributions

from empiricaldist import Pmf, Cdf

#%% 2.1 Probability mass functions
gss = pd.read_hdf('data/23/gss.hdf5')

educ = gss['educ']

plt.hist(educ.dropna(), label='educ', bins=21, edgecolor='k')
plt.show()

pmf_educ = Pmf(educ, normalize=False)
pmf_educ.head()
pmf_educ[12]

pmf_educ = Pmf(educ, normalize=True)
pmf_educ.head()
pmf_educ[12]

pmf_educ.bar(label='educ')
plt.xlabel('Years of education')
plt.ylabel('PMF')
plt.show()

#%%
# Compute the PMF for year
pmf_year = Pmf(gss['year'], normalize=False)

# Print the result
print(pmf_year)

#%%
# Select the age column
age = gss['age']

# Make a PMF of age
pmf_age = Pmf(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()

#%% 2.2 Cumulative distribution functions
cdf = Cdf(gss['age'].dropna())
cdf.plot()
plt.xlabel('Age')
plt.ylabel('CDF')
plt.show()

q = 51
p = cdf[q]
print(p)

p = 0.25
q = cdf.inverse(p)
print(q)

p = 0.75
q = cdf.inverse(p)
print(q)

#%%
# Select the age column
age = gss['age']

# Compute the CDF of age
cdf_age = Cdf(age)

# Calculate the CDF of 30
print(cdf_age[30])

#%%
# Calculate the 75th percentile
percentile_75th = cdf_income.inverse(0.75)

# Calculate the 25th percentile
percentile_25th = cdf_income.inverse(0.25)

# Calculate the interquartile range
iqr = percentile_75th - percentile_25th

# Print the interquartile range
print(iqr)

#%%
# Select realinc
income = gss['realinc']

# Make the CDF
cdf_income = Cdf(income)

# Plot it
cdf_income.plot()

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.show()

#%% 2.3 Comparing distributions
male = gss['sex'] == 1
age = gss['age']
male_age = age[male]
female_age = age[~male]

Pmf(male_age).plot(label='Male')
Pmf(female_age).plot(label='Female')

Cdf(male_age).plot(label='Male')
Cdf(female_age).plot(label='Female')

income = gss['realinc']
pre95 = gss['year'] < 1995

Pmf(income[pre95]).plot(label='Before 1995')
Pmf(income[~pre95]).plot(label='After 1995')

Cdf(income[pre95]).plot(label='Before 1995')
Cdf(income[~pre95]).plot(label='After 1995')

#%%
# Select educ
educ = gss['educ']

# Bachelor's degree
bach = (educ >= 16)

# Associate degree
assc = (educ >= 14) & (educ < 16)

# High school
high = (educ <= 12)
print(high.mean())

#%%
income = gss['realinc']

# Plot the CDFs
Cdf(income[high]).plot(label='High school')
Cdf(income[assc]).plot(label='Associate')
Cdf(income[bach]).plot(label='Bachelor')

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.legend()
plt.show()

#%% 2.4 Modeling distributions
sample = np.random.normal(size=1000)
Cdf(sample).plot()

from scipy.stats import norm

xs = np.linspace(-3, 3)
ys = norm(0, 1).cdf(xs)

plt.plot(xs, ys, color='g')
Cdf(sample).plot()

xs = np.linspace(-3, 3)
ys = norm(0, 1).pdf(xs)
plt.plot(xs, ys, color='g')

import seaborn as sns
sns.kdeplot(sample)

xs = np.linspace(-3, 3)
ys = norm(0, 1).pdf(xs)
plt.plot(xs, ys, color='g')
sns.kdeplot(sample)

#%%
# Extract realinc and compute its log
income = gss['realinc']
log_income = np.log10(income)

# Compute mean and standard deviation
mean = log_income.mean()
std = log_income.std()
print(mean, std)

# Make a norm object
from scipy.stats import norm
dist = np.random.normal(mean, std)

#%%
# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income
Cdf(log_income).plot()

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()

#%%
# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income).plot()

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show()


#%% 3. Relationships

#%% 3.1 Exploring relationships
