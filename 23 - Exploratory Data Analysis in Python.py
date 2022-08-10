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
height = brfss['HTM4']
weight = brfss['WTKG3']

plt.scatter(x=height, y=weight)
plt.show()

plt.plot(height, weight, 'o', alpha=0.02, markersize=1)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

height_jitter = height + np.random.normal(0, 2, size=len(brfss))
weight_jitter = weight + np.random.normal(0, 2, size=len(brfss))

plt.plot(height_jitter, weight_jitter, 'o', alpha=0.02, markersize=1)
plt.axis([140, 200, 0, 160])
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

# sns.scatterplot('HTM4', 'WTKG3', data=brfss, alpha=0.02, x_jitter=0.9, y_jitter=0.9, size=0.01)

#%%
# Extract age
age = brfss['AGE']

# Plot the PMF
pmf_age = Pmf(age)
pmf_age.bar()

# Label the axes
plt.xlabel('Age in years')
plt.ylabel('PMF')
plt.show()

#%%
# Select the first 1000 respondents
brfss = brfss[:1000]

# Extract age and weight
age = brfss['AGE']
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', alpha=0.1)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')

plt.show()

#%%
# Select the first 1000 respondents
brfss = brfss[:1000]

# Add jittering to age
age = brfss['AGE'] + np.random.normal(0, 2.5, size=len(brfss))
# Extract weight
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', markersize=5, alpha=0.2)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')
plt.show()

#%% 3.2 Visualizing relationships
age = brfss['AGE'] + np.random.normal(0, 0.5, size=len(brfss))
weight = brfss['WTKG3'] + np.random.normal(0, 2, size=len(brfss))
plt.plot(age, weight, 'o', markersize=1, alpha=0.2)
plt.axis([15, 95, 0, 150])
plt.xlabel('Age in years')
plt.ylabel('Weight in kg')
plt.show()

data = brfss.dropna(subset=['AGE', 'WTKG3'])

sns.violinplot('AGE', 'WTKG3', data=data, inner=None)
plt.show()

sns.boxplot('AGE', 'WTKG3', data=data, whis=10)
plt.yscale('log')
plt.show()

#%%
# Drop rows with missing data
data = brfss.dropna(subset=['_HTMG10', 'WTKG3'])

# Make a box plot
sns.boxplot(x='_HTMG10', y='WTKG3', data=data, whis=10)

# Plot the y-axis on a log scale
plt.yscale('log')

# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

#%%
# Extract income
income = brfss['INCOME2']

# Plot the PMF
Pmf(income).bar()

# Label the axes
plt.xlabel('Income level')
plt.ylabel('PMF')
plt.show()

#%%
# Drop rows with missing data
data = brfss.dropna(subset=['INCOME2', 'HTM4'])

# Make a violin plot
sns.violinplot(x='INCOME2', y='HTM4', data=data, inner=None)

# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Income level')
plt.ylabel('Height in cm')
plt.show()

#%% 2.3 Correlation
columns = ['HTM4', 'WTKG3', 'AGE']
subset = brfss[columns]

subset.corr()

xs = np.linspace(-1, 1)
ys = xs**2
ys += np.random.normal(0, 0.05, len(xs))

plt.plot(xs, ys, 'o')
plt.show()

np.corrcoef(xs, ys)

#%%
# Select columns
columns = ['AGE', 'INCOME2', '_VEGESU1']
subset = brfss[columns]

# Compute the correlation matrix
print(subset.corr())

#%% 3.3 Simple regression
from scipy.stats import linregress

res = linregress(xs, ys)
print(res)

fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(xs, ys, 'o')
plt.plot(fx, fy, '-')

#%%
subset = brfss.dropna(subset=['WTKG3', 'HTM4'])

xs = subset['HTM4']
ys = subset['WTKG3']
res = linregress(xs, ys)

fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-')
plt.plot(height_jitter, weight_jitter, 'o', alpha=0.02, markersize=1)
plt.axis([140, 200, 0, 160])
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

#%%
from scipy.stats import linregress

# Extract the variables
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']

# Compute the linear regression
res = linregress(xs, ys)
print(res)

#%%
# Plot the scatter plot
plt.clf()
x_jitter = xs + np.random.normal(0, 0.15, len(xs))
plt.plot(x_jitter, ys, 'o', alpha=0.2)

# Plot the line of best fit
fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-', alpha=0.7)

plt.xlabel('Income code')
plt.ylabel('Vegetable servings per day')
plt.ylim([0, 6])
plt.show()

#%% 4. Multivariate Thinking

#%% 4.1 Limits of simple regression
