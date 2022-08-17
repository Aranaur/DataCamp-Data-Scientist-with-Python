#%% 1. Summary Statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import iqr
from scipy.stats import uniform
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import expon

amir_deals = pd.read_csv('data/25/amir_deals.csv')
food_consumption = pd.read_csv('data/25/food_consumption.csv')
world_happiness = pd.read_csv('data/25/world_happiness.csv')

#%% 1.1 What is statistics?


#%% 1.2 Measures of center
statistics.mode(amir_deals['status'])

amir_deals[amir_deals['amount'] > amir_deals['amount'].mean()]['num_users'].agg([np.mean, np.median])

#%%
# Import numpy with alias np
import numpy as np

# Filter for Belgium
be_consumption = food_consumption[food_consumption['country'] == 'Belgium']

# Filter for USA
usa_consumption = food_consumption[food_consumption['country'] == 'USA']

# Calculate mean and median consumption in Belgium
print(be_consumption['consumption'].mean())
print(be_consumption['consumption'].median())

# Calculate mean and median consumption in USA
print(usa_consumption['consumption'].mean())
print(usa_consumption['consumption'].median())

#%%
# Import numpy as np
import numpy as np

# Subset for Belgium and USA only
be_and_usa = food_consumption[(food_consumption['country'] == 'Belgium') | (food_consumption['country'] == 'USA')]

# Group by country, select consumption column, and compute mean and median
print(be_and_usa.groupby('country')['consumption'].agg([np.mean, np.median]))

#%%
# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Histogram of co2_emission for rice and show plot
rice_consumption['co2_emission'].hist()
plt.show()

#%%
# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Calculate mean and median of co2_emission with .agg()
print(rice_consumption['co2_emission'].agg([np.mean, np.median]))

#%% 1.3 Measures of spread
dist = food_consumption['consumption'] - np.mean(food_consumption['consumption'])
sq_dist = dist ** 2
sum_sq_dist = np.sum(sq_dist)
variance = sum_sq_dist / (len(sq_dist) - 1)
variance

np.var(food_consumption['consumption'], ddof=1)

np.sqrt(np.var(food_consumption['consumption'], ddof=1))
np.std(food_consumption['consumption'], ddof=1)

dist = food_consumption['consumption'] - np.mean(food_consumption['consumption'])
np.mean(np.abs(dist))

np.quantile(food_consumption['consumption'], 0.5)
np.quantile(food_consumption['consumption'], [0, 0.25, 0.5, 0.75, 1])

plt.boxplot(food_consumption['consumption'])
plt.show()

np.quantile(food_consumption['consumption'], [0, 0.2, 0.4, 0.6, 0.8, 1])
np.quantile(food_consumption['consumption'], np.linspace(0, 1, 5))

np.quantile(food_consumption['consumption'], 0.75) - np.quantile(food_consumption['consumption'], 0.25)
iqr(food_consumption['consumption'])

lower_threshold = np.quantile(food_consumption['consumption'], 0.25) - 1.5 * iqr(food_consumption['consumption'])
upper_threshold = np.quantile(food_consumption['consumption'], 0.75) + 1.5 * iqr(food_consumption['consumption'])
food_consumption[(food_consumption['consumption'] < lower_threshold) | (food_consumption['consumption'] > upper_threshold)]
food_consumption['consumption'].describe()

#%%
# Calculate the quartiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 5)))

#%%
# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 6)))

#%%
print(np.quantile(food_consumption['co2_emission'], np.linspace(0, 1, 11)))

#%%
# Print variance and sd of co2_emission for each food_category
print(food_consumption.groupby('food_category')['co2_emission'].agg([np.var, np.std]))

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Create histogram of co2_emission for food_category 'beef'
plt.hist(food_consumption[food_consumption['food_category'] == 'beef']['co2_emission'])
# Show plot
plt.show()

# Create histogram of co2_emission for food_category 'eggs'
plt.hist(food_consumption[food_consumption['food_category'] == 'eggs']['co2_emission'])
# Show plot
plt.show()

#%%
# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()

# Compute the first and third quantiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3 - q1

# Calculate the lower and upper cutoffs for outliers
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Subset emissions_by_country to find outliers
outliers = emissions_by_country[(emissions_by_country < lower) | (emissions_by_country > upper)]
print(outliers)

#%% 2. Random Numbers and Probability

#%% 2.1 What are the chances?
np.random.seed(10)
food_consumption['country'].sample()

np.random.seed(10)
food_consumption['country'].sample(2)

np.random.seed(10)
food_consumption['country'].sample(5, replace=True)

#%%
# Count the deals for each product
counts = amir_deals['product'].value_counts()

# Calculate probability of picking a deal with each product
probs = counts / counts.sum()
print(probs)

#%%
# Set random seed
np.random.seed(24)

# Sample 5 deals without replacement
sample_without_replacement = amir_deals.sample(5)
print(sample_without_replacement)

#%%
# Set random seed
np.random.seed(24)

# Sample 5 deals with replacement
sample_with_replacement = amir_deals.sample(5, replace=True)
print(sample_with_replacement)

#%% 2.2 Discrete distributions
# Create a histogram of restaurant_groups and show plot
restaurant_groups['group_size'].hist(bins=[2, 3, 4, 5, 6])
plt.show()

#%%
# Create probability distribution
size_dist = restaurant_groups['group_size'] / restaurant_groups['group_size'].sum()

# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

print(size_dist)

#%%
# Create probability distribution
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0]
# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

# Calculate expected value
expected_value = (size_dist['group_size'] * size_dist['prob']).sum()
print(expected_value)

#%%
# Create probability distribution
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0]
# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

# Expected value
expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])

# Subset groups of size 4 or more
groups_4_or_more = size_dist[size_dist['group_size'] >= 4]

# Sum the probabilities of groups_4_or_more
prob_4_or_more = groups_4_or_more['prob'].sum()
print(prob_4_or_more)

#%% 2.3 Continuous distributions
uniform.cdf(7, 0, 12)

1 - uniform.cdf(7, 0, 12)

uniform.cdf(7, 0, 12) - uniform.cdf(4, 0, 12)

uniform.rvs(0, 5, size=10)

#%%
# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
from scipy.stats import uniform

# Calculate probability of waiting less than 5 mins
prob_less_than_5 = uniform.cdf(5, min_time, max_time)
print(prob_less_than_5)

#%%
# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
from scipy.stats import uniform

# Calculate probability of waiting more than 5 mins
prob_greater_than_5 = 1 - uniform.cdf(5, min_time, max_time)
print(prob_greater_than_5)

#%%
# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
from scipy.stats import uniform

# Calculate probability of waiting 10-20 mins
prob_between_10_and_20 = uniform.cdf(20, min_time, max_time) - uniform.cdf(10, min_time, max_time)
print(prob_between_10_and_20)

#%%
# Set random seed to 334
np.random.seed(334)

# Import uniform
from scipy.stats import uniform

# Generate 1000 wait times between 0 and 30 mins
wait_times = uniform.rvs(0, 30, size=1000)

# Create a histogram of simulated times and show plot
plt.hist(wait_times)
plt.show()

#%% 2.4 The binomial distribution
# binom.rvs(# of coins, prob of heads, size=# of trials)

# Flip 1 coin with 50% chance of success 8 time
binom.rvs(1, 0.5, size=8)

# Flip 8 coin with 50% chance of success 8 time
binom.rvs(8, 0.5, size=8)

# Flip 10 coin with 25% chance of success 10 time
binom.rvs(3, 0.25, size=10)

# Prob of 7 heads?
# P(heads = 7)?
# binom.pmf(num heads, num trails, prob of heads)
binom.pmf(7, 10, 0.5)

# Prob of 7 or fewer heads?
# P(heads <= 7)?
binom.cdf(7, 10, 0.5)

# P(heads > 7)?
1 - binom.cdf(7, 10, 0.5)

#%%
# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

#%%
# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

# Simulate a single deal
print(binom.rvs(1, 0.3, size=1))

#%%
# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

# Simulate 1 week of 3 deals
print(binom.rvs(3, 0.3, size=1))

#%%
# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

# Simulate 52 weeks of 3 deals
deals = binom.rvs(3, 0.3, size=52)

# Print mean deals won per week
print(np.mean(deals))

#%%
# Probability of closing 3 out of 3 deals
prob_3 = binom.pmf(3, 3, 0.3)

print(prob_3)

#%%
# Probability of closing <= 1 deal out of 3 deals
prob_less_than_or_equal_1 = binom.cdf(1, 3, 0.3)

print(prob_less_than_or_equal_1)

#%%
# Probability of closing > 1 deal out of 3 deals
prob_greater_than_1 = 1 - binom.cdf(1, 3, 0.3)

print(prob_greater_than_1)

#%%
# Expected number won with 30% win rate
won_30pct = 3 * 0.3
print(won_30pct)

# Expected number won with 25% win rate
won_25pct = 3 * 0.25
print(won_25pct)

# Expected number won with 35% win rate
won_35pct = 3 * 0.35
print(won_35pct)

#%% 3 More Distributions and the Central Limit Theorem

#%% 3.1 The normal distribution
# 1 sd = 68%
# 2 sd = 95%
# 3 sd = 99.7%

# How many women are shorter then 154cm?
# what, mean, sd
norm.cdf(154, 161, 7)

# How many women are taller then 154cm?
1 - norm.cdf(154, 161, 7)

# Percent of women are 154-157cm?
norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)

# What height are 90% of women shorter then?
norm.ppf(0.9, 161, 7)

# What height are 90% of women taller then?
norm.ppf((1-0.9), 161, 7)

# Generator norm dist
norm.rvs(161, 7, size=10)

#%%
# Histogram of amount with 10 bins and show plot
plt.hist(amir_deals['amount'], bins=10)
plt.show()

# Histogram of amount with 10 bins and show plot
amir_deals.hist('amount', bins=10)
plt.show()

# Histogram of amount with 10 bins and show plot
amir_deals.amount.hist(bins=10)
plt.show()

# Histogram of amount with 10 bins and show plot
amir_deals['amount'].hist(bins=10)
plt.show()

#%%
# Probability of deal < 7500
prob_less_7500 = norm.cdf(7500, 5000, 2000)

print(prob_less_7500)

#%%
# Probability of deal > 1000
prob_over_1000 = 1 - norm.cdf(1000, 5000, 2000)

print(prob_over_1000)

#%%
# Probability of deal between 3000 and 7000
prob_3000_to_7000 = norm.cdf(7000, 5000, 2000) - norm.cdf(3000, 5000, 2000)

print(prob_3000_to_7000)

#%%
# Calculate amount that 25% of deals will be less than
pct_25 = norm.ppf(0.25, 5000, 2000)

print(pct_25)

#%%
# Calculate new average amount
new_mean = 5000 * 1.2

# Calculate new standard deviation
new_sd = 2000 * 1.3

# Simulate 36 new sales
new_sales = norm.rvs(new_mean, new_sd, size=36)

# Create histogram and show
plt.hist(new_sales)
plt.show()

#%% 3.2 The central limit theorem
die = pd.Series([1, 2, 3, 4, 5, 6])
# Roll 5 times
samp_5 = die.sample(5, replace=True)
print(samp_5)

np.mean(samp_5)

#%%
sample_means = []
for i in range(10):
    samp_5 = die.sample(5, replace=True)
    sample_means.append(np.mean(samp_5))

sample_means

plt.hist(sample_means)
plt.show()

#%%
sample_means = []
for i in range(100):
    samp_5 = die.sample(5, replace=True)
    sample_means.append(np.mean(samp_5))

plt.hist(sample_means)
plt.show()

#%%
sample_means = []
for i in range(1000):
    samp_5 = die.sample(5, replace=True)
    sample_means.append(np.mean(samp_5))

plt.hist(sample_means)
plt.show()

np.mean(sample_means)

#%%
sample_sds = []
for i in range(1000):
    samp_5 = die.sample(5, replace=True)
    sapmle_sds.append(np.std(samp_5))

plt.hist(sample_sds)
plt.show()

#%%
sales_team = pd.Series(['Amir', 'Brian', 'Claire', 'Damian'])
sales_team.sample(10, replace=True)

#%%
# Create a histogram of num_users and show
amir_deals['num_users'].hist()
plt.show()

#%%
# Set seed to 104
np.random.seed(104)

# Sample 20 num_users with replacement from amir_deals
samp_20 = amir_deals['num_users'].sample(20, replace=True)

# Take mean of samp_20
print(np.mean(samp_20))

#%%
# Set seed to 104
np.random.seed(104)

# Sample 20 num_users with replacement from amir_deals and take mean
samp_20 = amir_deals['num_users'].sample(20, replace=True)
np.mean(samp_20)

sample_means = []
# Loop 100 times
for i in range(100):
    # Take sample of 20 num_users
    samp_20 = amir_deals['num_users'].sample(20, replace=True)
    # Calculate mean of samp_20
    samp_20_mean = np.mean(samp_20)
    # Append samp_20_mean to sample_means
    sample_means.append(samp_20_mean)

print(sample_means)

#%%
# Set seed to 104
np.random.seed(104)

sample_means = []
# Loop 100 times
for i in range(100):
    # Take sample of 20 num_users
    samp_20 = amir_deals['num_users'].sample(20, replace=True)
    # Calculate mean of samp_20
    samp_20_mean = np.mean(samp_20)
    # Append samp_20_mean to sample_means
    sample_means.append(samp_20_mean)

# Convert to Series and plot histogram
sample_means_series = pd.Series(sample_means)
sample_means_series.hist()
# Show plot
plt.show()

#%%
# Set seed to 321
np.random.seed(321)

sample_means = []
# Loop 30 times to take 30 means
for i in range(30):
    # Take sample of size 20 from num_users col of all_deals with replacement
    cur_sample = all_deals['num_users'].sample(20, replace=True)
    # Take mean of cur_sample
    cur_mean = np.mean(cur_sample)
    # Append cur_mean to sample_means
    sample_means.append(cur_mean)

# Print mean of sample_means
print(np.mean(sample_means))

# Print mean of num_users in amir_deals
print(amir_deals['num_users'].mean())

#%% 3.3 The Poisson distribution
# P(5) in lambda 8
poisson.pmf(5, 8)

# P(<=5) in lambda 8
poisson.cdf(5, 8)

# P(>5) in lambda 8
1 - poisson.cdf(5, 8)

# P(>5) in lambda 10
1 - poisson.cdf(5, 10)

# Random Poisson
poisson.rvs(8, size=10)

#%%
# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of 5 responses
prob_5 = poisson.pmf(5, 4)

print(prob_5)

#%%
# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of 5 responses
prob_coworker = poisson.pmf(5, 5.5)

print(prob_coworker)

#%%
# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of 2 or fewer responses
prob_2_or_less = poisson.cdf(2, 4)

print(prob_2_or_less)

#%%
# Import poisson from scipy.stats
from scipy.stats import poisson

# Probability of > 10 responses
prob_over_10 = 1 - poisson.cdf(10, 4)

print(prob_over_10)

#%% 3.3 More probability distributions
# Exp dist
# lambda - time between action
# 1 / lambda = 1 req per min

# P(wait < 1min)
expon.cdf(1, scale=0.5)

# P(wait > 3min)
1 - expon.cdf(3, scale=0.5)

# P(1min < wait < 3min)
expon.cdf(3, scale=0.5) - expon.cdf(1, scale=0.5)

#%%
# Stud, t-dist

# Log-normal dist
# Length of chess games
# Adult blood pressure
# Num of hospitalization

#%%
# Import expon from scipy.stats
from scipy.stats import expon

# Print probability response takes < 1 hour
print(expon.cdf(1, scale=2.5))

#%%
# Import expon from scipy.stats
from scipy.stats import expon

# Print probability response takes > 4 hours
print(1 - expon.cdf(4, scale=2.5))

#%%
# Import expon from scipy.stats
from scipy.stats import expon

# Print probability response takes 3-4 hours
print(expon.cdf(4, scale=2.5) - expon.cdf(3, scale=2.5))

#%% 4. Correlation and Experimental Design


#%% 4.1
