#%% 1. Summary Statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import iqr
from scipy.stats import uniform
from scipy.stats import binom

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
