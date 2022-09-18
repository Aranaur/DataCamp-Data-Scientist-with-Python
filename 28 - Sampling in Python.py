import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from scipy.stats import norm

attrition_pop = pd.read_feather('data/28/attrition.feather')
spotify_population = pd.read_feather('data/28/spotify_2000_2020.feather')
coffee_ratings = pd.read_feather('data/28/coffee_ratings_full.feather')

# %% 1. Bias Any Stretch of the Imagination

# %% 1.1 Living the sample life
coffee_ratings.head()

np.random.seed(1987)

pts_vs_flavor_pop = coffee_ratings[['total_cup_points', 'flavor']]
pts_vs_flavor_samp = pts_vs_flavor_pop.sample(n=10)
pts_vs_flavor_samp

cup_points_samp = coffee_ratings['total_cup_points'].sample(n=10)

np.mean(pts_vs_flavor_pop['total_cup_points'])
np.mean(cup_points_samp)

pts_vs_flavor_pop['flavor'].mean()
pts_vs_flavor_samp['flavor'].mean()

# %%
# Sample 1000 rows from spotify_population
spotify_sample = spotify_population.sample(n=1000)

# Print the sample
print(spotify_sample)

# Calculate the mean duration in mins from spotify_population
mean_dur_pop = spotify_population.duration_minutes.mean()

# Calculate the mean duration in mins from spotify_sample
mean_dur_samp = spotify_sample.duration_minutes.mean()

# Print the means
print(mean_dur_pop)
print(mean_dur_samp)
# %%
# Create a pandas Series from the loudness column of spotify_population
loudness_pop = spotify_population['loudness']

# Sample 100 values of loudness_pop
loudness_samp = loudness_pop.sample(n=100)

# Calculate the mean of loudness_pop
mean_loudness_pop = np.mean(loudness_pop)

# Calculate the mean of loudness_samp
mean_loudness_samp = np.mean(loudness_samp)

# Print the means
print(mean_loudness_pop)
print(mean_loudness_samp)
# %% 1.2 A little too convenient
coffee_ratings['total_cup_points'].mean()

coffee_ratings_first10 = coffee_ratings.head(10)
coffee_ratings_first10['total_cup_points'].mean()

coffee_ratings['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()

coffee_ratings_first10['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()

coffee_sample = coffee_ratings.sample(n=10)
coffee_sample['total_cup_points'].hist(bins=np.arange(59, 93, 2))
plt.show()

# %%
# Visualize the distribution of acousticness with a histogram
spotify_population['acousticness'].hist(bins=np.arange(0, 1.01, 0.01))
plt.show()

# %%
# Update the histogram to use spotify_mysterious_sample
spotify_mysterious_sample['acousticness'].hist(bins=np.arange(0, 1.01, 0.01))
plt.show()

# Visualize the distribution of duration_minutes as a histogram
spotify_population['duration_minutes'].hist(bins=np.arange(0, 15.5, 0.05))
plt.show()

# %%
# Visualize the distribution of duration_minutes as a histogram
spotify_population['duration_minutes'].hist(bins=np.arange(0, 15.5, 0.5))
plt.show()

# Update the histogram to use spotify_mysterious_sample2
spotify_mysterious_sample2['duration_minutes'].hist(bins=np.arange(0, 15.5, 0.5))
plt.show()

# %% 1.3 How does Sue do sampling?
randoms = np.random.beta(a=2, b=2, size=5000)
randoms

plt.hist(randoms, bins=np.arange(0, 1, 0.05))
plt.show()

np.random.seed(20000229)
np.random.normal(loc=2, scale=1.5, size=2)
np.random.normal(loc=2, scale=1.5, size=2)

np.random.seed(20000229)
np.random.normal(loc=2, scale=1.5, size=2)
np.random.normal(loc=2, scale=1.5, size=2)

np.random.seed(20041004)
np.random.normal(loc=2, scale=1.5, size=2)
np.random.normal(loc=2, scale=1.5, size=2)
# %%
# Generate random numbers from a Uniform(-3, 3)
uniforms = np.random.uniform(low=-3, high=3, size=5000)

# Generate random numbers from a Normal(5, 2)
normals = np.random.normal(loc=5, scale=2, size=5000)

# Print normals
print(normals)

# Generate random numbers from a Uniform(-3, 3)
uniforms = np.random.uniform(low=-3, high=3, size=5000)

# Plot a histogram of uniform values, binwidth 0.25
plt.hist(uniforms, bins=np.arange(-3, 3.25, 0.25))
plt.show()

# Generate random numbers from a Normal(5, 2)
normals = np.random.normal(loc=5, scale=2, size=5000)

# Plot a histogram of normal values, binwidth 0.5
plt.hist(normals, bins=np.arange(-2, 13.5, 0.5))
plt.show()

# %%
import numpy as np

np.random.seed(123)
x = np.random.normal(size=5)
y = np.random.normal(size=5)

# %%
import numpy as np

np.random.seed(123)
x = np.random.normal(size=5)
np.random.seed(123)
y = np.random.normal(size=5)

# %%
import numpy as np

np.random.seed(123)
x = np.random.normal(size=5)
np.random.seed(456)
y = np.random.normal(size=5)

# %% 2. Don't get theory eyed

# %% 2.1 Simple is as simple does
coffee_ratings.sample(n=5, random_state=19000113)

sample_size = 5
pop_size = len(coffee_ratings)
interval = pop_size // sample_size

coffee_ratings.iloc[::interval]

coffee_ratings_with_id = coffee_ratings.reset_index()
coffee_ratings_with_id.plot(x='index',
                            y='aftertaste',
                            kind='scatter')
plt.show()

shuffled = coffee_ratings.sample(frac=1)
shuffled = shuffled.reset_index(drop=True).reset_index()
shuffled.plot(x='index',
              y='aftertaste',
              kind='scatter')
plt.show()
# %%
# Sample 70 rows using simple random sampling and set the seed
attrition_samp = attrition_pop.sample(n=70, random_state=18900217)

# Print the sample
print(attrition_samp)

# %%
# Set the sample size to 70
sample_size = 70

# Calculate the population size from attrition_pop
pop_size = len(attrition_pop)

# Calculate the interval
interval = pop_size // sample_size

# Systematically sample 70 rows
attrition_sys_samp = attrition_pop[::interval]

# Print the sample
print(attrition_sys_samp)

# %%
# Shuffle the rows of attrition_pop
attrition_shuffled = attrition_pop.sample(frac=1)

# Reset the row indexes and create an index column
attrition_shuffled = attrition_shuffled.reset_index(drop=True).reset_index()

# Plot YearsAtCompany vs. index for attrition_shuffled
attrition_shuffled.plot(x='index',
                        y='YearsAtCompany',
                        kind='scatter')
plt.show()

# %% 2.2 Can't get no stratisfaction
top_counts = coffee_ratings['country_of_origin'].value_counts()
top_counts.head(5)

top_counted_countries = ['Mexico', 'Colombia', 'Guatemala', 'Brazil', 'Taiwan', 'United States (Hawaii)']
top_counted_subset = coffee_ratings['country_of_origin'].isin(top_counted_countries)
coffee_ratings_top = coffee_ratings[top_counted_subset]

coffee_ratings_samp = coffee_ratings_top.sample(frac=0.1, random_state=2021)
coffee_ratings_samp['country_of_origin'].value_counts(normalize=True)

coffee_ratings_strat = coffee_ratings_top.groupby('country_of_origin') \
    .sample(frac=0.1, random_state=2021)
coffee_ratings_strat['country_of_origin'].value_counts(normalize=True)

coffee_ratings_eq = coffee_ratings_top.groupby('country_of_origin') \
    .sample(n=15, random_state=2021)
coffee_ratings_eq['country_of_origin'].value_counts(normalize=True)

coffee_ratings_weight = coffee_ratings_top
condition = coffee_ratings_weight['country_of_origin'] == 'Taiwan'
coffee_ratings_weight['weight'] = np.where(condition, 2, 1)
coffee_ratings_weight = coffee_ratings_weight.sample(frac=0.1, weights='weight')
coffee_ratings_weight['country_of_origin'].value_counts(normalize=True)

# %%
# Proportion of employees by Education level
education_counts_pop = attrition_pop['Education'].value_counts(normalize=True)

# Print education_counts_pop
print(education_counts_pop)

# Proportional stratified sampling for 40% of each Education group
attrition_strat = attrition_pop.groupby('Education') \
    .sample(frac=0.4, random_state=2022)

# Calculate the Education level proportions from attrition_strat
education_counts_strat = attrition_strat['Education'].value_counts(normalize=True)

# Print education_counts_strat
print(education_counts_strat)

# %%
# Get 30 employees from each Education group
attrition_eq = attrition_pop.groupby('Education') \
    .sample(n=30, random_state=2022)

# Get the proportions from attrition_eq
education_counts_eq = attrition_eq['Education'].value_counts(normalize=True)

# Print the results
print(education_counts_eq)

# %% 2.3 What a cluster...
varieties_pop = list(coffee_ratings['variety'].unique())

varieties_samp = random.sample(varieties_pop, k=3)
variety_conditions = coffee_ratings['variety'].isin(varieties_samp)
coffee_ratings_cluster = coffee_ratings[variety_conditions]
coffee_ratings_cluster['variety'] = coffee_ratings_cluster['variety'].astype("category").cat.remove_unused_categories()
coffee_ratings_cluster['variety'].unique()
coffee_ratings_cluster.groupby('variety') \
    .sample(n=5, random_state=2021)

# %%
# Create a list of unique JobRole values
job_roles_pop = list(attrition_pop['JobRole'].unique())

# Randomly sample four JobRole values
job_roles_samp = random.sample(job_roles_pop, k=4)

# Filter for rows where JobRole is in job_roles_samp
jobrole_condition = attrition_pop['JobRole'].isin(job_roles_samp)
attrition_filtered = attrition_pop[jobrole_condition]

# Remove categories with no rows
attrition_filtered['JobRole'] = attrition_filtered['JobRole'].cat.remove_unused_categories()

# Randomly sample 10 employees from each sampled job role
attrition_clust = attrition_filtered.groupby('JobRole') \
    .sample(n=10, random_state=2022)

# Print the sample
print(attrition_clust)

# %% 2.4 Straight to the point (estimate)
# Create a list of unique RelationshipSatisfaction values
satisfaction_unique = list(attrition_pop['RelationshipSatisfaction'].unique())

# Randomly sample 2 unique satisfaction values
satisfaction_samp = random.sample(satisfaction_unique, k=2)

# Filter for satisfaction_samp and clear unused categories from RelationshipSatisfaction
satis_condition = attrition_pop['RelationshipSatisfaction'].isin(satisfaction_samp)
attrition_clust_prep = attrition_pop[satis_condition]
attrition_clust_prep['RelationshipSatisfaction'] = attrition_clust_prep[
    'RelationshipSatisfaction'].cat.remove_unused_categories()

# Perform cluster sampling on the selected group, getting 0.25 of attrition_pop
attrition_clust = attrition_clust_prep.groupby('RelationshipSatisfaction').sample(n=len(attrition_pop) // 4,
                                                                                  random_state=2022)

# %%
# Mean Attrition by RelationshipSatisfaction group
mean_attrition_pop = attrition_pop.groupby('RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_pop)

# Calculate the same thing for the simple random sample
mean_attrition_srs = attrition_srs.groupby('RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_srs)

# Calculate the same thing for the stratified sample
mean_attrition_strat = attrition_strat.groupby('RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_strat)

# Calculate the same thing for the cluster sample
mean_attrition_clust = attrition_clust.groupby('RelationshipSatisfaction')['Attrition'].mean()

# Print the result
print(mean_attrition_clust)

# %% 3. The n's justify the means

# %% 3.1 An ample sample
len(coffee_ratings)
len(coffee_ratings.sample(n=300))

len(coffee_ratings.sample(frac=0.25))

coffee_ratings['total_cup_points'].mean()

coffee_ratings.sample(n=10)['total_cup_points'].mean()
coffee_ratings.sample(n=100)['total_cup_points'].mean()
coffee_ratings.sample(n=1000)['total_cup_points'].mean()

population_mean = coffee_ratings['total_cup_points'].mean()
sample_mean = coffee_ratings.sample(n=sample_size)['total_cup_points'].mean()
rel_error_pct = 100 * abs(population_mean - sample_mean) / population_mean

#%%
# Generate a simple random sample of 50 rows, with seed 2022
attrition_srs50 = attrition_pop.sample(n=50, random_state=2022)

# Calculate the mean employee attrition in the sample
mean_attrition_srs50 = attrition_srs50['Attrition'].mean()

# Calculate the relative error percentage
rel_error_pct50 = 100 * abs(attrition_pop['Attrition'].mean() - mean_attrition_srs50) / attrition_pop['Attrition'].mean()

# Print rel_error_pct50
print(rel_error_pct50)

#%%
# Generate a simple random sample of 100 rows, with seed 2022
attrition_srs100 = attrition_pop.sample(n=100, random_state=2022)

# Calculate the mean employee attrition in the sample
mean_attrition_srs100 = attrition_srs100['Attrition'].mean()

# Calculate the relative error percentage
rel_error_pct100 = 100 * abs(attrition_pop['Attrition'].mean() - mean_attrition_srs100) / attrition_pop['Attrition'].mean()

# Print rel_error_pct100
print(rel_error_pct100)

#%% 3.2 Baby back dist-rib-ution
mean_cup_points_1000 = []
for i in range(1000):
    mean_cup_points_1000.append(
        coffee_ratings.sample(n=30)['total_cup_points'].mean()
    )

print(mean_cup_points_1000)

plt.hist(mean_cup_points_1000, bins=30)
plt.show()

#%%
# Create an empty list
mean_attritions = []
# Loop 500 times to create 500 sample means
for i in range(500):
    mean_attritions.append(
        attrition_pop.sample(n=60)['Attrition'].mean()
    )

# Create a histogram of the 500 sample means
plt.hist(mean_attritions, bins=16)
plt.show()

#%% 3.3 Be our guess, put our samples to the test

dice = pd.DataFrame(np.array(np.meshgrid(range(1, 7), range(1, 7), range(1, 7), range(1, 7))).reshape(4, 1296).T)
dice['mean_roll'] = (dice[0] + dice[1] + dice[2] + dice[3]) / 4

dice['mean_roll'] = dice['mean_roll'].astype('category')
dice['mean_roll'].value_counts(sort=False).plot(kind='bar')

n_dice = list(range(1, 101))
n_outcomes = []
for n in n_dice:
    n_outcomes.append(6**n)

outcomes = pd.DataFrame(
    {'n_dice': n_dice,
     'n_outcomes': n_outcomes}
)

outcomes.plot(x='n_dice',
              y='n_outcomes',
              kind='scatter')
plt.show()

sample_size_1000 = []
for i in range(1000):
    sample_size_1000.append(
        np.random.choice(list(range(1, 7)), size=4, replace=True).mean()
    )
print(sample_size_1000)

plt.hist(sample_size_1000, bins=20)

#%%
def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

# Expand a grid representing 5 8-sided dice
dice = expand_grid(
    {'die1': list(range(1, 9)),
     'die2': list(range(1, 9)),
     'die3': list(range(1, 9)),
     'die4': list(range(1, 9)),
     'die5': list(range(1, 9))
     })

# Add a column of mean rolls and convert to a categorical
dice['mean_roll'] = (dice['die1'] + dice['die2'] +
                     dice['die3'] + dice['die4'] +
                     dice['die5']) / 5
dice['mean_roll'] = dice['mean_roll'].astype('category')

# Draw a bar plot of mean_roll
dice['mean_roll'].value_counts(sort=False).plot(kind='bar')
plt.show()

#%%
# Sample one to eight, five times, with replacement
five_rolls = np.random.choice(list(range(1, 9)), size=5, replace=True)

# Print the mean of five_rolls
print(five_rolls.mean())

#%%
# Replicate the sampling code 1000 times
sample_means_1000 = []
for i in range(1000):
    sample_means_1000.append(
        np.random.choice(list(range(1, 9)), size=5, replace=True).mean()
    )

# Print the first 10 entries of the result
print(sample_means_1000[0:10])

# Draw a histogram of sample_means_1000 with 20 bins
plt.hist(sample_means_1000, bins=20)
plt.show()

#%% 3.4 Err on the side of Gaussian

coffee_ratings['total_cup_points'].std(ddof=0)

#%%
# Calculate the mean of the mean attritions for each sampling distribution
mean_of_means_5 = np.mean(sampling_distribution_5)
mean_of_means_50 = np.mean(sampling_distribution_50)
mean_of_means_500 = np.mean(sampling_distribution_500)

# Print the results
print(mean_of_means_5)
print(mean_of_means_50)
print(mean_of_means_500)

#%%
# Calculate the std. dev. of the mean attritions for each sampling distribution
sd_of_means_5 = np.std(sampling_distribution_5, ddof=1)
sd_of_means_50 = np.std(sampling_distribution_50, ddof=1)
sd_of_means_500 = np.std(sampling_distribution_500, ddof=1)

# Print the results
print(sd_of_means_5)
print(sd_of_means_50)
print(sd_of_means_500)

#%% 4. Pull Your Data Up By Its Bootstraps

#%% 4.1 This bears a striking resample-lance
coffee_focus = coffee_ratings[['variety', 'country_of_origin', 'flavor']]
coffee_focus = coffee_focus.reset_index()
coffee_resamp = coffee_focus.sample(frac=1, replace=True)
coffee_resamp['index'].value_counts()
num_unique_coffees = len(coffee_resamp.drop_duplicates(subset='index'))
len(coffee_ratings) - num_unique_coffees

mean_flavors_1000 = []
for i in range(1000):
    mean_flavors_1000.append(
        np.mean(coffee_ratings.sample(frac=1, replace=True)['flavor'])
    )

plt.hist(mean_flavors_1000)
plt.show()

#%%
# Generate 1 bootstrap resample
spotify_1_resample = spotify_sample.sample(frac=1, replace=True)

# Print the resample
print(spotify_1_resample)
#%%
# Generate 1 bootstrap resample
spotify_1_resample = spotify_sample.sample(frac=1, replace=True)

# Calculate mean danceability of resample
mean_danceability_1 = np.mean(spotify_1_resample['danceability'])

# Print the result
print(mean_danceability_1)

#%%
# Replicate this 1000 times
mean_danceability_1000 = []
for i in range(1000):
    mean_danceability_1000.append(
        np.mean(spotify_sample.sample(frac=1, replace=True)['danceability'])
    )

# Print the result
print(mean_danceability_1000)
#%%
# Replicate this 1000 times
mean_danceability_1000 = []
for i in range(1000):
    mean_danceability_1000.append(
        np.mean(spotify_sample.sample(frac=1, replace=True)['danceability'])
    )

# Draw a histogram of the resample means
plt.hist(mean_danceability_1000)
plt.show()

#%% 4.2 A breath of fresh error
# Pop std.dev = Std.Error * sqrt(Sample.size)

mean_popularity_2000_samp = []

# Generate a sampling distribution of 2000 replicates
for i in range(2000):
    mean_popularity_2000_samp.append(
        # Sample 500 rows and calculate the mean popularity
        spotify_population.sample(n=500)['popularity'].mean()
    )

# Print the sampling distribution results
print(mean_popularity_2000_samp)
#%%
mean_popularity_2000_boot = []

# Generate a bootstrap distribution of 2000 replicates
for i in range(2000):
    mean_popularity_2000_boot.append(
        # Resample 500 rows and calculate the mean popularity
        spotify_sample.sample(n=500, replace=True)['popularity'].mean()
    )

# Print the bootstrap distribution results
print(mean_popularity_2000_boot)
#%%
# Calculate the population mean popularity
pop_mean = np.mean(spotify_population['popularity'])

# Calculate the original sample mean popularity
samp_mean = np.mean(spotify_sample['popularity'])

# Calculate the sampling dist'n estimate of mean popularity
samp_distn_mean = np.mean(sampling_distribution)

# Calculate the bootstrap dist'n estimate of mean popularity
boot_distn_mean = np.mean(bootstrap_distribution)

# Print the means
print([pop_mean, samp_mean, samp_distn_mean, boot_distn_mean])

#%%
# Calculate the population std dev popularity
pop_sd = spotify_population['popularity'].std(ddof=0)

# Calculate the original sample std dev popularity
samp_sd = spotify_sample['popularity'].std()

# Calculate the sampling dist'n estimate of std dev popularity
samp_distn_sd = np.std(sampling_distribution, ddof=1) * np.sqrt(5000)

# Calculate the bootstrap dist'n estimate of std dev popularity
boot_distn_sd = np.std(bootstrap_distribution, ddof=1) * np.sqrt(5000)

# Print the standard deviations
print([pop_sd, samp_sd, samp_distn_sd, boot_distn_sd])

#%% 4.3 Venus infers
np.quantile(coffee_boot_dustn, 0.025)
np.quantile(coffee_boot_dustn, 0.975)

norm.ppf(quantile, loc=0, scale=1)

point_estimate = np.mean(coffee_boot_dist)
std_error = np.std(coffee_boot_dist, ddof=1)

lower = norm.ppf(0.025, loc=point_estimate, scale=std_error)
upper = norm.ppf(0.975, loc=point_estimate, scale=std_error)
print(lower, upper)
#%%

# Generate a 95% confidence interval using the quantile method
lower_quant = np.quantile(bootstrap_distribution, 0.025)
upper_quant = np.quantile(bootstrap_distribution, 0.975)

# Print quantile method confidence interval
print((lower_quant, upper_quant))
#%%
# Find the mean and std dev of the bootstrap distribution
point_estimate = np.mean(bootstrap_distribution)
standard_error = np.std(bootstrap_distribution, ddof=1)

# Find the lower limit of the confidence interval
lower_se = norm.ppf(0.025, loc=point_estimate, scale=standard_error)

# Find the upper limit of the confidence interval
upper_se = norm.ppf(0.975, loc=point_estimate, scale=standard_error)

# Print standard error method confidence interval
print((lower_se, upper_se))