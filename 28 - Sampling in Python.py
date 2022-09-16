import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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


