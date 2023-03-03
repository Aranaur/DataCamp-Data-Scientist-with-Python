import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

late_shipments = pd.read_feather('data/29/late_shipments.feather')
stack_overflow = pd.read_feather('data/29/stack_overflow.feather')
dem_votes_potus_12_16 = pd.read_feather('data/29/dem_votes_potus_12_16.feather')
repub_votes_potus_08_12 = pd.read_feather('data/29/repub_votes_potus_08_12.feather')


#%% 1. Yum, That Dish Tests Good

#%% 1.1 To the lab for testing
print(stack_overflow)

mean_comp_samp = stack_overflow['converted_comp'].mean()

so_boot_distn = []
for i in range(5000):
    so_boot_distn.append(
        np.mean(
            stack_overflow.sample(frac=1, replace=True)['converted_comp']
        )
    )

plt.hist(so_boot_distn, bins=50)
plt.show()

std_error = np.std(so_boot_distn, ddof=1)
std_error

z_score = (mean_comp_samp - 110000) / std_error

#%%
# Print the late_shipments dataset
print(late_shipments)

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late'] == 'Yes').mean()

# Print the results
print(late_prop_samp)

#%%
# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn, ddof=1)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)

#%% A tail of two z's
prop_child_samp = (stack_overflow['age_first_code_cut'] == 'child').mean()
prop_child_hyp = 0.35

first_code_boot_distn = []
for i in range(5000):
    first_code_boot_distn.append(
        np.mean(
            (stack_overflow.sample(frac=1, replace=True)['age_first_code_cut'] == 'child')
        )
    )

std_error = np.std(first_code_boot_distn, ddof=1)

z_score = (prop_child_samp - prop_child_hyp) / std_error

1 - norm.cdf(z_score, loc=0, scale=1)  # for right-tailed
norm.cdf(z_score, loc=0, scale=1)  # for left-tailed

#%%
# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Calculate the p-value
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)

# Print the p-value
print(p_value)

#%% Statistically significant other
alpha = 0.05
prop_child_samp = (stack_overflow['age_first_code_cut'] == 'child').mean()
prop_child_hyp = 0.35
std_error = np.std(first_code_boot_distn, ddof=1)
z_score = (prop_child_samp - prop_child_hyp) / std_error
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)
p_value <= alpha  # Reject H0
lower = np.quantile(first_code_boot_distn, 0.025)
upper = np.quantile(first_code_boot_distn, 0.975)
print(lower, upper)

#%%
# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn, 0.025)
upper = np.quantile(late_shipments_boot_distn, 0.975)
# Print the confidence interval
print((lower, upper))

#%% 2. Pass Me ANOVA Glass of Iced t
#%% Performing t-tests
xbar = stack_overflow.groupby('age_first_code_cut')['converted_comp'].mean()
s = stack_overflow.groupby('age_first_code_cut')['converted_comp'].std()
n = stack_overflow.groupby('age_first_code_cut')['converted_comp'].count()

numerator = xbar_child - xbar_adult
denominator = np.sqrt(s_chind ** 2 / n_chind + s_adult ** 2 / n_adult)
t_stat = numerator / denominator

#%% Two sample mean test statistic
# Calculate the numerator of the test statistic
numerator = xbar_no - xbar_yes

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_no ** 2 / n_no + s_yes ** 2 / n_yes)

# Calculate the test statistic
t_stat = numerator / denominator

# Print the test statistic
print(t_stat)

#%% Calculating p-values from t-statistics
# Calculate the degrees of freedom
degrees_of_freedom =  n_no + n_yes - 2

# Calculate the p-value from the test stat
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Print the p_value
print(p_value)

#%% Calculate the differences from 2012 to 2016
sample_dem_data['diff'] = sample_dem_data['dem_percent_12'] - sample_dem_data['dem_percent_16']

# Find the mean of the diff column
xbar_diff = sample_dem_data['diff'].mean()

# Find the standard deviation of the diff column
s_diff = sample_dem_data['diff'].std()

# Plot a histogram of diff with 20 bins
sample_dem_data['diff'].hist(bins=20)
plt.show()

#%%
import pingouin

# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'], y=0)

# Print the test results
print(test_results)

#%%
# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'],
                              y=0,
                              alternative="two-sided")

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(x=sample_dem_data['dem_percent_12'],
                                     y=sample_dem_data['dem_percent_16'],
                                     alternative="two-sided")

# Print the paired test results
print(paired_test_results)

#%% ANOVA tests
# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].mean()

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].std()

# Boxplot of shipment_mode vs. pack_price
sns.boxplot(x='pack_price', y='shipment_mode', data=late_shipments)
plt.show()

#%%
# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(data=late_shipments,
                               dv="pack_price",
                               between="shipment_mode")

# Print anova_results
print(anova_results)

#%%
# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(data=late_shipments,
                               dv="pack_price",
                               between="shipment_mode")

# Print anova_results
print(anova_results)

#%%
# Perform a pairwise t-test on pack price, grouped by shipment mode
pairwise_results = pingouin.pairwise_tests(dv='pack_price', between='shipment_mode', data=late_shipments, padjust="none")

# Print pairwise_results
print(pairwise_results)

#%%
# Modify the pairwise t-tests to use Bonferroni p-value adjustment
pairwise_results = pingouin.pairwise_tests(data=late_shipments,
                                           dv="pack_price",
                                           between="shipment_mode",
                                           padjust='bonf')

# Print pairwise_results
print(pairwise_results)

#%% 3. Proportion Tests