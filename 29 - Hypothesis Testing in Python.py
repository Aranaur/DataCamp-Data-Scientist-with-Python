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
