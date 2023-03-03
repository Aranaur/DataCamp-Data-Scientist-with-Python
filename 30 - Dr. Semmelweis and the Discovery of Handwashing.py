# %% 1. Meet Dr. Ignaz Semmelweis
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv('datasets/yearly_deaths_by_clinic.csv')

# Print out yearly
yearly.head()

# %% 2. The alarming number of deaths

# Calculate proportion of deaths per no. births
yearly['proportion_deaths'] = yearly['deaths'] / yearly['births']
# Extract Clinic 1 data into clinic_1 and Clinic 2 data into clinic_2
clinic_1 = yearly[yearly['clinic'] == 'clinic 1']
clinic_2 = yearly[yearly['clinic'] == 'clinic 2']

# Print out clinic_1
clinic_1.head()

# %% 3. Death at the clinics
# Import matplotlib
import matplotlib.pyplot as plt

# This makes plots appear in the notebook
%matplotlib inline

# Plot yearly proportion of deaths at the two clinics
ax = clinic_1.plot(x="year", y="proportion_deaths", 
                   label="Clinic 1")
clinic_2.plot(x="year", y="proportion_deaths", 
              label="Clinic 2", ax=ax,
              ylabel="Proportion deaths", 
              title='yearly proportion of deaths at Clinic 1 and Clinic 2')
plt.show()

# %% 4. The handwashing begins
# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('datasets/monthly_deaths.csv', parse_dates=['date'])

# Calculate proportion of deaths per no. births
monthly['proportion_deaths'] = monthly['deaths']/monthly['births']

# Print out the first rows in monthly
monthly.head()

# %% 5. The effect of handwashing
# Plot monthly proportion of deaths
ax = monthly.plot(x='date', y='proportion_deaths', ylabel='Proportion deaths')

# %% 6. The effect of handwashing highlighted

# Date when handwashing was made mandatory
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date'] < handwashing_start]
after_washing = monthly[monthly['date'] >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing
ax = before_washing.plot(x='date', y='proportion_deaths', label='Before Washing')
after_washing.plot(x='date', y='proportion_deaths', label='After washing', ax=ax)

# %% 7. More handwashing, fewer deaths?
# Difference in mean monthly proportion of deaths due to handwashing
before_proportion = before_washing['proportion_deaths']
after_proportion = after_washing['proportion_deaths']
mean_diff = after_proportion.mean() - before_proportion.mean()
mean_diff

# %% 8. A Bootstrap analysis of Semmelweis handwashing data
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append(boot_before.mean()-boot_after.mean() )

# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval

# %% 9. The fate of Dr. Semmelweis
# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True