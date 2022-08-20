import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

churn = pd.read_csv('data/26/churn.csv')
fish = pd.read_csv('data/26/fish.csv')
ad_conversion = pd.read_csv('data/26/ad_conversion.csv')
sp500_yearly_returns = pd.read_csv('data/26/sp500_yearly_returns.csv')
taiwan_real_estate = pd.read_csv('data/26/taiwan_real_estate2.csv')

#%% 1. Simple Linear Regression Modeling
fish.mean()

fish.mass_g.corr(fish.length_cm)

sns.scatterplot('mass_g', 'length_cm', data=fish)
plt.show()

sns.regplot('mass_g', 'length_cm', data=fish, ci=None)
plt.show()

#%%
# Import seaborn with alias sns
import seaborn as sns

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Draw the scatter plot
sns.scatterplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate)

# Show the plot
plt.show()

#%%
# Import seaborn with alias sns
import seaborn as sns

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Draw the scatter plot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=taiwan_real_estate)

# Draw a trend line on the scatter plot of price_twd_msq vs. n_convenience
sns.regplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate,
            ci=None,
            scatter_kws={'alpha': 0.5})

# Show the plot
plt.show()

#%% 1.2 Fitting a linear regression
estate_fit = ols('price_twd_msq ~ dist_to_mrt_m', data=taiwan_real_estate)
estate_fit = estate_fit.fit()
print(estate_fit.params)

#%%
# Import the ols function
from statsmodels.formula.api import ols

# Create the model object
mdl_price_vs_conv = ols('price_twd_msq ~ n_convenience', data=taiwan_real_estate)

# Fit the model
mdl_price_vs_conv = mdl_price_vs_conv.fit()

# Print the parameters of the fitted model
print(mdl_price_vs_conv.params)

#%%
# On average, a house with zero convenience stores nearby had a price of 8.2242 TWD per square meter.

#%%
# If you increase the number of nearby convenience stores by one,
# then the expected increase in house price is 0.7981 TWD per square meter.

#%% 1.3 Categorical explanatory variables
fish

sns.displot(data=fish,
            x='mass_g',
            col='species',
            col_wrap=2,
            bins=9)
plt.show()

summary_stats = fish.groupby('species')['mass_g'].mean()
print(summary_stats)

fish_ols = ols('mass_g ~ species', data=fish).fit()
fish_ols.params

fish_ols_zero = ols('mass_g ~ species + 0', data=fish).fit()
fish_ols_zero.params

#%%
# Histograms of price_twd_msq with 10 bins, split by the age of each house
sns.displot(data=taiwan_real_estate,
            x='price_twd_msq',
            col='house_age_years',
            bins=10)

# Show the plot
plt.show()

#%%
# Calculate the mean of price_twd_msq, grouped by house age
mean_price_by_age = taiwan_real_estate.groupby('house_age_years')['price_twd_msq'].mean()

# Print the result
print(mean_price_by_age)

#%%
# Create the model, fit it
mdl_price_vs_age = ols('price_twd_msq ~ house_age_years', data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age.params)

#%%
# Update the model formula to remove the intercept
mdl_price_vs_age0 = ols("price_twd_msq ~ house_age_years + 0", data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age0.params)

#%% 2. Predictions and model objects

#%% 2.1 Making predictions
bream = fish[fish['species'] == 'Bream']
bream.head()

sns.regplot(x='length_cm',
            y='mass_g',
            data=bream,
            ci=None)
plt.show()

bream_ols = ols('mass_g ~ length_cm', data=bream).fit()
bream_ols.params

exp_data = pd.DataFrame({'length_cm': np.arange(20, 41)})
bream_ols.predict(exp_data)

predict_data = exp_data.assign(
    mass_g=bream_ols.predict(exp_data)
)