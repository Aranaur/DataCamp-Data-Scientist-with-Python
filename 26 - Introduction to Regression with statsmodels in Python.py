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

predict_data

fig = plt.figure()
sns.regplot(x='length_cm',
            y='mass_g',
            ci=None,
            data=bream)
sns.scatterplot(x='length_cm',
                y='mass_g',
                data=predict_data,
                color='r',
                marker='s')
plt.show()

little_bream = pd.DataFrame({'length_cm': [10]})
pred_little_bream = little_bream.assign(
    mass_g=mdl_mass_vs_lenght.predict(little_bream)
)

#%%
# Import numpy with alias np
import numpy as np

# Create explanatory_data
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})

# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)

# Create prediction_data
prediction_data = explanatory_data.assign(
    price_twd_msq = price_twd_msq)

# Print the result
print(prediction_data)

#%%
# Create a new figure, fig
fig = plt.figure()

sns.regplot(x="n_convenience",
            y="price_twd_msq",
            data=taiwan_real_estate,
            ci=None)
# Add a scatter plot layer to the regplot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=prediction_data,
                color='r')

# Show the layered plot
plt.show()

#%%
# Define a DataFrame impossible
impossible = pd.DataFrame({'n_convenience': [-1, 2.5]})

#%% 2.2 Working with model objects
mdl_mass_vs_length = ols("mass_g ~ length_cm", data=bream).fit()
print(mdl_mass_vs_length.params)
print(mdl_mass_vs_length.fittedvalues)
print(mdl_mass_vs_length.resid)
print(mdl_mass_vs_length.summary())

#%%
# Print the model parameters of mdl_price_vs_conv
print(mdl_price_vs_conv.params)

# Print the fitted values of mdl_price_vs_conv
print(mdl_price_vs_conv.fittedvalues)

# Print the residuals of mdl_price_vs_conv
print(mdl_price_vs_conv.resid)

# Print a summary of mdl_price_vs_conv
print(mdl_price_vs_conv.summary())
#%%
# Get the coefficients of mdl_price_vs_conv
coeffs = mdl_price_vs_conv.params

# Get the intercept
intercept = coeffs[0]

# Get the slope
slope = coeffs[1]

# Manually calculate the predictions
price_twd_msq = intercept + slope * explanatory_data
print(price_twd_msq)

# Compare to the results from .predict()
print(price_twd_msq.assign(predictions_auto=mdl_price_vs_conv.predict(explanatory_data)))

#%% 2.3 Regression to the mean
pearson = pd.read_csv('data/26/Pearson.txt', sep="\t", header=None, skiprows=1)
pearson.columns = ['Father', 'Son']

fig = plt.figure()
sns.scatterplot(x='Father',
                y='Son',
                data=pearson)
plt.axline(xy1=(80, 80),
           slope=1,
           linewidth=2,
           color='g')
sns.regplot(x='Father',
            y='Son',
            data=pearson,
            ci=None,
            line_kws={'color': 'black'})
plt.axis('equal')
plt.show()

fig = plt.figure()
plt.axline(xy1=(80, 80),
           slope=1,
           linewidth=2,
           color='g')
sns.regplot(x='Father',
            y='Son',
            data=pearson,
            ci=None,
            line_kws={'color': 'black'})
plt.axis('equal')
plt.show()

mdl_son_vs_father = ols('Son ~ Father',
                        data=pearson).fit()
print(mdl_son_vs_father.params)

really_tall_father = pd.DataFrame(
    {'Father': [90]})
mdl_son_vs_father.predict(really_tall_father)

really_short_father = pd.DataFrame(
    {'Father': [50]})
mdl_son_vs_father.predict(really_short_father)

#%%
# Create a new figure, fig
fig = plt.figure()

# Plot the first layer: y = x
plt.axline(xy1=(0,0), slope=1, linewidth=2, color="green")

# Add scatter plot with linear regression trend line
sns.regplot(x='return_2018',
            y='return_2019',
            data=sp500_yearly_returns,
            ci=None)

# Set the axes so that the distances along the x and y axes look the same
plt.axis('equal')

# Show the plot
plt.show()

#%%
mdl_returns = ols("return_2019 ~ return_2018", data=sp500_yearly_returns).fit()

# Create a DataFrame with return_2018 at -1, 0, and 1
explanatory_data = pd.DataFrame({'return_2018': [-1, 0, 1]})

# Use mdl_returns to predict with explanatory_data
print(mdl_returns.predict(explanatory_data))

#%% 2.4 Transforming variables
perch = fish[fish['species'] == 'Perch']
perch.head()

sns.regplot(x='length_cm',
            y='mass_g',
            data=perch,
            ci=None)
plt.show()

perch['length_cm_cubed'] = perch['length_cm'] ** 3

sns.regplot(x='length_cm_cubed',
            y='mass_g',
            data=perch,
            ci=None)
plt.show()

mdl_perch = ols('mass_g ~ length_cm_cubed', data=perch).fit()
mdl_perch.params

explanatory_data = pd.DataFrame({'length_cm_cubed': np.arange(10, 41, 5) ** 3,
                                 'length_cm': np.arange(10, 41, 5)})

prediction_data = explanatory_data.assign(
    mass_g=mdl_perch.predict(explanatory_data)
)

prediction_data

fig = plt.figure()
sns.regplot(x='length_cm_cubed',
            y='mass_g',
            data=perch,
            ci=None)
sns.scatterplot(data=prediction_data,
                x='length_cm_cubed',
                y='mass_g',
                color='r',
                markers='s')
plt.show()

fig = plt.figure()
sns.regplot(x='length_cm',
            y='mass_g',
            data=perch,
            ci=None)
sns.scatterplot(data=prediction_data,
                x='length_cm',
                y='mass_g',
                color='r',
                markers='s')
plt.show()

#%%
ad_conversion
sns.regplot(x='spent_usd',
            y='n_impressions',
            data=ad_conversion,
            ci=None)

ad_conversion['sqer_spend'] = np.sqrt(ad_conversion['spent_usd'])
ad_conversion['sqer_n_impressions'] = np.sqrt(ad_conversion['n_impressions'])

sns.regplot(x='sqer_spend',
            y='sqer_n_impressions',
            data=ad_conversion,
            ci=None)