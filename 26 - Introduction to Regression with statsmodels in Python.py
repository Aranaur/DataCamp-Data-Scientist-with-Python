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

#%%
mdl_ad = ols('sqer_n_impressions ~ sqer_spend', data=ad_conversion).fit()

explanatory_data = pd.DataFrame({
    'sqer_spend': np.sqrt(np.arange(0, 601, 100)),
    'spent_usd': np.arange(0, 601, 100)
})

prediction_data = explanatory_data.assign(sqrt_n_impress=mdl_ad.predict(explanatory_data),
                                          n_impression=mdl_ad.predict(explanatory_data) ** 2)

prediction_data

#%%
# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

plt.figure()

# Plot using the transformed variable
sns.regplot(x='sqrt_dist_to_mrt_m',
            y='price_twd_msq',
            data=taiwan_real_estate,
            ci=None)
plt.show()

#%%
# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

# Run a linear regression of price_twd_msq vs. sqrt_dist_to_mrt_m
mdl_price_vs_dist = ols("price_twd_msq ~ sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

# Use this explanatory data
explanatory_data = pd.DataFrame({"sqrt_dist_to_mrt_m": np.sqrt(np.arange(0, 81, 10) ** 2),
                                 "dist_to_mrt_m": np.arange(0, 81, 10) ** 2})

# Use mdl_price_vs_dist to predict explanatory_data
prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_dist.predict(explanatory_data)
)

fig = plt.figure()
sns.regplot(x="sqrt_dist_to_mrt_m", y="price_twd_msq", data=taiwan_real_estate, ci=None)

# Add a layer of your prediction points
sns.scatterplot(data=prediction_data, x='sqrt_dist_to_mrt_m', y='price_twd_msq', color='r')
plt.show()

#%%
ad_conversion["qdrt_n_impressions"] = ad_conversion["n_impressions"] ** 0.25
ad_conversion["qdrt_n_clicks"] = ad_conversion["n_clicks"] ** 0.25

mdl_click_vs_impression = ols("qdrt_n_clicks ~ qdrt_n_impressions", data=ad_conversion, ci=None).fit()

explanatory_data = pd.DataFrame({"qdrt_n_impressions": np.arange(0, 3e6+1, 5e5) ** .25,
                                 "n_impressions": np.arange(0, 3e6+1, 5e5)})

# Complete prediction_data
prediction_data = explanatory_data.assign(
    qdrt_n_clicks = mdl_click_vs_impression.predict(explanatory_data)
)

# Print the result
print(prediction_data)
#%%
# Back transform qdrt_n_clicks
prediction_data["n_clicks"] = prediction_data["qdrt_n_clicks"] ** 4

# Plot the transformed variables
fig = plt.figure()
sns.regplot(x="qdrt_n_impressions", y="qdrt_n_clicks", data=ad_conversion, ci=None)

# Add a layer of your prediction points
sns.scatterplot(data=prediction_data, x='qdrt_n_impressions', y='qdrt_n_clicks', color='r')
plt.show()

#%% 3. Assessing model fit

#%% 3.1 Quantifying model fit

print(bream_ols.summary())

bream_ols.rsquared

bream['length_cm'].corr(bream['mass_g']) ** 2

bream_ols.mse_resid

np.sqrt(bream_ols.mse_resid)

# RSE
df = len(bream.index) - 2
np.sqrt(sum(bream_ols.resid ** 2) / df)

# RMSE
np.sqrt(sum(bream_ols.resid ** 2) / len(bream.index))

#%%
# Calculate mse_orig for mdl_click_vs_impression_orig
mse_orig = mdl_click_vs_impression_orig.mse_resid

# Calculate rse_orig for mdl_click_vs_impression_orig and print it
rse_orig = np.sqrt(mse_orig)
print("RSE of original model: ", rse_orig)

# Calculate mse_trans for mdl_click_vs_impression_trans
mse_trans = mdl_click_vs_impression_trans.mse_resid

# Calculate rse_trans for mdl_click_vs_impression_trans and print it
rse_trans = np.sqrt(mse_trans)
print("RSE of transformed model: ", rse_trans)

#%% 3.2 Visualizing model fit
# Resid vs fitted
sns.residplot(x='length_cm',
              y='mass_g',
              data=bream,
              lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

# QQ plot
from statsmodels.api import qqplot
qqplot(data=bream_ols.resid, fit=True, line='45')
plt.show()

# Scale-location plot
model_norm_resud_bream = bream_ols.get_influence().resid_studentized_internal
model_norm_abs_sqrt_bream = np.sqrt(np.abs(model_norm_resud_bream))
sns.regplot(x=bream_ols.fittedvalues,
            y=model_norm_abs_sqrt_bream,
            ci=None,
            lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Sqrt of abs val of std resid')
plt.show()

#%%
# Plot the residuals vs. fitted values
sns.residplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

# Show the plot
plt.show()

#%%
# Import qqplot
from statsmodels.api import qqplot

# Create the Q-Q plot of the residuals
qqplot(data=mdl_price_vs_conv.resid, fit=True, line="45")

# Show the plot
plt.show()

#%%
# Preprocessing steps
model_norm_residuals = mdl_price_vs_conv.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# Create the scale-location plot
sns.regplot(x=mdl_price_vs_conv.fittedvalues, y=model_norm_residuals_abs_sqrt, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")

# Show the plot
plt.show()

#%% 3.3 Outliers, leverage, and influence
roach = fish[fish['species'] == 'Roach']
roach.head()

sns.regplot(x='length_cm',
            y='mass_g',
            data=roach,
            ci=None)
plt.show()

roach['extreme_l'] = ((roach['length_cm'] < 15) | (roach['length_cm'] > 26))

fig = plt.figure()
sns.regplot(x='length_cm',
            y='mass_g',
            data=roach,
            ci=None)
sns.scatterplot(x='length_cm',
                y='mass_g',
                data=roach,
                hue='extreme_l')
plt.show()

roach['extreme_m'] = roach['mass_g'] < 1

fig = plt.figure()
sns.regplot(x='length_cm',
            y='mass_g',
            data=roach,
            ci=None)
sns.scatterplot(x='length_cm',
                y='mass_g',
                data=roach,
                hue='extreme_l',
                style='extreme_m')
plt.show()

mdl_roach = ols('mass_g ~ length_cm', data=roach).fit()
summary_roach = mdl_roach.get_influence().summary_frame()
roach['leverage'] = summary_roach['hat_diag']
roach.head()

roach['cooks_dist'] = summary_roach['cooks_d']
roach.head()

roach.sort_values('cooks_dist', ascending=False)

roach_not_short = roach[roach['length_cm'] != 12.9]
sns.regplot(x='length_cm',
            y='mass_g',
            data=roach,
            ci=None,
            line_kws={'color': 'g'})
sns.regplot(x='length_cm',
            y='mass_g',
            data=roach_not_short,
            ci=None,
            line_kws={'color': 'r'})
plt.show()

#%%
# Create summary_info
summary_info = mdl_price_vs_dist.get_influence().summary_frame()

# Add the hat_diag column to taiwan_real_estate, name it leverage
taiwan_real_estate["leverage"] = summary_info["hat_diag"]

# Add the cooks_d column to taiwan_real_estate, name it cooks_dist
taiwan_real_estate['cooks_dist'] = summary_info['cooks_d']

# Sort taiwan_real_estate by cooks_dist in descending order and print the head.
print(taiwan_real_estate.sort_values('cooks_dist', ascending=False).head())


#%% 4. Simple Logistic Regression Modeling

#%% 4.1 Why you need logistic regression
churn.head()

mdl_churn_vs_recency_lm = ols('has_churned ~ time_since_last_purchase', data=churn).fit()
mdl_churn_vs_recency_lm.params

intercept, slope = mdl_churn_vs_recency_lm.params

sns.scatterplot(x='time_since_last_purchase',
                y='has_churned',
                data=churn)
plt.axline(xy1=(0, intercept), slope=slope)
plt.xlim(-10, 10)
plt.ylim(-0.2, 1.2)
plt.show()

from statsmodels.formula.api import logit
mdl_churn_vs_recency_logit = logit('has_churned ~ time_since_last_purchase', data=churn).fit()
mdl_churn_vs_recency_logit.params

sns.regplot(x='time_since_last_purchase',
            y='has_churned',
            data=churn,
            ci=None,
            logistic=True)
plt.axline(xy1=(0, intercept), slope=slope, color='black')
plt.show()

#%%
sns.displot(data=churn, x='time_since_last_purchase', col='has_churned')
plt.show()
sns.displot(data=churn, x='time_since_first_purchase', col='has_churned')
plt.show()

#%%
# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            line_kws={"color": "red"})

# Draw a logistic regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x='time_since_first_purchase',
            y='has_churned',
            data=churn,
            ci=None,
            logistic=True,
            line_kws={"color": "blue"})

plt.show()

#%%
# Import logit
from statsmodels.formula.api import logit

# Fit a logistic regression of churn vs. length of relationship using the churn dataset
mdl_churn_vs_relationship = logit('has_churned ~ time_since_first_purchase', data=churn).fit()

# Print the parameters of the fitted model
print(mdl_churn_vs_relationship.params)

#%% 4.2 Predictions and odds ratios
mdl_recency = logit('has_churned ~ time_since_last_purchase', data=churn).fit()

explanatory_data = pd.DataFrame(
    {'time_since_last_purchase': np.arange(-1, 6.25, 0.25)}
)

prediction_data = explanatory_data.assign(has_churned=mdl_recency.predict(explanatory_data))

sns.regplot(x='time_since_last_purchase',
            y='has_churned',
            data=churn,
            ci=None,
            logistic=True)
sns.scatterplot(x='time_since_last_purchase',
                y='has_churned',
                data=prediction_data,
                color='r')
plt.show()

prediction_data['most_likely_outcome'] = np.round(prediction_data['has_churned'])

sns.regplot(x='time_since_last_purchase',
            y='has_churned',
            data=churn,
            ci=None,
            logistic=True)
sns.scatterplot(x='time_since_last_purchase',
                y='most_likely_outcome',
                data=prediction_data,
                color='r')
plt.show()

prediction_data['odds_ratio'] = prediction_data['has_churned'] / (1 - prediction_data['has_churned'])

sns.lineplot(x='time_since_last_purchase',
             y='odds_ratio',
             data=prediction_data)
plt.axhline(y=1, linestyle='dotted')
plt.show()

sns.lineplot(x='time_since_last_purchase',
             y='odds_ratio',
             data=prediction_data)
plt.axhline(y=1, linestyle='dotted')
plt.yscale('log')
plt.show()

prediction_data['log_odds_ratio'] = np.log(prediction_data['odds_ratio'])

#%%
# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned=mdl_churn_vs_relationship.predict(explanatory_data)
)

# Print the head
print(prediction_data.head())
#%%
# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned = mdl_churn_vs_relationship.predict(explanatory_data)
)

fig = plt.figure()

# Create a scatter plot with logistic trend line
sns.regplot(x='time_since_first_purchase', y='has_churned', ci=None, logistic=True, data=churn)

# Overlay with prediction_data, colored red
sns.scatterplot(x='time_since_first_purchase',
                y='has_churned',
                data=prediction_data,
                color='r')

plt.show()

#%%
# Update prediction data by adding most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])

fig = plt.figure()

# Create a scatter plot with logistic trend line (from previous exercise)
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            logistic=True)

# Overlay with prediction_data, colored red
sns.scatterplot(x='time_since_first_purchase',
                y='most_likely_outcome',
                data=prediction_data,
                color='r')

plt.show()

#%%
# Update prediction data with odds_ratio
prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])

fig = plt.figure()

# Create a line plot of odds_ratio vs time_since_first_purchase
sns.lineplot(x='time_since_first_purchase',
             y='odds_ratio',
             data=prediction_data)

# Add a dotted horizontal line at odds_ratio = 1
plt.axhline(y=1, linestyle="dotted")

plt.show()
#%%
# Update prediction data with log_odds_ratio
prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])

fig = plt.figure()

# Update the line plot: log_odds_ratio vs. time_since_first_purchase
sns.lineplot(x="time_since_first_purchase",
             y="log_odds_ratio",
             data=prediction_data)

# Add a dotted horizontal line at log_odds_ratio = 0
plt.axhline(y=0, linestyle="dotted")
plt.yscale('log')
plt.show()

#%% 4.2 Quantifying logistic regression fit
actual_response = churn['has_churned']
predicted_response = np.round(mdl_recency.predict())
outcomes = pd.DataFrame(
    {'actual_response': actual_response,
     'predicted_response': predicted_response}
)
print(outcomes.value_counts(sort=False))

conf_matrix = mdl_recency.pred_table()
print(conf_matrix)

from statsmodels.graphics.mosaicplot import mosaic
mosaic(conf_matrix)

TN = conf_matrix[0, 0]
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]

acc = (TN + TP) / (TN + TP + FN + FP)
acc

sens = TP / (FN + TP)
sens

spec = TN / (TN + FP)
spec

#%%
# Get the actual responses
actual_response = churn['has_churned']

# Get the predicted responses
predicted_response = np.round(mdl_churn_vs_relationship.predict())

# Create outcomes as a DataFrame of both Series
outcomes = pd.DataFrame({'actual_response': actual_response,
                         'predicted_response': predicted_response})

# Print the outcomes
print(outcomes.value_counts(sort = False))

#%%
# Import mosaic from statsmodels.graphics.mosaicplot
from statsmodels.graphics.mosaicplot import mosaic

# Calculate the confusion matrix conf_matrix
conf_matrix = mdl_churn_vs_relationship.pred_table()

# Print it
print(conf_matrix)

# Draw a mosaic plot of conf_matrix
mosaic(conf_matrix)
plt.show()

#%%
# Extract TN, TP, FN and FP from conf_matrix
TN = conf_matrix[0, 0]
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + TP + FN + FP)
print("accuracy: ", accuracy)

# Calculate and print the sensitivity
sensitivity = TP / (FN + TP)
print("sensitivity: ", sensitivity)

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("specificity: ", specificity)