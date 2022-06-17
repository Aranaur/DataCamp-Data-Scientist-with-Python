# %% Importing course packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Importing course datasets as DataFrames

climate_change = pd.read_csv('data/08/climate_change.csv', parse_dates=["date"], index_col="date")
medals = pd.read_csv('data/08/medals_by_country_2016.csv', index_col=0)
summer_2016 = pd.read_csv('data/08/summer2016.csv')
austin_weather = pd.read_csv("data/08/austin_weather.csv", index_col="DATE")
weather = pd.read_csv("data/08/seattle_weather.csv", index_col="DATE")

# %% Some pre-processing on the weather datasets, including adding a month column

seattle_weather = weather[weather["STATION"] == "USW00094290"]
month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
seattle_weather["MONTH"] = month
austin_weather["MONTH"] = month

austin_weather.head()

# %% 1. Introduction to Matplotlib

# %%
fig, ax = plt.subplots()
plt.show()

# %%
seattle_weather["MONTH"]
seattle_weather["MLY-TAVG-NORMAL"]

# %%
plt.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
plt.show()

# %%
plt.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
plt.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

# %% 1.1 Customizing your plots
