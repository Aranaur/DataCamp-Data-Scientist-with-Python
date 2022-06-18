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

fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], marker='o', linestyle='--', color='g')
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"], marker='v', linestyle='None', color='r')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
ax.set_title('Weather in Seattle')
plt.show()

# %% 1.2 Small multiples

fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"], color='b')
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-25PCTL"], color='b', linestyle='--')
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-75PCTL"], color='b', linestyle='--')
ax.plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"], color='r')
ax.plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-25PCTL"], color='r', linestyle='--')
ax.plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-75PCTL"], color='r', linestyle='--')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Precipitation (inches)')
plt.show()

fig, ax = plt.subplots(2, 1, sharey=True)
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"], color='b')
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-25PCTL"], color='b', linestyle='--')
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-75PCTL"], color='b', linestyle='--')
ax[1].plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"], color='r')
ax[1].plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-25PCTL"], color='r', linestyle='--')
ax[1].plot(seattle_weather["MONTH"], austin_weather["MLY-PRCP-75PCTL"], color='r', linestyle='--')
ax[1].set_xlabel('Time (months)')
ax[0].set_ylabel('Precipitation (inches)')
ax[1].set_ylabel('Precipitation (inches)')
plt.show()

# %%
# Create a Figure and an array of subplots with 2 rows and 2 columns
fig, ax = plt.subplots(2, 2)

# Addressing the top left Axes as index 0, 0, plot month and Seattle precipitation
ax[0, 0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"])

# In the top right (index 0,1), plot month and Seattle temperatures
ax[0, 1].plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])

# In the bottom left (1, 0) plot month and Austin precipitations
ax[1, 0].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"])

# In the bottom right (1, 1) plot month and Austin temperatures
ax[1, 1].plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

# %%

# Create a figure and an array of axes: 2 rows, 1 column with shared y axis
fig, ax = plt.subplots(2, 1, sharey=True)

# %% 2. Plotting time-series data

climate_change

climate_change.index

climate_change['co2']

fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

sixties = climate_change['1960-01-01':'1969-12-31']

fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

sixty_nine = climate_change['1969-01-01':'1969-12-31']

fig, ax = plt.subplots()
ax.plot(sixty_nine.index, sixty_nine['co2'])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
plt.show()

# %%

# Add the time-series for "relative_temp" to the plot
ax.plot(climate_change.index, climate_change['relative_temp'])

# Set the x-axis label
ax.set_xlabel('Time')

# Set the y-axis label
ax.set_ylabel('Relative temperature (Celsius)')

# Show the figure
plt.show()

# %%

# Use plt.subplots to create fig and ax
fig, ax = plt.subplots()

# Create variable seventies with data from "1970-01-01" to "1979-12-31"
seventies = climate_change['1970-01-01':'1979-12-31']

# Add the time-series for "co2" data from seventies to the plot
ax.plot(seventies.index, seventies['co2'])

# Show the figure
plt.show()

# %% 2.1 Plotting time-series with different variables


