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

climate_change

# %%
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change['co2'], color='b')
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color='b')
ax.tick_params('y', colors='b')
ax2 = ax.twinx()
ax2.plot(climate_change.index, climate_change['relative_temp'], color='r')
ax2.set_xlabel('Time')
ax2.tick_params('y', colors='r')
ax2.set_ylabel('Relative temperature (Celsius)', color='r')
plt.show()

# %%
def plot_timeseries (axes, x, y, color, xlabel, ylabel):
    axes.plot(x, y, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color=color)
    axes.tick_params('y', colors=color)

fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time (years)', 'CO2 levels')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time (years)',
                'Relative temperature (Celsius)')
plt.show()

# %% 2.2 Annotating time-series data

fig, ax = plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time (years)', 'CO2 levels')
ax2 = ax.twinx()
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time (years)',
                'Relative temperature (Celsius)')
ax2.annotate('>1 degree',
             xy=(pd.Timestamp('2015-10-06'), 1),
             xytext=(pd.Timestamp('2008-10-06'), -0.2),
             arrowprops={'arrowstyle': '->', 'color': 'gray'})
plt.show()

# %%
fig, ax = plt.subplots()

# Plot the relative temperature data
ax.plot(climate_change.index, climate_change['relative_temp'])
ax.set_xlabel('Time')
ax.tick_params('y')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()

# Annotate the date at which temperatures exceeded 1 degree
ax.annotate('>1 degree', xy=(pd.Timestamp('2015-10-06'), 1))

plt.show()

# %%
fig, ax = plt.subplots()

# Plot the CO2 levels time-series in blue
plot_timeseries(ax, climate_change.index, climate_change['co2'], 'blue', 'Time (years)', 'CO2 levels')

# Create an Axes object that shares the x-axis
ax2 = ax.twinx()

# Plot the relative temperature data in red
plot_timeseries(ax2, climate_change.index, climate_change['relative_temp'], 'red', 'Time (years)',
                'Relative temp (Celsius)')

# Annotate point with relative temperature >1 degree
ax2.annotate('>1 degree',
             xy=(pd.Timestamp('2015-10-06'), 1),
             xytext=(pd.Timestamp('2008-10-06'), -0.2),
             arrowprops={'arrowstyle': '->', 'color': 'gray'})

plt.show()

# %% 3. Quantitative comparisons: bar-charts

medals

# %%

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
plt.show()

# %%

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
plt.show()

# %%

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'], label='Gold')
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'], label='Silver')
ax.bar(medals.index, medals['Bronze'], bottom=medals['Gold']+medals['Silver'], label='Bronze')
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
ax.legend()
plt.show()

# %%

fig, ax = plt.subplots()

# Plot a bar-chart of gold medals as a function of country
ax.bar(medals.index, medals['Gold'])

# Set the x-axis tick labels to the country names
ax.set_xticklabels(medals.index, rotation=90)

# Set the y-axis label
ax.set_ylabel('Number of medals')

plt.show()

# %%

# Add bars for "Gold" with the label "Gold"
ax.bar(medals.index, medals['Gold'], label='Gold')

# Stack bars for "Silver" on top with label "Silver"
ax.bar(medals.index, medals['Silver'], bottom=medals['Gold'], label='Silver')

# Stack bars for "Bronze" on top of that with label "Bronze"
ax.bar(medals.index, medals['Bronze'], bottom=medals['Gold']+medals['Silver'], label='Bronze')

# Display the legend
ax.legend()

plt.show()

# %% 3.1 Quantitative comparisons: histograms

summer_2016

mens_rowing = summer_2016.query('Sex == "M" and Sport == "Rowing"')
mens_gymnastics = summer_2016.query('Sex == "M" and Sport == "Gymnastics"')

# %%

fig, ax = plt.subplots()
ax.bar('Rowing', mens_rowing['Height'].mean())
ax.bar('Gymnastics', mens_gymnastics['Height'].mean())
ax.set_ylabel('Height (cm)')
plt.show()

# %%

fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'])
ax.hist(mens_gymnastics['Height'])
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
plt.show()

# %%

fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label='Rowing')
ax.hist(mens_gymnastics['Height'], label='Gymnastics')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()

# %%

fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label='Rowing', bins=5)
ax.hist(mens_gymnastics['Height'], label='Gymnastics', bins=5)
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()

# %%

fig, ax = plt.subplots()
ax.hist(mens_rowing['Height'], label='Rowing', bins=[150, 160, 170, 180, 190, 200, 210], histtype='step')
ax.hist(mens_gymnastics['Height'], label='Gymnastics', bins=[150, 160, 170, 180, 190, 200, 210], histtype='step')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('# of observations')
ax.legend()
plt.show()

# %%

fig, ax = plt.subplots()
# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing['Weight'])

# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics['Weight'])

# Set the x-axis label to "Weight (kg)"
ax.set_xlabel('Weight (kg)')

# Set the y-axis label to "# of observations"
ax.set_ylabel('# of observations')

plt.show()

# %%

fig, ax = plt.subplots()

# Plot a histogram of "Weight" for mens_rowing
ax.hist(mens_rowing['Weight'], label='Rowing', histtype='step',  bins=5)

# Compare to histogram of "Weight" for mens_gymnastics
ax.hist(mens_gymnastics['Weight'], label='Gymnastics', histtype='step',  bins=5)

ax.set_xlabel("Weight (kg)")
ax.set_ylabel("# of observations")

# Add the legend and show the Figure
ax.legend()
plt.show()

# %% 3.2 Statistical plotting

# %% std
fig, ax = plt.subplots()
ax.bar("Rowing",
       mens_rowing['Height'].mean(),
       yerr=mens_rowing['Height'].std())
ax.bar("Gymnastics",
       mens_gymnastics['Height'].mean(),
       yerr=mens_gymnastics['Height'].std())
ax.set_ylabel("Height (cm)")
plt.show()

# %% line std

fig, ax = plt.subplots()
ax.errorbar(seattle_weather['MONTH'],
            seattle_weather['MLY-TAVG-NORMAL'],
            yerr=seattle_weather['MLY-TAVG-STDDEV'])
ax.errorbar(austin_weather['MONTH'],
            austin_weather['MLY-TAVG-NORMAL'],
            yerr=austin_weather['MLY-TAVG-STDDEV'])
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()

# %% boxplots

fig, ax = plt.subplots()
ax.boxplot([mens_rowing['Height'],
            mens_gymnastics['Height']])
ax.set_xticklabels(['Rowing', 'Gymnastics'])
ax.set_ylabel('Height (cm)')
plt.show()

# %% 3.2 Quantitative comparisons: scatter plots

# %%

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'])
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()

# %%

eighties = climate_change['1980-01-01':'1989-12-31']
nineties = climate_change['1990-01-01':'1999-12-31']

fig, ax = plt.subplots()
ax.scatter(eighties['co2'], eighties['relative_temp'],
           color='red', label='eighties')
ax.scatter(nineties['co2'], nineties['relative_temp'],
           color='blue', label='nineties')
ax.legend()
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()

# %%

fig, ax = plt.subplots()
ax.scatter(climate_change['co2'], climate_change['relative_temp'],
           c=climate_change.index)
ax.set_xlabel('CO2 (ppm)')
ax.set_ylabel('Relative temperature (Celsius)')
plt.show()

# %%

fig, ax = plt.subplots()

# Add data: "co2" on x-axis, "relative_temp" on y-axis
ax.scatter(climate_change['co2'], climate_change['relative_temp'])

# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel('CO2 (ppm)')

# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel('Relative temperature (C)')

plt.show()

# %%

fig, ax = plt.subplots()

# Add data: "co2", "relative_temp" as x-y, index as color
ax.scatter(climate_change['co2'], climate_change['relative_temp'],
           c=climate_change.index)

# Set the x-axis label to "CO2 (ppm)"
ax.set_xlabel('CO2 (ppm)')

# Set the y-axis label to "Relative temperature (C)"
ax.set_ylabel('Relative temperature (C)')

plt.show()

# %% 3.3 Preparing your figures to share with others

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

plt.style.use('default')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

plt.style.use('bmh')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

plt.style.use('grayscale')
fig, ax = plt.subplots()
ax.plot(seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"])
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
ax.set_xlabel('Time (months)')
ax.set_ylabel('Average temp. (Fahrenheit degree)')
plt.show()

# %%

# Use the "Solarize_Light2" style and create new Figure/Axes
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(austin_weather["MONTH"], austin_weather["MLY-TAVG-NORMAL"])
plt.show()

# %% 4. Saving your visualizations

# %%

plt.style.use('default')
fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
plt.show()

# %% png

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
fig.savefig('figs/gold_medals.png', dpi=300)

# %% jpg

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
fig.savefig('figs/gold_medals.jpg', quality=50)

# %% svg

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
fig.savefig('figs/gold_medals.svg')

# %% size

fig, ax = plt.subplots()
ax.bar(medals.index, medals['Gold'])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_xlabel('Number of medals')
fig.set_size_inches([5, 3])
fig.savefig('figs/gold_medals.jpg')

# %% 4.1 Automating figures from data

sports = summer_2016['Sport'].unique()

fig, ax = plt.subplots()
for sport in sports:
    sport_df = summer_2016[summer_2016['Sport'] == sport]
    ax.bar(sport, sport_df['Height'].mean(),
           yerr=sport_df['Height'].std())
ax.set_ylabel('Height (cm)')
ax.set_xticklabels(sports, rotation=90)
plt.show()

# %%

# Extract the "Sport" column
sports_column = summer_2016_medals["Sport"]

# Find the unique values of the "Sport" column
sports = sports_column.unique()

# Print out the unique sports values
print(sports)

# %%

fig, ax = plt.subplots()

# Loop over the different sports branches
for sport in sports:
  # Extract the rows only for this sport
  sport_df = summer_2016[summer_2016['Sport'] == sport]
  # Add a bar for the "Weight" mean with std y error bar
  ax.bar(sport, sport_df['Weight'].mean(),
           yerr=sport_df['Weight'].std())

ax.set_ylabel("Weight")
ax.set_xticklabels(sports, rotation=90)

# Save the figure to file
fig.savefig('figs/sports_weights.png')

