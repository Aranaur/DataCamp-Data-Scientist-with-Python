# Importing course packages; you can add more too!
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timezone, timedelta
from dateutil import tz
import pickle

# Importing course datasets
rides = pd.read_csv('data/19/capital-onebike.csv')
with open('data/19/florida_hurricane_dates.pkl', 'rb') as f:
    florida_hurricane_dates = pickle.load(f)
florida_hurricane_dates = sorted(florida_hurricane_dates)

rides.head()  # Display the first five rows

#%% 1. Dates and Calendars

#%% 1.1 Dates in Python

two_hurricanes = ['10/7/2016', '6/21/2017']

two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
two_hurricanes_dates[0].year
two_hurricanes_dates[0].month
two_hurricanes_dates[0].day
two_hurricanes_dates[0].weekday()

#%%
# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())

#%%

# Counter for how many before June 1
early_hurricanes = 0

# We loop over the dates
for hurricane in florida_hurricane_dates:
    # Check if the month is before June (month number 6)
    if hurricane.month < 6:
        early_hurricanes = early_hurricanes + 1

print(early_hurricanes)

#%% 1.2 Math with dates
a = 11
b = 14
l = [a, b]
min(l)
b - a

d1 = date(2017, 11, 5)
d2 = date(2017, 12, 4)
l = [d1, d2]
print(min(l))
delta = d2 - d1
print(delta.days)

from datetime import timedelta
td = timedelta(days=29)
print(d1 + td)

#%%
# Import date
from datetime import date

# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)

#%%
# A dictionary to count hurricanes per calendar month
hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0,
                         7: 0, 8:0, 9:0, 10:0, 11:0, 12:0}

# Loop over all hurricanes
for hurricane in florida_hurricane_dates:
    # Pull out the month
    month = hurricane.month
    # Increment the count in your dictionary by one
    hurricanes_each_month[month] += 1

print(hurricanes_each_month)

#%%
# Print the first and last scrambled dates
print(dates_scrambled[0])
print(dates_scrambled[-1])

# Put the dates in order
dates_ordered = sorted(dates_scrambled)

# Print the first and last ordered dates
print(dates_ordered[0])
print(dates_ordered[-1])

#%% 1.3 Turning dates into strings

d = date(2017, 11, 5)
print(d)
print([d.isoformat()])

some_dates = ['2000-01-01', '1999-12-31']
sorted(some_dates)

d = date(2017, 1, 5)
d.strftime('%Y')
d.strftime('Year is %Y')
d.strftime('%Y/%m/%d')

#%%
# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)

#%%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-MM'
print(andrew.strftime('%Y-%m'))

#%%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'MONTH (YYYY)'
print(andrew.strftime('%B (%Y)'))

#%%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-DDD'
print(andrew.strftime('%Y-%j'))

#%% 2. Combining Dates and Times

#%% 2.1 Dates and times
