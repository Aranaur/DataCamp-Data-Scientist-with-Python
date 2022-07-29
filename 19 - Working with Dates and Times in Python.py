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

# %% 1. Dates and Calendars

# %% 1.1 Dates in Python

two_hurricanes = ['10/7/2016', '6/21/2017']

two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
two_hurricanes_dates[0].year
two_hurricanes_dates[0].month
two_hurricanes_dates[0].day
two_hurricanes_dates[0].weekday()

# %%
# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())

# %%

# Counter for how many before June 1
early_hurricanes = 0

# We loop over the dates
for hurricane in florida_hurricane_dates:
    # Check if the month is before June (month number 6)
    if hurricane.month < 6:
        early_hurricanes = early_hurricanes + 1

print(early_hurricanes)

# %% 1.2 Math with dates
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

# %%
# Import date
from datetime import date

# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)

# %%
# A dictionary to count hurricanes per calendar month
hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
                         7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

# Loop over all hurricanes
for hurricane in florida_hurricane_dates:
    # Pull out the month
    month = hurricane.month
    # Increment the count in your dictionary by one
    hurricanes_each_month[month] += 1

print(hurricanes_each_month)

# %%
# Print the first and last scrambled dates
print(dates_scrambled[0])
print(dates_scrambled[-1])

# Put the dates in order
dates_ordered = sorted(dates_scrambled)

# Print the first and last ordered dates
print(dates_ordered[0])
print(dates_ordered[-1])

# %% 1.3 Turning dates into strings

d = date(2017, 11, 5)
print(d)
print([d.isoformat()])

some_dates = ['2000-01-01', '1999-12-31']
sorted(some_dates)

d = date(2017, 1, 5)
d.strftime('%Y')
d.strftime('Year is %Y')
d.strftime('%Y/%m/%d')

# %%
# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)

# %%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-MM'
print(andrew.strftime('%Y-%m'))

# %%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'MONTH (YYYY)'
print(andrew.strftime('%B (%Y)'))

# %%
# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-DDD'
print(andrew.strftime('%Y-%j'))

# %% 2. Combining Dates and Times

# %% 2.1 Dates and times

from datetime import datetime

dt = datetime(2017, 10, 1, 15, 23, 25, 500000)

dt_hr = dt.replace(minute=0, second=0, microsecond=0)
print(dt_hr)

# %%
# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Print the results in ISO 8601 format
print(dt.isoformat())

# %%
# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Print the results in ISO 8601 format
print(dt.isoformat())

# %%
# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt_old)

# %%

# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}

# Loop over all trips
for trip in onebike_datetimes:
    # Check to see if the trip starts before noon
    if trip['start'].hour < 12:
        # Increment the counter for before noon
        trip_counts['AM'] += 1
    else:
        # Increment the counter for after noon
        trip_counts['PM'] += 1

print(trip_counts)

# %% 2.2 Printing and parsing datetimes

dt = datetime(2017, 12, 30, 15, 19, 13)

print(dt.strftime('%Y-%m-%d'))
print(dt.strftime('%Y-%m-%d %H:%M:%S'))
print(dt.strftime('%H:%M:%S on %Y/%m/%d'))
print(dt.isoformat())

dt = datetime.strptime('12/30/2017 15:19:13', '%m/%d/%Y %H:%M:%S')
print(type(dt))
print(dt)

ts = 1514665153.0  # sec since 1 January 1970
print(datetime.fromtimestamp(ts))

#%%
# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD HH:MM:SS format
s = '2017-02-03 00:00:01'

# Write a format string to parse s
fmt = '%Y-%m-%d %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)

#%%
# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD format
s = '2030-10-15'

# Write a format string to parse s
fmt = '%Y-%m-%d'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)

#%%
# Import the datetime class
from datetime import datetime

# Starting string, in MM/DD/YYYY HH:MM:SS format
s = '12/15/1986 08:00:00'

# Write a format string to parse s
fmt = '%m/%d/%Y %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)

#%%
# Write down the format string
fmt = "%Y-%m-%d %H:%M:%S"

# Initialize a list for holding the pairs of datetime objects
onebike_datetimes = []

# Loop over all trips
for (start, end) in onebike_datetime_strings:
    trip = {'start': datetime.strptime(start, fmt),
            'end': datetime.strptime(end, fmt)}

    # Append the trip
    onebike_datetimes.append(trip)

#%%
# Import datetime
from datetime import datetime

# Pull out the start of the first trip
first_start = onebike_datetimes[0]['start']

# Format to feed to strftime()
fmt = "%Y-%m-%dT%H:%M:%S"

# Print out date with .isoformat(), then with .strftime() to compare
print(first_start.isoformat())
print(first_start.strftime(fmt))

#%%
# Import datetime
from datetime import datetime

# Starting timestamps
timestamps = [1514665153, 1514664543]

# Datetime objects
dts = []

# Loop
for ts in timestamps:
    dts.append(datetime.fromtimestamp(ts))

# Print results
print(dts)

#%% 2.3 Working with durations

start = datetime(2017, 10, 8, 23, 46, 47)
end = datetime(2017, 10, 9, 0, 10, 57)

duration = end - start
duration.total_seconds()

delta1 = timedelta(seconds=1)
print(start)
print(start + delta1)

delta2 = timedelta(days=1, seconds=1)
print(start + delta2)

delta3 = timedelta(weeks=-1)
print(start + delta3)

delta4 = timedelta(weeks=1)
print(start - delta4)

#%%
# Initialize a list for all the trip durations
onebike_durations = []

for trip in onebike_datetimes:
    # Create a timedelta object corresponding to the length of the trip
    trip_duration = trip['end'] - trip['start']

    # Get the total elapsed seconds in trip_duration
    trip_length_seconds = trip_duration.total_seconds()

    # Append the results to our list
    onebike_durations.append(trip_length_seconds)

#%%
# What was the total duration of all trips?
total_elapsed_time = sum(onebike_durations)

# What was the total number of trips?
number_of_trips = len(onebike_durations)

# Divide the total duration by the number of trips
print(total_elapsed_time / number_of_trips)

#%%
# Calculate shortest and longest trips
shortest_trip = min(onebike_durations)
longest_trip = max(onebike_durations)

# Print out the results
print("The shortest trip was " + str(shortest_trip) + " seconds")
print("The longest trip was " + str(longest_trip) + " seconds")

#%% 3. Time Zones and Daylight Saving

#%% 3.1 UTC offsets
from datetime import datetime, timezone, timedelta

ET = timezone(timedelta(hours=-5))
dt = datetime(2017, 12, 30, 15, 9, 3, tzinfo=ET)
print(dt)

IST = timezone(timedelta(hours=5, minutes=30))
print(dt.astimezone(IST))

print(dt)
print(dt.replace(tzinfo=timezone.utc))
print(dt.astimezone(timezone.utc))

#%%
# Import datetime, timezone
from datetime import datetime, timezone, timedelta

# October 1, 2017 at 15:26:26, UTC
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=timezone.utc)

# Print results
print(dt.isoformat())

#%%
# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Pacific Standard Time, or UTC-8
pst = timezone(timedelta(hours=-8))

# October 1, 2017 at 15:26:26, UTC-8
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=pst)

# Print results
print(dt.isoformat())

#%%
# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Australian Eastern Daylight Time, or UTC+11
aedt = timezone(timedelta(hours=11))

# October 1, 2017 at 15:26:26, UTC+11
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=aedt)

# Print results
print(dt.isoformat())

#%%
# Create a timezone object corresponding to UTC-4
edt = timezone(timedelta(hours=-4))

# Loop over trips, updating the start and end datetimes to be in UTC-4
for trip in onebike_datetimes[:10]:
    # Update trip['start'] and trip['end']
    trip['start'] = trip['start'].replace(tzinfo=edt)
    trip['end'] = trip['end'].replace(tzinfo=edt)

#%%
# Loop over the trips
for trip in onebike_datetimes[:10]:
    # Pull out the start
    dt = trip['start']
    # Move dt to be in UTC
    dt = dt.astimezone(timezone.utc)

    # Print the start time in UTC
    print('Original:', trip['start'], '| UTC:', dt.isoformat())

#%% 3.2 Time zone database
from datetime import datetime
from dateutil import tz

et = tz.gettz('America/New_York')
last = datetime(2017, 12, 30, 15, 9, 3, tzinfo=et)
print(last)

first = datetime(2017, 10, 1, 15, 23, 25, tzinfo=et)
print(first)

#%%
# Import tz
from dateutil import tz

# Create a timezone object for Eastern Time
et = tz.gettz('America/New_York')

# Loop over trips, updating the datetimes to be in Eastern Time
for trip in onebike_datetimes[:10]:
    # Update trip['start'] and trip['end']
    trip['start'] = trip['start'].replace(tzinfo=et)
    trip['end'] = trip['end'].replace(tzinfo=et)

#%%
# Create the timezone object
uk = tz.gettz('Europe/London')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in the UK?
notlocal = local.astimezone(uk)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())

#%%
# Create the timezone object
ist = tz.gettz('Asia/Kolkata')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in India?
notlocal = local.astimezone(ist)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())

#%%
# Create the timezone object
sm = tz.gettz('Pacific/Apia')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in Samoa?
notlocal = local.astimezone(sm)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())

#%% 3.3 Starting daylight saving time

spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59)
spring_ahead_159am.isoformat()

spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0)
spring_ahead_3am.isoformat()

(spring_ahead_3am - spring_ahead_159am).total_seconds()

EST = timezone(timedelta(hours=-5))
EDT = timezone(timedelta(hours=-4))

spring_ahead_159am = spring_ahead_159am.replace(tzinfo=EST)
spring_ahead_159am.isoformat()
spring_ahead_3am = spring_ahead_3am.replace(tzinfo=EDT)
spring_ahead_3am.isoformat()
(spring_ahead_3am - spring_ahead_159am).total_seconds()

eastern = tz.gettz('America/New_York')
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59, tzinfo=eastern)
spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0, tzinfo=eastern)

#%%
# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo=tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
print(start.isoformat() + " to " + end.isoformat())

#%%
# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo=tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
print(start.isoformat() + " to " + end.isoformat())

# How many hours have elapsed?
print((end - start).total_seconds()/(60*60))

#%%
# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo=tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
print(start.isoformat() + " to " + end.isoformat())

# How many hours have elapsed?
print((end - start).total_seconds()/(60*60))

# What if we move to UTC?
print((end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()/(60*60))

#%%
# Import datetime and tz
from datetime import datetime
from dateutil import tz

# Create starting date
dt = datetime(2000, 3, 29, tzinfo = tz.gettz('Europe/London'))

# Loop over the dates, replacing the year, and print the ISO timestamp
for y in range(2000, 2011):
    print(dt.replace(year=y).isoformat())

#%% 3.4 Ending daylight saving time

eastern = tz.gettz('US/Eastern')
first_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
tz.datetime_exists(first_1am)

second_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
second_1am = tz.enfold(second_1am)
(first_1am - second_1am).total_seconds()

first_1am = first_1am.astimezone(tz.UTC)
second_1am = second_1am.astimezone(tz.UTC)
(first_1am - second_1am).total_seconds()

#%%
# Loop over trips
for trip in onebike_datetimes:
    # Rides with ambiguous start
    if tz.datetime_ambiguous(trip['start']):
        print("Ambiguous start at " + str(trip['start']))
    # Rides with ambiguous end
    if tz.datetime_ambiguous(trip['end']):
        print("Ambiguous end at " + str(trip['end']))

#%%
trip_durations = []
for trip in onebike_datetimes:
    # When the start is later than the end, set the fold to be 1
    if trip['start'] > trip['end']:
        trip['end'] = tz.enfold(trip['end'])
    # Convert to UTC
    start = trip['start'].astimezone(tz.UTC)
    end = trip['end'].astimezone(tz.UTC)

    # Subtract the difference
    trip_length_seconds = (end-start).total_seconds()
    trip_durations.append(trip_length_seconds)

# Take the shortest trip duration
print("Shortest trip: " + str(min(trip_durations)))

#%% 4. Easy and Powerful: Dates and Times in Pandas

#%% 4.1 Reading date and time data in Pandas
