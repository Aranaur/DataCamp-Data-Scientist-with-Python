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
