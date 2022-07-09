# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing course datasets as DataFrames
bike_share = pd.read_csv('data/13/bike_share.csv')
college_data = pd.read_csv('data/13/college_datav3.csv')
daily_show = pd.read_csv('data/13/daily_show_guests_cleaned.csv')
insurance = pd.read_csv('data/13/insurance_premiums.csv')
grants = pd.read_csv('data/13/schoolimprovement2010grants.csv', index_col=0)

bike_share.head()  # Display the first five rows of this DataFrame

#%% 1. Seaborn Introduction

#%% 1.1 Introduction to Seaborn
