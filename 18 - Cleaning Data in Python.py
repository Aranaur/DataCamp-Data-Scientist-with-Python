#%% Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import missingno as msno
import fuzzywuzzy
import recordlinkage

# Importing course datasets as DataFrames
ride_sharing = pd.read_csv('data/18/ride_sharing_new.csv', index_col='Unnamed: 0')
airlines = pd.read_csv('data/18/airlines_final.csv',  index_col='Unnamed: 0')
banking = pd.read_csv('data/18/banking_dirty.csv', index_col='Unnamed: 0')
restaurants = pd.read_csv('data/18/restaurants_L2.csv', index_col='Unnamed: 0')
restaurants_new = pd.read_csv('data/18/restaurants_L2_dirty.csv', index_col='Unnamed: 0')

#%% 1. Common data problems
