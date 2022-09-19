import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

late_shipments = pd.read_feather('data/29/late_shipments.feather')
stack_overflow = pd.read_feather('data/29/stack_overflow.feather')
dem_votes_potus_12_16 = pd.read_feather('data/29/dem_votes_potus_12_16.feather')
repub_votes_potus_08_12 = pd.read_feather('data/29/repub_votes_potus_08_12.feather')


#%% 1. Yum, That Dish Tests Good

#%% 1.1 To the lab for testing
