import pandas as pd
import numpy as np

attrition = pd.read_feather('data/28/attrition.feather')
spotify_population = pd.read_feather('data/28/spotify_2000_2020.feather')
coffee_ratings_full = pd.read_feather('data/28/coffee_ratings_full.feather')


#%% 1. Bias Any Stretch of the Imagination

#%% 1.1 Living the sample life
coffee_ratings_full.head()

np.random.seed(1987)

pts_vs_flavor_pop = coffee_ratings_full[['total_cup_points', 'flavor']]
pts_vs_flavor_samp = pts_vs_flavor_pop.sample(n=10)
pts_vs_flavor_samp

cup_points_samp = coffee_ratings_full['total_cup_points'].sample(n=10)

np.mean(pts_vs_flavor_pop['total_cup_points'])
np.mean(cup_points_samp)

pts_vs_flavor_pop['flavor'].mean()
pts_vs_flavor_samp['flavor'].mean()