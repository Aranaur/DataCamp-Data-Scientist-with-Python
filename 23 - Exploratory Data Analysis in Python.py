# Importing course packages; you can add more too!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.interpolate
import statsmodels.formula.api as smf

# Importing course datasets as DataFrames
brfss = pd.read_hdf('data/23/brfss.hdf5', 'brfss') # Behavioral Risk Factor Surveillance System (BRFSS)
gss = pd.read_hdf('data/23/gss.hdf5', 'gss') # General Social Survey (GSS)
nsfg = pd.read_hdf('data/23/nsfg.hdf5', 'nsfg') # National Survey of Family Growth (NSFG)

brfss.head() # Display the first five rows