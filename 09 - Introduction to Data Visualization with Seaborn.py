# %% Importing course packages; you can add more too!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Importing course datasets as DataFrames

country_data = pd.read_csv('data/09/countries-of-the-world.csv', decimal=",")
mpg = pd.read_csv('data/09/mpg.csv')
student_data = pd.read_csv('data/09/student-alcohol-consumption.csv', index_col=0)
survey = pd.read_csv('data/09/young-people-survey-responses.csv', index_col=0)

survey.head()

# %% 1. Introduction to Seaborn

