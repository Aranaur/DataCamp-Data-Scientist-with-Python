# Import the course packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
from sas7bdat import SAS7BDAT
from sqlalchemy import create_engine
import pickle

# %% 1. Introduction and flat files

# %% 1.1 Welcome to the course!

filename = 'data/16/seaslug.txt'
file = open(filename, mode='r')  # 'r' is to read, 'w' is to write
text = file.read()
file.close()

print(text)

with open(filename, 'r') as file:  # best practice
    print(file.read())


#%%

# Open a file: file
file = open('moby_dick.txt', 'r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)

#%%

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())

#%% 1.2 The importance of flat files in data science



# %% 2. Importing data from other file types


# %% 3. Working with relational databases in Python
