#%% Import the course packages
import pandas as pd
import matplotlib.pyplot as plt

#%% Import the course datasets as DataFrames
world_ind = pd.read_csv('data/12/world_ind_pop_data.csv')
tweets = pd.read_csv('data/12/tweets.csv')

# Preview the first DataFrame
world_ind

#%% 1. Using iterators in PythonLand

#%% 1.1 Introduction to iterators

employees = ['Nick', 'Lore', 'Hugo']

for name in employees:
    print(name)

for letter in 'DataCamp':
    print(letter)

for i in range(4):
    print(i)

word = 'Da'
it = iter(word)
next(it)

word = 'Data'
it = iter(word)
print(*it)

pythonistas = {'hugo': 'bowne-anderson', 'francis': ' castro'}
for key, value in pythonistas.items():
    print(key, value)

file = open('data/12/file.txt')
it = iter(file)
print(next(it))

#%%

# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for name in flash:
    print(name)

# Create an iterator for flash: superhero
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

#%%

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for i in range(3):
    print(i)


# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

#%%

# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)

#%% 1.2 Playing with iterators

# enumerate()
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e))
e_list = list(e)
print(e_list)

for index, value in enumerate(avengers):
    print(index, value)

for index, value in enumerate(avengers, start=10):
    print(index, value)

# zip()
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(type(z))
z_list = list(z)
print(z_list)

for z1, z2 in zip(avengers, names):
    print(z1, z2)

print(*z)

#%%

# Create a list of strings: mutants
mutants = ['charles xavier',
           'bobby drake',
           'kurt wagner',
           'max eisenhardt',
           'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)

#%%

aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
powers = ['telepathy',
          'thermokinesis',
          'teleportation',
          'magnetokinesis',
          'intangibility']

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

#%%

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

#%% 1.3 Using iterators to load large files into memory
