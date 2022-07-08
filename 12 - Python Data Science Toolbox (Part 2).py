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

import pandas as pd
result = []
for chunk in pd.read_csv('data/12/world_ind_pop_data.csv', chunksize=1000):
    result.append(sum(chunk['Urban population (% of total)']))

total = sum(result)
total

total = 0
for chunk in pd.read_csv('data/12/world_ind_pop_data.csv', chunksize=1000):
    total += sum(chunk['Urban population (% of total)'])
total

#%%

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('data/12/tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)

#%%


# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries(csv_file='data/12/tweets.csv', c_size=10, colname='lang')

# Print result_counts
print(result_counts)

#%% 2. List comprehensions

nums = [12, 8, 21, 3, 16]
new_nums = []
for num in nums:
    new_nums.append(num + 1)
print(new_nums)

nums = [12, 8, 21, 3, 16]
new_nums = [num + 1 for num in nums]
new_nums

result = [num for num in range(11)]
result

pairs_1 = []
for num1 in range(0, 2):
    for num2 in range(6, 8):
        pairs_1.append((num1, num2))
pairs_1

pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
pairs_2

#%%

doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']

[doc[0] for doc in doctor]

#%%

# Create list comprehension: squares
squares = [i ** 2 for i in range(0, 10)]

#%%

# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)

#%% 2.1 Advanced comprehensions

[num ** 2 for num in range(10) if num % 2 == 0]

[num ** 2 if num % 2 == 0 else 0 for num in range(10)]

pos_neg = {num: -num for num in range(9)}

#%%

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]

# Print the new list
print(new_fellowship)

#%%

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)

#%%

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member: len(member) for member in fellowship}

# Print the new dictionary
print(new_fellowship)

#%% 2.2 Introduction to generator expressions

[2 * num for num in range(10)]

(2 * num for num in range(10))

result = (num for num in range(6))
for num in result:
    print(num)

result = (num for num in range(6))
print(list(result))

result = (num for num in range(6))
print(next(result))

# [num for num if range(10**1000000)]  # store in memory
(num for num if range(10**1000000))  # lazy store in memory

even_nums = (num for num in range(10) if num % 2 ==0)
print(list(even_nums))

def num_sequence(n):
    """Generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1

result = num_sequence(5)
result
list(result)

result = num_sequence(5)
for i in result:
    print(i)

#%%

# List of strings
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# List comprehension
fellow1 = [member for member in fellowship if len(member) >= 7]

# Generator expression
fellow2 = (member for member in fellowship if len(member) >= 7)

#%%

# Create generator object: result
result = (num for num in range(0, 31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)

#%%

# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)

#%%

# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)

#%% 2.3 Wrapping up comprehensions and generators.

df = pd.read_csv('data/12/tweets.csv')

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)

#%%

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)

#%% 3. Welcome to the case study!
