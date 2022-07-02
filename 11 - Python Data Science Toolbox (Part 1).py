# Importing course packages; you can add more too!
import pandas as pd
from functools import reduce

# Importing course datasets as DataFrames
tweets_df = pd.read_csv('data/11/tweets.csv')
df = pd.read_csv('data/11/tweets.csv')

tweets.head()

# %% 1. Writing your own functions

# %% 1.1 User-defined functions

x = str(5)
x
type(x)


def square():
    new_value = 4 ** 2
    print(new_value)


square()


def square(value):
    new_value = value ** 2
    print(new_value)


square(4)
square(5)


def square(value):
    new_value = value ** 2
    return new_value


num = square(4)
num


def square(value):
    """Return the square of a value."""
    new_value = value ** 2
    return new_value


# %%

# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)


# Call shout
shout()


# %%

# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)


# Call shout with the string 'congratulations'
shout('congratulations')


# %%

# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Replace print with return
    return (shout_word)


# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print(yell)


# %% 1.2 Multiple parameters and return values

def raise_to_power(value1, value2):
    """Raise value1 to the power of value2"""
    new_value = value1 ** value2
    return new_value


result = raise_to_power(2, 3)
result

even_nums = (2, 4, 6)  # tuples - unchangeable
type(even_nums)

a, b, c = even_nums  # unpack tuples
even_nums[1]
second_num = even_nums[1]
second_num


def raise_both(value1, value2):
    """Raise value1 to the power of value2
    and vice versa"""
    new_value1 = value1 ** value2
    new_value2 = value2 ** value1
    new_tuple = (new_value1, new_value2)
    return new_tuple


result = raise_both(2, 3)
result


# %%

# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'

    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'

    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2

    # Return new_shout
    return new_shout


# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations', 'you')

# Print yell
print(yell)

# %%

# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2, num2, num3)


# %%


# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'

    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'

    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words


# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)

# %% 1.3 Bringing it all together

tweets
df

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:
    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)


# %%

# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    # Initialize an empty dictionary: langs_count
    langs_count = {}
    # Extract column from DataFrame: col
    col = df[col_name]
    # Iterate over lang column in DataFrame
    for entry in col:
        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1
    # Return the langs_count dictionary
    return langs_count


# Call count_entries(): result
result = count_entries(tweets_df, 'lang')
# Print the result
print(result)

# %% 2. Default arguments, variable-length arguments and scope

# %% 2.1 Scope and user-defined functions

# Scopes: Global, Local, Built-in

new_val = 10


def square(value):
    """Return the square of a value."""
    global new_val
    new_val = new_val ** 2
    return new_val


square(3)
new_val

# %%

# Create a string: team
team = "teen titans"


# Define change_team()
def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team = "justice league"


# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)

# %%

import builtins

dir(builtins)

# array not in builtins

#%% 2.1 Nested functions

# def outer(...):
#     """..."""
#     x = ...
#     def inner(...):
#         """..."""
#         y = x ** 2
#         return ...


def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values"""
    new_x1 = x1 % 2 + 5
    new_x2 = x2 % 2 + 5
    new_x3 = x3 % 2 + 5
    return new_x1, new_x2, new_x3


def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values"""
    def inner(x):
        """Returns the remainder plus 5 of three values"""
        return x % 2 + 5
    return inner(x1), inner(x2), inner(x3)


mod2plus5(1, 2, 3)
