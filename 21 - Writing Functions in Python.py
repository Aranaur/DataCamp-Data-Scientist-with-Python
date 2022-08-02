# %% 1. Writing Functions in Python

# %% 1.1 Docstrings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_and_stack(df, new_names):
    """Split a DataFrame's columns into two halves and then stack
    them vertically, returning a new Dataframe with `new_names` as the columns names.

    Args:
        df (DataFrame): The DataFrame to split.
        new_names (iterable of str): The column names for the new DataFrame.

    Returns:
        DataFrame
    """
    half = int(len(df.columns) / 2)
    left = df.iloc[:, :half]
    right = df.iloc[:, half:]
    return pd.DataFrame(
        data=np.vstack([left.values, right.values]),
        columns=new_names
    )


def function_name(arguments):
    """
    Description of what the function does.

    Description of the arguments, if any.

    Description of the return value(s), if any.

    Description of errors raised, if any.

    Optional extra notes or examples of usage
    """


# Google style
def funbction(arg_1, arg_2=42):
    """Description of what the function does.

    Args:
        arg_1 (str):
        arg_2 (int, optional):

    Returns:
        bool:Optional description if the return value
        Extra lines are nor indented.

    Raises:
        ValueError:

    Notes:
        See https://...
        for more info.
    """


# Numpydoc
def function(arg_1, arg_2):
    """
    Description of what the function does.

    Parameters
    ----------
    arg_1 :
    arg_2 :

    Returns
    ----------
    The type of the return value
    """


# Retrieving docstrings
def the_answer():
    """Return the answer of life,
    the universe, and everything

    Returns:
    int
    """
    return 42

print(the_answer.__doc__)

import inspect
print(inspect.getdoc(the_answer))


#%%
# Add a docstring to count_letter()
def count_letter(content, letter):
    """Count the number of times `letter` appears in `content`.
    """
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('`letter` must be a single character string.')
    return len([char for char in content if char == letter])


#%%
def count_letter(content, letter):
    """Count the number of times `letter` appears in `content`.

    # Add a Google style arguments section
    Args:
      content (str): The string to search.
      letter (str): The letter to search for.
    """
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('`letter` must be a single character string.')
    return len([char for char in content if char == letter])


#%%
def count_letter(content, letter):
    """Count the number of times `letter` appears in `content`.

    Args:
      content (str): The string to search.
      letter (str): The letter to search for.

    # Add a returns section
    Returns:
      int
    """
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('"letter" must be a single character string.')
    return len([char for char in content if char == letter])


#%%
def count_letter(content, letter):
    """Count the number of times `letter` appears in `content`.

    Args:
      content (str): The string to search.
      letter (str): The letter to search for.

    Returns:
      int

    # Add a section detailing what errors might be raised
    Raises:
      ValueError: If `letter` is not a one-character string.
    """
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('`letter` must be a single character string.')
    return len([char for char in content if char == letter])


#%%
# Get the "count_letter" docstring by using an attribute of the function
docstring = count_letter.__doc__

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))

#%%
import inspect

# Inspect the count_letter() function to get its docstring
docstring = inspect.getdoc(count_letter)

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))

#%%
import inspect


def build_tooltip(function):
    """Create a tooltip for any function that shows the
    function's docstring.

    Args:
      function (callable): The function we want a tooltip for.

    Returns:
      str
    """
    # Get the docstring for the "function" argument by using inspect
    docstring = inspect.getdoc(function)
    border = '#' * 28
    return '{}\n{}\n{}'.format(border, docstring, border)

print(build_tooltip(count_letter))
print(build_tooltip(range))
print(build_tooltip(print))

#%%
numpy.fywdkxa.__doc__

#%% 1.2 DRY and "Do One Thing"


def load_and_plot(path):
    """Load a dataset and plot the first two PCA.

    Args:
        path (str): The location of .csv file

    Returns:
        tuple of ndarray: (features, labels)
    """
    data = pd.read_csv(path)
    y = data['label'].values
    X = data[col for col in train.columns if col != ' label'].values
    pca = PCA(n_components=2).fit_transform(X)
    plt.scatter(pca[:, 0], pca[:, 1])
    return X, y


train_X, tain_y = load_and_plot('train.csv')
val_X, val_y = load_and_plot('validation.csv')
test_X, test_y = load_and_plot('test.csv')


def load_data(path):
    """Load a dataset.

    Args:
        path (str): The location of .csv file

    Returns:
        tuple of ndarray: (features, labels)
    """
    data = pd.read_csv(path)
    y = data['label'].values
    X = data[col for col in train.columns if col != ' label'].values
    return X, y


def plot_data(X):
    """Plot the first two PCA of a matrix

    Args:
        X (numpy.ndarray): The data to plot
    """
    pca = PCA(n_components=2).fit_transform(X)
    plt.scatter(pca[:, 0], pca[:, 1])

#%%


def standardize(column):
    """Standardize the values in a column.

    Args:
      column (pandas Series): The data to standardize.

    Returns:
      pandas Series: the values as z-scores
    """
    # Finish the function so that it returns the z-scores
    z_score = (column - column.mean()) / column.std()
    return z_score

# Use the standardize() function to calculate the z-scores
df['y1_z'] = standardize(df['y1_gpa'])
df['y2_z'] = standardize(df['y2_gpa'])
df['y3_z'] = standardize(df['y3_gpa'])
df['y4_z'] = standardize(df['y4_gpa'])