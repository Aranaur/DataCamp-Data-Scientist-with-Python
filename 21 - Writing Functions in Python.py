# %% 1. Writing Functions in Python

# %% 1.1 Docstrings
import contextlib
import os

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


#%%
def mean(values):
    """Get the mean of a sorted list of values

    Args:
      values (iterable of float): A list of numbers

    Returns:
      float
    """
    # Write the mean() function
    mean = sum(values) / len(values)
    return mean


#%%
def median(values):
    """Get the median of a sorted list of values

    Args:
      values (iterable of float): A list of numbers

    Returns:
      float
    """
    # Write the median() function
    midpoint = int(len(values) / 2)
    if len(values) % 2 == 0:
        median = (values[midpoint - 1] + values[midpoint]) / 2
    else:
        median = values[midpoint]
    return median


#%% 1.3 Pass by assignment
def foo(x):
    x[0] = 99

my_list = [1, 2, 3]
foo(my_list)  # list is mutable


def bar(x):
    x = x + 90

my_var = 3
bar(my_var)
my_var  # int can`t be changed


#%%
a = [1, 2, 3]
b = a
a.append(4)
b
b.append(5)
a

# Mutable: list, dict, set, bytearray, object, functions etc
# Immutable: int, float, bool, string, bytes, tuple, frozenset, None

def foo(var=[]):
    var.append(1)
    return var


foo()
foo()


def foo(var=None):
    if var is None:
        var=[]
    var.append(1)
    return var

foo()
foo()

#%%
def store_lower(_dict, _string):
    """Add a mapping between `_string` and a lowercased version of `_string` to `_dict`

    Args:
      _dict (dict): The dictionary to update.
      _string (str): The string to add.
    """
    orig_string = _string
    _string = _string.lower()
    _dict[orig_string] = _string

d = {}
s = 'Hello'

store_lower(d, s)

# d = {'Hello': 'hello'}, s = 'Hello'


#%%
# Use an immutable variable for the default argument
def better_add_column(values, df=None):
    """Add a column of `values` to a DataFrame `df`.
    The column will be named "col_<n>" where "n" is
    the numerical index of the column.

    Args:
      values (iterable): The values of the new column
      df (DataFrame, optional): The DataFrame to update.
        If no DataFrame is passed, one is created by default.

    Returns:
      DataFrame
    """
    # Update the function to create a default DataFrame
    if df is None:
        df = pandas.DataFrame()
    df['col_{}'.format(len(df.columns))] = values
    return df


#%% 2. Context Managers

#%% 2.1 Using context managers
with open('my_file.txt') as my_file:
    text = my_file.read()
    length = len(text)

print('The file is {} characters long'.format(length))

# with <context-manager>(<args>):
#     # Run code
#     # This code is running inside the context
#
# # This code runs after the context is removed

#%%
# Open "alice.txt" and assign the file to "file"
with open('alice.txt') as file:
    text = file.read()

n = 0
for word in text.split():
    if word.lower() in ['cat', 'cats']:
        n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))

#%%
image = get_image_from_instagram()

# Time how long process_with_numpy(image) takes to run
with timer():
    print('Numpy version')
    process_with_numpy(image)

# Time how long process_with_pytorch(image) takes to run
with timer():
    print('Pytorch version')
    process_with_pytorch(image)

#%% 2.2 Writing context managers

@contextlib.contextmanager
def my_context():
    # add ant set up code
    yield
    # add any teardown code

#%%
@contextlib.contextmanager
def my_context():
    print('hello')
    yield 42
    print('goodbye')

with my_context() as foo:
    print('foo is {}'.format(foo))

#%%
@contextlib.contextmanager
def database(url):
    # set up database conn
    df = posgres.connect(url)

    yield db

    # tear down database conn
    db.disconnect()

url = 'http://datacamp.com/data'
with database(url) as my_db:
    course_list = my_db.execute(
        'SELECT * FROM courses'
    )

#%%
@contextlib.contextmanager
def in_dir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)

with in_dir('/data/project_1'):
    project_files = os.listdir()

#%%
# Add a decorator that will make timer() a context manager
@contextlib.contextmanager
def timer():
    """Time the execution of a context block.

    Yields:
      None
    """
    start = time.time()
    # Send control back to the context block
    yield
    end = time.time()
    print('Elapsed: {:.2f}s'.format(end - start))

with timer():
    print('This should take approximately 0.25 seconds')
    time.sleep(0.25)

#%%
@contextlib.contextmanager
def open_read_only(filename):
    """Open a file in read-only mode.

    Args:
      filename (str): The location of the file to read

    Yields:
      file object
    """
    read_only_file = open(filename, mode='r')
    # Yield read_only_file so it can be assigned to my_file
    yield read_only_file
    # Close read_only_file
    read_only_file.close()

with open_read_only('my_file.txt') as my_file:
    print(my_file.read())

#%% 2.3 Advanced topics

# not bad
def copy(src, dst):
    """Copy the contents of one file to another
    :arg
        src (str): File name of the file to be copied.
        dst (str): Where to write the new file.
    """
    with open(src) as f_src:
        contents = f_src.read()
    with open(dst, 'w') as f_dst:
        f_dst.write(contents)

# better
def copy(src, dst):
    """Copy the contents of one file to another
    :arg
        src (str): File name of the file to be copied.
        dst (str): Where to write the new file.
    """
    with open(src) as f_src:
        with open(dst) as f_dst:
            for line in f_src:
                f_dst.write(line)

#%%
def get_printer(ip):
    p = connect_to_printer(ip)
    yield
    p.disconnect()
    print('disconnected from printer')

doc = {'text': 'This is my text.'}

with get_printer('10.0.34.111') as printer:
    printer.print_page(doc['txt'])  # error here: txt != text


# try:
#   code that might raise an error
# except:
#   do something about the error
# finally:
#   this code runs no matter what

def get_printer(ip):
    p = connect_to_printer(ip)

    try:
        yield
    finally:
        p.disconnect()
        print('disconnected from printer')

with get_printer('10.0.34.111') as printer:
    printer.print_page(doc['txt'])  # error here: txt != text

#%%
# Use the "stock('NVDA')" context manager
# and assign the result to the variable "nvda"
with stock('NVDA') as nvda:
    # Open 'NVDA.txt' for writing as f_out
    with open('NVDA.txt', 'w') as f_out:
        for _ in range(10):
            value = nvda.price()
            print('Logging ${:.2f} for NVDA'.format(value))
            f_out.write('{:.2f}\n'.format(value))

#%%
def in_dir(directory):
    """Change current working directory to `directory`,
    allow the user to run some code, and change back.

    Args:
      directory (str): The path to a directory to work in.
    """
    current_dir = os.getcwd()
    os.chdir(directory)

    # Add code that lets you handle errors
    try:
        yield
    # Ensure the directory is reset,
    # whether there was an error or not
    finally:
        os.chdir(current_dir)

#%% 3. Decorators

#%% 3.1 Functions are objects
