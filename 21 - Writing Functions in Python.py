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
def my_function():
    print('Hello!')

x = my_function
type(x)
x()

list_of_functions = [my_function, open, print]
list_of_functions[2]("I'm printing with an element of a list!")

#%%
dict_of_functions = {
    'func1': my_function,
    'func2': open,
    'func3': print
}

dict_of_functions['func3']("I'm printing with a value of a dict!")

def my_function():
    return 42

x = my_function
my_function()
x()

#%%
def hes_docstring(func):
    """Check to see if the function `func`
    has a docstring

    :arg:
        func (callable): A function

    :returns
        bool
    """
    return func.__doc__ is not None

def no():
    return 42

def yes():
    """Return the value 42
    """
    return 42

hes_docstring(no)
hes_docstring(yes)

#%%
def foo():
    x = [3, 6, 9]

    def bar(y):
        print(y)

    for value in x:
        bar(x)

#%%
def foo(x, y):
    if 4 < x < 10 and 4 < y < 10:
        print(x * y)

def foo(x, y):
    def in_range(v):
        return 4 < v < 10

    if in_range(x) and in_range(y):
        print(x * y)

#%%
def get_function():
    def print_me(s):
        print(s)
    return print_me

new_func = get_function()
new_func('This is sentence.')

#%%
# Add the missing function references to the function map
function_map = {
    'mean': mean,
    'std': std,
    'minimum': minimum,
    'maximum': maximum
}

data = load_data()
print(data)

func_name = get_user_input()

# Call the chosen function and pass "data" as an argument
function_map[func_name](data)

#%%
# Call has_docstring() on the load_and_plot_data() function
ok = has_docstring(load_and_plot_data)

if not ok:
    print("load_and_plot_data() doesn't have a docstring!")
else:
    print("load_and_plot_data() looks ok")

#%%
# Call has_docstring() on the as_2D() function
ok = has_docstring(as_2D)

if not ok:
    print("as_2D() doesn't have a docstring!")
else:
    print("as_2D() looks ok")

#%%
# Call has_docstring() on the log_product() function
ok = has_docstring(log_product)

if not ok:
    print("log_product() doesn't have a docstring!")
else:
    print("log_product() looks ok")

#%%
def create_math_function(func_name):
    if func_name == 'add':
        def add(a, b):
            return a + b
        return add
    elif func_name == 'subtract':
        # Define the subtract() function
        def subtract(a, b):
            return a - b
        return subtract
    else:
        print("I don't know that one")

add = create_math_function('add')
print('5 + 2 = {}'.format(add(5, 2)))

subtract = create_math_function('subtract')
print('5 - 2 = {}'.format(subtract(5, 2)))

#%% 3.2 Scope
x = 7
y = 200
print(x)

def foo():
    x = 42
    print(x)
    print(y)

foo()
print(x)

#%%
x = 7

def foo():
    global x
    x = 42
    print(x)

foo()
print(x)

#%%
def foo():
    x = 10
    def bar():
        x = 200
        print(x)
    bar()
    print(x)

foo()

def foo():
    x = 10
    def bar():
        nonlocal x
        x = 200
        print(x)
    bar()
    print(x)

foo()

#%%
x = 50

def one():
    x = 10

def two():
    global x
    x = 30

def three():
    x = 100
    print(x)

for func in [one, two, three]:
    func()
    print(x)

# 50, 30, 100, 30

#%%
call_count = 0

def my_function():
    # Use a keyword that lets us update call_count
    global call_count
    call_count += 1

    print("You've called my_function() {} times!".format(
        call_count
    ))

for _ in range(20):
    my_function()

#%%
def read_files():
    file_contents = None

    def save_contents(filename):
        # Add a keyword that lets us modify file_contents
        nonlocal file_contents
        if file_contents is None:
            file_contents = []
        with open(filename) as fin:
            file_contents.append(fin.read())

    for filename in ['1984.txt', 'MobyDick.txt', 'CatsEye.txt']:
        save_contents(filename)

    return file_contents

print('\n'.join(read_files()))

#%%
def wait_until_done():
    def check_is_done():
        # Add a keyword so that wait_until_done()
        # doesn't run forever
        global done
        if random.random() < 0.1:
            done = True

    while not done:
        check_is_done()

done = False
wait_until_done()

print('Work done? {}'.format(done))

#%% 3.3 Closures
def foo():
    a = 5
    def bar():
        print(a)
    return bar

func = foo()
func()

type(func.__closure__)
len(func.__closure__)
func.__closure__[0].cell_contents

#%%
x = 25
def foo(value):
    def bar():
        print(value)
    return bar

my_func = foo(x)
my_func()

del(x)
my_func()

len(my_func.__closure__)
my_func.__closure__[0].cell_contents


x = 25
x = foo(x)
x()
len(x.__closure__)
x.__closure__[0].cell_contents

#%%
def return_a_func(arg1, arg2):
    def new_func():
        print('arg1 was {}'.format(arg1))
        print('arg2 was {}'.format(arg2))
    return new_func

my_func = return_a_func(2, 17)

# Show that my_func()'s closure is not None
print(my_func.__closure__ is not None)

#%%
def return_a_func(arg1, arg2):
    def new_func():
        print('arg1 was {}'.format(arg1))
        print('arg2 was {}'.format(arg2))
    return new_func

my_func = return_a_func(2, 17)

print(my_func.__closure__ is not None)

# Show that there are two variables in the closure
print(len(my_func.__closure__) == 2)

#%%
def return_a_func(arg1, arg2):
    def new_func():
        print('arg1 was {}'.format(arg1))
        print('arg2 was {}'.format(arg2))
    return new_func

my_func = return_a_func(2, 17)

print(my_func.__closure__ is not None)
print(len(my_func.__closure__) == 2)

# Get the values of the variables in the closure
closure_values = [
    my_func.__closure__[i].cell_contents for i in range(2)
]
print(closure_values == [2, 17])

#%%
def my_special_function():
    print('You are running my_special_function()')

def get_new_func(func):
    def call_func():
        func()
    return call_func

new_func = get_new_func(my_special_function)

# Redefine my_special_function() to just print "hello"
def my_special_function():
    print('hello')

new_func()

#%%
def my_special_function():
    print('You are running my_special_function()')

def get_new_func(func):
    def call_func():
        func()
    return call_func

new_func = get_new_func(my_special_function)

# Delete my_special_function()
del(my_special_function)

new_func()

#%%
def my_special_function():
    print('You are running my_special_function()')

def get_new_func(func):
    def call_func():
        func()
    return call_func

# Overwrite `my_special_function` with the new function
my_special_function = get_new_func(my_special_function)

my_special_function()

#%% 3.4 Decorators
# @double_args
def multiply(a, b):
    return a * b
multiply(1, 5)

#%%
def multiply(a, b):
    return a * b
def double_args(func):
    return func
new_multiply = double_args(multiply)
new_multiply(1, 5)

#%%
def multiply(a, b):
    return a * b
def double_args(func):
    def wrapper(a, b):
        return func(a, b)
    return wrapper
new_multiply = double_args(multiply)
new_multiply(1, 5)

#%%
def multiply(a, b):
    return a * b
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper
new_multiply = double_args(multiply)
new_multiply(1, 5)

multiply.__closure__[0].cell_contents

#%%
@double_args
def multiply(a, b):
    return a * b
multiply(1, 5)

#%%
def my_function(a, b, c):
    print(a + b + c)

# Decorate my_function() with the print_args() decorator
my_function = print_args(my_function)

my_function(1, 2, 3)

#%%
# Decorate my_function() with the print_args() decorator
@print_args
def my_function(a, b, c):
    print(a + b + c)

my_function(1, 2, 3)

#%%
def print_before_and_after(func):
    def wrapper(*args):
        print('Before {}'.format(func.__name__))
        # Call the function being decorated with *args
        func(*args)
        print('After {}'.format(func.__name__))
    # Return the nested function
    return wrapper

@print_before_and_after
def multiply(a, b):
    print(a * b)

multiply(5, 10)

#%% 4. More on Decorators

#%% 4.1 Real-world examples
import time

def timer(func):
    """A decorator that prints how long a function took to run.

    :arg
        func (callable): The function being decorated.

    :returns
        callable: The decorated function.
    """
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

@timer
def sleep_n_seconds(n):
    time.sleep(n)

sleep_n_seconds(5)
sleep_n_seconds(10)

def memoize(func):
    """Store the result of the decorated func for fast lookup
    """
    cache = {}
    def wrapper(*args, **kwargs):
        if (args, kwargs) not in cache:
            cache[(args, kwargs)] = func(*args, **kwargs)
        return cache[(args, kwargs)]
    return wrapper

@memoize
def slow_function(a, b):
    print('Sleeping...')
    time.sleep(5)
    return a + b

slow_function(3, 4)

#%%
def print_return_type(func):
    # Define wrapper(), the decorated function
    def wrapper(*args, **kwargs):
        # Call the function being decorated
        result = func(*args, **kwargs)
        print('{}() returned type {}'.format(
            func.__name__, type(result)
        ))
        return result
    # Return the decorated function
    return wrapper

@print_return_type
def foo(value):
    return value

print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))

#%%
def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

# Decorate foo() with the counter() decorator
@counter
def foo():
    print('calling foo()')

foo()
foo()

print('foo() was called {} times.'.format(foo.count))

#%% 4.2 Decorators and metadata
def sleep_n_seconds(n=10):
    """Pause processing for n sec"""
    time.sleep(n)

print(sleep_n_seconds.__doc__)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)

@timer
def sleep_n_seconds(n=10):
    """Pause processing for n sec"""
    time.sleep(n)

print(sleep_n_seconds.__doc__)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)

#%%
from functools import wraps

def timer(func):
    """A decorator that prints how long a function took to run.

    :arg
        func (callable): The function being decorated.

    :returns
        callable: The decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

@timer
def sleep_n_seconds(n=10):
    """Pause processing for n sec"""
    time.sleep(n)

print(sleep_n_seconds.__doc__)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)
sleep_n_seconds.__wrapped__

#%%
def add_hello(func):
    def wrapper(*args, **kwargs):
        print('Hello')
        return func(*args, **kwargs)
    return wrapper

# Decorate print_sum() with the add_hello() decorator
@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)

print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)

#%%
def add_hello(func):
    # Add a docstring to wrapper
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)
    return wrapper

@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)

print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)

#%%
# Import the function you need to fix the problem
from functools import wraps

def add_hello(func):
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)
    return wrapper

@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)

print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)

#%%
from functools import wraps

def add_hello(func):
    # Decorate wrapper() so that it keeps func()'s metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)
    return wrapper

@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)

print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)

#%%
def check_everything(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        check_inputs(*args, **kwargs)
        result = func(*args, **kwargs)
        check_outputs(result)
        return result
    return wrapper

@check_everything
def duplicate(my_list):
    """Return a new list that repeats the input twice"""
    return my_list + my_list

t_start = time.time()
duplicated_list = duplicate(list(range(50)))
t_end = time.time()
decorated_time = t_end - t_start

t_start = time.time()
# Call the original function instead of the decorated one
duplicated_list = duplicate.__wrapped__(list(range(50)))
t_end = time.time()
undecorated_time = t_end - t_start

print('Decorated time: {:.5f}s'.format(decorated_time))
print('Undecorated time: {:.5f}s'.format(undecorated_time))

#%% 4.3 Decorators that take arguments
def run_three_times(func):(func):
    def wrapper(*args, **kwargs):
        for i in range(3):
            func(*args, **kwargs)
    return wrapper

@run_three_times
def print_sum(a, b):
    print(a + b)

print_sum(3, 5)

#%%
def run_n_times(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@run_n_times(5)
def print_sum(a, b):
    print(a + b)

print_sum(3, 5)

#%%
# Make print_sum() run 10 times with the run_n_times() decorator
@run_n_times(10)
def print_sum(a, b):
    print(a + b)

print_sum(15, 20)

#%%
# Use run_n_times() to create the run_five_times() decorator
run_five_times = run_n_times(5)

@run_five_times
def print_sum(a, b):
    print(a + b)

print_sum(4, 100)

#%%
# Modify the print() function to always run 20 times
print = run_n_times(20)(print)

print('What is happening?!?!')

#%%
def html(open_tag, close_tag):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = func(*args, **kwargs)
            return '{}{}{}'.format(open_tag, msg, close_tag)
        # Return the decorated function
        return wrapper
    # Return wrapper decorator
    return decorator

#%%
# Make hello() return bolded text
@html('<b>', '</b>')
def hello(name):
    return 'Hello {}!'.format(name)

print(hello('Alice'))

#%%
# Make goodbye() return italicized text
@html('<i>', '</i>')
def goodbye(name):
    return 'Goodbye {}.'.format(name)

print(goodbye('Alice'))

#%%
# Wrap the result of hello_goodbye() in <div> and </div>
@html('<div>', '</div>')
def hello_goodbye(name):
    return '\n{}\n{}\n'.format(hello(name), goodbye(name))

print(hello_goodbye('Alice'))

#%% 4.4 Timeout(): a real world example
import signal

def raise_timeout(*args, **kwargs):
    raise TimeoutError()
signal.signal(signalnum=signal.SIGALRM, hadler=raise_timeout)
signal.alarm(5)

def timeout_in_5s(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signal.alarm(5)
        try:
            return func(*args, **kwargs)
        finally:
            signal.alarm(0)
    return wrapper

@timeout_in_5s
def foo():
    time.sleep(10)
    print('foo!')

foo()

#%%
def timeout(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwarg):
            signal.alarm(n_seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

@timeout(5)
def foo():
    time.sleep(10)
    print('foo!')

@timeout(20)
def bar():
    time.sleep(10)
    print('foo!')

foo()
bar()

#%%
def tag(*tags):
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
        wrapper.tags = tags
        return wrapper
    # Return the new decorator
    return decorator

@tag('test', 'this is a tag')
def foo():
    pass

print(foo.tags)

#%%
def returns_dict(func):
    # Complete the returns_dict() decorator
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert type(result) == dict
        return result
    return wrapper

@returns_dict
def foo(value):
    return value

try:
    print(foo([1,2,3]))
except AssertionError:
    print('foo() did not return a dict!')

#%%
def returns(return_type):
    # Complete the returns() decorator
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            assert type(result) == return_type
            return result
        return wrapper
    return decorator

@returns(dict)
def foo(value):
    return value

try:
    print(foo([1,2,3]))
except AssertionError:
    print('foo() did not return a dict!')