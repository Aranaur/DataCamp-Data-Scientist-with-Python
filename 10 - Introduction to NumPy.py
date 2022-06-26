import numpy as np
from matplotlib import pyplot as plt

# %% 1. Introducing arrays

python_list = [3, 2, 5, 8, 4, 9, 7, 6, 1]
array = np.array(python_list)

type(array)

python_list_of_lists = [[3, 2, 5],
                        [8, 4, 9],
                        [7, 6, 1]]
np.array(python_list_of_lists)

python_list = ['beep', False, 65, .945, [3, 2, 5]]

numpy_boolean_array = [[True, False], [True, True], [False, True]]
numpy_float_array = [1.9, 5.4, 8.8, 3.6, 3.2]

np.zeros(5)
np.random.random(5)
np.arange(5)

np.zeros((5, 3))
np.random.random((5, 3))
np.arange(-5, 3)
np.arange(-5, 3, 1.5)

plt.scatter(np.arange(0, 7), np.arange(-3, 4))
plt.show()

# %%

# Convert sudoku_list into an array
sudoku_array = np.array(sudoku_list)
# Print the type of sudoku_array
print(type(sudoku_array))

# %%

# Create an array of zeros which has four columns and two rows
zero_array = np.zeros((2, 4))
print(zero_array)

# %%

# Create an array of random floats which has six columns and three rows
random_array = np.random.random((3, 6))
print(random_array)

# %%

# Create an array of integers from one to ten
one_to_ten = np.arange(1, 11)

# Create your scatterplot
plt.scatter(one_to_ten, doubling_array)
plt.show()

# %% 1.1 Array dimensionality

array_1_2D = np.array([[1, 2], [5, 7]])
array_2_2D = np.array([[8, 9], [5, 7]])
array_3_2D = np.array([[1, 2], [5, 7]])
array_3D = np.array([array_1_2D, array_2_2D, array_3_2D])

array_3D.shape
array_3D.flatten()
array_3D.reshape((6, 1, 2))

# %%

# Create the game_and_solution 3D array
game_and_solution = np.array([sudoku_game, sudoku_solution])

# Print game_and_solution
print(game_and_solution)

# %%

# Create a second 3D array of another game and its solution
new_game_and_solution = np.array([new_sudoku_game, new_sudoku_solution])

# Create a 4D array of both game and solution 3D arrays
games_and_solutions = np.array([game_and_solution, new_game_and_solution])

# Print the shape of your 4D array
print(games_and_solutions.shape)

# %%

# Flatten sudoku_game
flattened_game = sudoku_game.flatten()

# Print the shape of flattened_game
print(flattened_game.shape)

# %%

# Flatten sudoku_game
flattened_game = sudoku_game.flatten()

# Print the shape of flattened_game
print(flattened_game.shape)

# Reshape flattened_game back to a nine by nine array
reshaped_game = flattened_game.reshape((9, 9))

# Print sudoku_game and reshaped_game
print(sudoku_game)
print(reshaped_game)

# %% 1.2 NumPy data types

bin(10436)

np_int32_range = [0-(2**32)/2, 0+(2**32)/2]

np.array([1.32, 5.78, 175.55]).dtype

int_array = np.array([[1, 2, 3], [4, 5, 6]])
int_array.dtype

np.array(['Introduction', 'to', 'NumPy']).dtype

float32_array = np.array([1.32, 5.78, 175.55], dtype=np.float32)
float32_array.dtype

np.array([True, "Boop", 42, 42.42]).dtype

np.array([0, 42, 42.42]).dtype

np.array([True, False, 42]).dtype

# %%

# Create an array of zeros with three rows and two columns
zero_array = np.zeros((3, 2))

# Print the data type of zero_array
print(zero_array.dtype)

# %%

# Create an array of zeros with three rows and two columns
zero_array = np.zeros((3, 2))

# Print the data type of zero_array
print(zero_array.dtype)

# Create a new array of int32 zeros with three rows and two columns
zero_int_array = np.zeros((3, 2), dtype=np.int32)

# Print the data type of zero_int_array
print(zero_int_array.dtype)

# %%

# Print the data type of sudoku_game
print(sudoku_game.dtype)

# Change the data type of sudoku_game to int8
small_sudoku_game = sudoku_game.astype('int8')

# Print the data type of small_sudoku_game
print(small_sudoku_game.dtype)

# %% 2. Indexing and slicing arrays
