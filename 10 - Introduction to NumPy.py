import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sudoku_game = np.load('data/10/sudoku_game.npy')
sudoku_solution = np.load('data/10/sudoku_solution.npy')
rgb_array = np.load('data/10/rgb_array.npy')
tree_census = np.load('data/10/tree_census.npy')
monthly_sales = np.load('data/10/monthly_sales.npy')

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
sudoku_array = np.array(sudoku_game)
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

np_int32_range = [0 - (2 ** 32) / 2, 0 + (2 ** 32) / 2]

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

array = np.array([2, 4, 6, 8, 10])
array[3]

sudoku_solution[2, 4]
sudoku_solution[0]
sudoku_solution[:, 3]
sudoku_solution[2:4]
sudoku_solution[2:4, 2:4]
sudoku_solution[3:6:2, 3:6:2]

np.sort(sudoku_game, )
np.sort(sudoku_game, axis=0)

# %%

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Print the first five block_ids
print(block_ids[:5])

# %%

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Select the tenth block ID from block_ids
tenth_block_id = block_ids[9]
print(tenth_block_id)

# %%

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Select five block IDs from block_ids starting with the tenth ID
block_id_slice = block_ids[9:14]
print(block_id_slice)

# %%

# Extract trunk diameters information and sort from smallest to largest
sorted_trunk_diameters = np.sort(tree_census[:, 2])
print(sorted_trunk_diameters)

# %%

# Create an array of the first 100 trunk diameters from tree_census
hundred_diameters = tree_census[:100, 2]
print(hundred_diameters)

# %%

# Create an array of trunk diameters with even row indices from 50 to 100 inclusive
every_other_diameter = tree_census[50:101:2, 2]
print(every_other_diameter)

# %% 2.1 Filtering arrays

# Masks and indexing or np.where()

# Masks
one_to_five = np.arange(1, 6)
mask = one_to_five % 2 == 0
mask
one_to_five[mask]

classroom_ids_and_size = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
classroom_ids_and_size
classroom_ids_and_size[:, 1] % 2 == 0
classroom_ids_and_size[:, 0][classroom_ids_and_size[:, 1] % 2 == 0]

# np.where()
np.where(classroom_ids_and_size[:, 1] % 2 == 0)

row_ind, column_ind = np.where(sudoku_game == 0)
row_ind
column_ind

np.where(sudoku_game == 0, '', sudoku_game)

# %%

# Create an array which contains row data on the largest tree in tree_census
largest_tree_data = tree_census[tree_census[:, 2] == 51]
print(largest_tree_data)

# %%

# Create an array which contains row data on the largest tree in tree_census
largest_tree_data = tree_census[tree_census[:, 2] == 51]
print(largest_tree_data)

# Slice largest_tree_data to get only the block id
largest_tree_block_id = largest_tree_data[:, 1]
print(largest_tree_block_id)

# %%

# Create an array which contains row data on the largest tree in tree_census
largest_tree_data = tree_census[tree_census[:, 2] == 51]
print(largest_tree_data)

# Slice largest_tree_data to get only the block ID
largest_tree_block_id = largest_tree_data[:, 1]
print(largest_tree_block_id)

# Create an array which contains row data on all trees with largest_tree_block_id
trees_on_largest_tree_block = tree_census[tree_census[:, 1] == largest_tree_block_id]
print(trees_on_largest_tree_block)

# %%

# Create the block_313879 array containing trees on block 313879
block_313879 = tree_census[tree_census[:, 1] == 313879]
print(block_313879)

# %%

# Create an array of row_indices for trees on block 313879
row_indices = np.where(tree_census[:, 1] == 313879)

# Create an array which only contains data for trees on block 313879
block_313879 = tree_census[row_indices]
print(block_313879)

# %%

# Create and print a 1D array of tree and stump diameters
trunk_stump_diameters = np.where(tree_census[:, 2] == 0, tree_census[:, 3], tree_census[:, 2])
print(trunk_stump_diameters)

# %% 2.2 Adding and removing data

classroom_ids_and_size
new_classrooms = np.array([[5, 30], [5, 17]])
np.concatenate((classroom_ids_and_size, new_classrooms))

grade_levels_and_teachers = np.array([[1, "James"], [1, 'George'], [3, 'Amy'], [3, 'Meehir']])
classroom_data = np.concatenate((classroom_ids_and_size, grade_levels_and_teachers), axis=1)
classroom_data

array_1D = np.array([1, 2, 3])
column_array_2D = array_1D.reshape((3, 1))
column_array_2D.shape
array_1D.shape

row_array_2D = array_1D.reshape((1, 3))
row_array_2D

classroom_data
np.delete(classroom_data, 1, axis=0)
np.delete(classroom_data, 1, axis=1)
np.delete(classroom_data, 1)

# %%

# Print the shapes of tree_census and new_trees
print(tree_census.shape, new_trees.shape)

# %%

# Print the shapes of tree_census and new_trees
print(tree_census.shape, new_trees.shape)

# Add rows to tree_census which contain data for the new trees
updated_tree_census = np.concatenate((tree_census, new_trees), axis=0)
print(updated_tree_census)

# %%

# Print the shapes of tree_census and trunk_stump_diameters
print(tree_census.shape, trunk_stump_diameters.shape)

# %%

# Print the shapes of tree_census and trunk_stump_diameters
print(trunk_stump_diameters.shape, tree_census.shape)

# Reshape trunk_stump_diameters
reshaped_diameters = trunk_stump_diameters.reshape((1000, 1))

# %%

# Print the shapes of tree_census and trunk_stump_diameters
print(trunk_stump_diameters.shape, tree_census.shape)

# Reshape trunk_stump_diameters
reshaped_diameters = trunk_stump_diameters.reshape((1000, 1))

# Concatenate reshaped_diameters to tree_census as the last column
concatenated_tree_census = np.concatenate((tree_census, reshaped_diameters), axis=1)
print(concatenated_tree_census)

# %%

# Delete the stump diameter column from tree_census
tree_census_no_stumps = np.delete(tree_census, 3, axis=1)

# Save the indices of the trees on block 313879
private_block_indices = np.where(tree_census[:, 1] == 313879)

# Delete the rows for trees on block 313879 from tree_census_no_stumps
tree_census_clean = np.delete(tree_census_no_stumps, private_block_indices, axis=0)

# Print the shape of tree_census_clean
print(tree_census_clean.shape)

# %% 3. Summarizing data

# .sum()
# .min()
# .max()
# .mean()
# .cumsum()

security_breaches = np.array([[0, 5, 1],
                              [0, 2, 0],
                              [1, 1, 2],
                              [2, 2, 1],
                              [0, 0, 0]])

security_breaches.sum()
np.sum(security_breaches)

security_breaches.sum(axis=0)
security_breaches.sum(axis=1)

security_breaches.min()
security_breaches.max()

security_breaches.max(axis=0)
security_breaches.max(axis=1)

security_breaches.mean()
security_breaches.mean(axis=0)
security_breaches.mean(axis=1)

security_breaches.sum(axis=0, keepdims=True)

security_breaches.cumsum(axis=0)

cum_sum_by_client = security_breaches.cumsum(axis=0)
plt.plot(np.arange(1, 6), cum_sum_by_client[:, 0], label='Client 1')
plt.plot(np.arange(1, 6), cum_sum_by_client[:, 1], label='Client 2')
plt.plot(np.arange(1, 6), cum_sum_by_client[:, 2], label='Client 3')
plt.plot(np.arange(1, 6), cum_sum_by_client.mean(axis=1), label='Average')
plt.legend()
plt.show()

# %%

monthly_sales = np.array([[4134, 23925, 8657],
                          [4116, 23875, 9142],
                          [4673, 27197, 10645],
                          [4580, 25637, 10456],
                          [5109, 27995, 11299],
                          [5011, 27419, 10625],
                          [5245, 27305, 10630],
                          [5270, 27760, 11550],
                          [4680, 24988, 9762],
                          [4913, 25802, 10456],
                          [5312, 25405, 13401],
                          [6630, 27797, 18403]])
# Create a 2D array of total monthly sales across industries
monthly_industry_sales = monthly_sales.sum(axis=1, keepdims=True)
print(monthly_industry_sales)

# Add this column as the last column in monthly_sales
monthly_sales_with_total = np.concatenate((monthly_sales, monthly_industry_sales), axis=1)
print(monthly_sales_with_total)

# %%

# Create the 1D array avg_monthly_sales
avg_monthly_sales = monthly_sales.mean(axis=1)
print(avg_monthly_sales)

# Plot avg_monthly_sales by month
plt.plot(np.arange(1, 13), avg_monthly_sales, label="Average sales across industries")

# Plot department store sales by month
plt.plot(np.arange(1, 13), monthly_sales[:, 2], label="Department store sales")
plt.legend()
plt.show()

# %%

# Find cumulative monthly sales for each industry
cumulative_monthly_industry_sales = monthly_sales.cumsum(axis=0)
print(cumulative_monthly_industry_sales)

# Plot each industry's cumulative sales by month as separate lines
plt.plot(np.arange(1, 13), cumulative_monthly_industry_sales[:, 0], label="Liquor Stores")
plt.plot(np.arange(1, 13), cumulative_monthly_industry_sales[:, 1], label="Restaurants")
plt.plot(np.arange(1, 13), cumulative_monthly_industry_sales[:, 2], label="Department stores")
plt.legend()
plt.show()

# %% 3.1 Vectorized operations

np.arange(1000000).sum()

slow_py = np.array([[1, 2, 3], [4, 5, 6]])

for row in range(slow_py.shape[0]):
    for column in range(slow_py.shape[1]):
        slow_py[row][column] += 3

fast_py = np.array([[1, 2, 3], [4, 5, 6]])
fast_py + 3

fast_py * 3

slow_py + fast_py

slow_py * fast_py

slow_py > 2

chr_array = np.array(['NumPy', 'is', 'awsome'])
len(chr_array) > 2 # not vectorized - its Python function

vectorize_len = np.vectorize(len)
vectorize_len(chr_array) > 2 # vectorized len()

# %%

# Create an array of tax collected by industry and month
tax_collected = monthly_sales * 0.05
print(tax_collected)

# Create an array of sales revenue plus tax collected by industry and month
total_tax_and_revenue = monthly_sales + tax_collected
print(total_tax_and_revenue)

# %%

monthly_industry_multipliers = np.array([[0.98, 1.02, 1],
                                         [1.00, 1.01, 0.97],
                                         [1.06, 1.03, 0.98],
                                         [1.08, 1.01, 0.98],
                                         [1.08, 0.98, 0.98],
                                         [1.1, 0.99, 0.99],
                                         [1.12, 1.01, 1],
                                         [1.1, 1.02, 1],
                                         [1.11, 1.01, 1.01],
                                         [1.08, 0.99, 0.97],
                                         [1.09, 1, 1.02],
                                         [1.13, 1.03, 1.02]])

# Create an array of monthly projected sales for all industries
projected_monthly_sales = monthly_sales * monthly_industry_multipliers
print(projected_monthly_sales)

# Graph current liquor store sales and projected liquor store sales by month
plt.plot(np.arange(1, 13), monthly_sales[:, 0], label="Current liquor store sales")
plt.plot(np.arange(1, 13), projected_monthly_sales[:, 0], label="Projected liquor store sales")
plt.legend()
plt.show()

# %%

names = np.array([["Izzy", "Monica", "Marvin"],
                  ["Weber", "Patel", "Hernandez"]])

# Vectorize the .upper() string method
vectorized_upper = np.vectorize(str.upper)

# Apply vectorized_upper to the names array
uppercase_names = vectorized_upper(names)
print(uppercase_names)

# %% 3.2 Broadcasting

broad_array_1 = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
broad_array_1.shape
broad_array_2 = np.array([[0, 1, 2, 3, 4]])
broad_array_2.shape

broad_array_1 + broad_array_2

broad_array_3 = np.array([[0, 1]])
broad_array_1 + broad_array_3                   # will not work: shapes (2,5) (1,2)
broad_array_1 + broad_array_3.reshape((2, 1))   # will work

# %%

monthly_growth_rate = [1.01, 1.03, 1.03, 1.02, 1.05, 1.03, 1.06, 1.04, 1.03, 1.04, 1.02, 1.01]

# Convert monthly_growth_rate into a NumPy array
monthly_growth_1D = np.array(monthly_growth_rate)

# Reshape monthly_growth_1D
monthly_growth_2D = monthly_growth_1D.reshape((len(monthly_growth_1D), 1))

# Multiply each column in monthly_sales by monthly_growth_2D
print(monthly_sales * monthly_growth_2D)

# %%

monthly_industry_multipliers = np.array([[0.98, 1.02, 1.],
                                         [1.00, 1.01, 0.97],
                                         [1.06, 1.03, 0.98],
                                         [1.08, 1.01, 0.98],
                                         [1.08, 0.98, 0.98],
                                         [1.1, 0.99, 0.99],
                                         [1.12, 1.01, 1.],
                                         [1.1, 1.02, 1.],
                                         [1.11, 1.01, 1.01],
                                         [1.08, 0.99, 0.97],
                                         [1.09, 1., 1.02],
                                         [1.13, 1.03, 1.02]])

# Find the mean sales projection multiplier for each industry
mean_multipliers = monthly_industry_multipliers.mean(axis=0)
print(mean_multipliers)

# Print the shapes of mean_multipliers and monthly_sales
print(mean_multipliers.shape, monthly_sales.shape)

# Multiply each value by the multiplier for that industry
projected_sales = monthly_sales * mean_multipliers
print(projected_sales)

# %% 4. Saving and loading arrays

# .csv
# .txt
# .pkl
# .npy

rgb = np.array([[[255, 0, 0], [255, 0, 0], [255, 0, 0]],
                [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
                [[0, 0, 255], [0, 0, 255], [0, 0, 255]]])

plt.imshow(rgb)
plt.show()

# %%

rgb = np.array([[[255, 0, 0], [255, 255, 0], [255, 255, 255]],
                [[255, 0, 255], [0, 255, 0], [0, 255, 255]],
                [[0, 0, 0], [0, 255, 255], [0, 0, 255]]])

plt.imshow(rgb)
plt.show()

# %%

# read .npy
with open('data/10/rgb_array.npy', 'rb') as f:
    monet_rgb_array = np.load(f)
plt.imshow(logo_rgb_array)
plt.show()

red_array = monet_rgb_array[:, :, 0]
blue_array = monet_rgb_array[:, :, 1]
green_array = monet_rgb_array[:, :, 2]

red_array[1], blue_array[1], green_array[1]

unique, counts = np.unique(monet_rgb_array, return_counts=True)
rgb_dict = dict(zip(unique, counts))
max(rgb_dict, key=rgb_dict.get)

dark_monet_array = np.where(monet_rgb_array == 148, 255, monet_rgb_array)
plt.imshow(dark_monet_array)
plt.show()

# write .npy
with open('data/10/dark_monet_array.npy', 'wb') as f:
    np.save(f, dark_monet_array)

# help
help(np.unique)
help(np.ndarray.flatten)

# %%

# Load the mystery_image.npy file
with open('data/10/rgb_array.npy', 'rb') as f:
    rgb_array = np.load(f)

plt.imshow(rgb_array)
plt.show()

# %%

# Display the documentation for .astype()
help(np.ndarray.astype)

# %%

# Reduce every value in rgb_array by 50 percent
darker_rgb_array = rgb_array * 0.5

# Convert darker_rgb_array into an array of integers
darker_rgb_int_array = darker_rgb_array.astype(np.int8)
plt.imshow(darker_rgb_int_array)
plt.show()

# Save darker_rgb_int_array to an .npy file called darker_monet.npy
with open('data/10/darker_monet.npy', 'wb') as f:
    np.save(f, darker_rgb_int_array)

# %% 4.1 Array acrobatics

flipped_monet = np.flip(monet_rgb_array)
plt.imshow(flipped_monet)
plt.show()

flipped_row_monet = np.flip(monet_rgb_array, axis=0)
plt.imshow(flipped_row_monet)
plt.show()

flipped_colors_monet = np.flip(monet_rgb_array, axis=2)
plt.imshow(flipped_colors_monet)
plt.show()

flipped_except_colors_monet = np.flip(monet_rgb_array, axis=(0, 1))
plt.imshow(flipped_except_colors_monet)
plt.show()

# %% Transposing

transposing_array = np.array([[1.1, 1.2, 1.3],
                              [2.1, 2.2, 2.3],
                              [3.1, 3.2, 3.3],
                              [4.1, 4.2, 4.3]])

np.flip(transposing_array)
np.transpose(transposing_array)

transposing_monet = np.transpose(monet_rgb_array, axes=(1, 0, 2))
plt.imshow(transposing_monet)
plt.show()

# %%

# Flip rgb_array so that it is the mirror image of the original
mirrored_monet = np.flip(rgb_array, axis=1)
plt.imshow(mirrored_monet)
plt.show()

# Flip rgb_array so that it is upside down
upside_down_monet = np.flip(rgb_array, axis=(0, 1))
plt.imshow(upside_down_monet)
plt.show()

# Transpose rgb_array
transposed_rgb = np.transpose(rgb_array, axes=(1, 0, 2))
plt.imshow(transposed_rgb)
plt.show()

# %% 4.2 Stacking and splitting

rgb

red_array = rgb[:, :, 0]
blue_array = rgb[:, :, 1]
green_array = rgb[:, :, 2]
red_array

red_array, blue_array, green_array = np.split(rgb, 3, axis=2)
red_array.shape

red_array_2D = red_array.reshape((3, 3))
red_array_2D
red_array_2D.shape

plt.imshow(red_array)
plt.show()

#%%

red_array, blue_array, green_array = np.split(rgb_array, 3, axis=2)

stacked_rbg = np.stack([red_array, green_array, blue_array], axis=2)
plt.imshow(red_array)
plt.show()

#%%

# Split monthly_sales into quarterly data
q1_sales, q2_sales, q3_sales, q4_sales = np.split(monthly_sales, 4, axis=0)
print(q1_sales)

# Stack the four quarterly sales arrays
quarterly_sales = np.stack([q1_sales, q2_sales, q3_sales, q4_sales], axis=0)
print(quarterly_sales)

#%%

# Split rgb_array into red, green, and blue arrays
red_array, green_array, blue_array = np.split(rgb_array, 3, axis=2)

#%%

# Split rgb_array into red, green, and blue arrays
red_array, green_array, blue_array = np.split(rgb_array, 3, axis=2)

# Create emphasized_blue_array
emphasized_blue_array = np.where(blue_array > blue_array.mean(), 255, blue_array)

# Print the shape of emphasized_blue_array
print(emphasized_blue_array.shape)

# Remove the trailing dimension from emphasized_blue_array
emphasized_blue_array_2D = emphasized_blue_array.reshape((675, 843))

#%%

# Print the shapes of blue_array and emphasized_blue_array_2D
print(blue_array.shape, emphasized_blue_array_2D.shape)

# Reshape red_array and green_array
red_array_2D = red_array.reshape((675, 843))
green_array_2D = green_array.reshape((675, 843))

# Stack red_array_2D, green_array_2D, and emphasized_blue_array_2D
emphasized_blue_monet = np.stack([red_array_2D, green_array_2D, emphasized_blue_array_2D], axis=2)
plt.imshow(emphasized_blue_monet)
plt.show()

