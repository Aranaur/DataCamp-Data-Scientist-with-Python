#%% 1. Scala's real-world project repository data

import pandas as pd
import matplotlib.pyplot as plt

pulls_one = pd.read_csv('data/07/pulls_2011-2013.csv')
pulls_two = pd.read_csv('data/07/pulls_2014-2018.csv')
pull_files = pd.read_csv('data/07/pull_files.csv')

#%% 2. Preparing and cleaning the data

# Append pulls_one to pulls_two
pulls = pd.concat([pulls_one, pulls_two], ignore_index=True)

# Convert the date for the pulls object
pulls['date'] = pd.to_datetime(pulls['date'], utc=True)

#%% 3. Merging the DataFrames

# Merge the two DataFrames
data = pulls.merge(pull_files, on='pid')

#%% 4. Is the project still actively maintained?

# Create a column that will store the month
data['month'] = data['date'].dt.month

# Create a column that will store the year
data['year'] = data['date'].dt.year

# Group by the month and year and count the pull requests
counts = data.groupby(['month', 'year'])['pid'].count()

# Plot the results
counts.plot(kind='bar', figsize=(12, 4))
plt.show()


#%% 5. Is there camaraderie in the project?

# Group by the submitter
by_user = data.groupby('user')['pid'].count()

# Plot the histogram
by_user.plot(kind='hist', figsize=(12, 4))
plt.show()

#%% 6. What files were changed in the last ten pull requests?

# Identify the last 10 pull requests
last_10 = pulls.nlargest(10, 'date')

# Join the two data sets
joined_pr = last_10.merge(pull_files, on='pid')

# Identify the unique files
files = set(joined_pr['file'])

# Print the results
files

#%% 7. Who made the most pull requests to a given file?

# This is the file we are interested in:
file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'

# Identify the commits that changed the file
file_pr = data[data['file'] == file]

# Count the number of changes made by each developer
author_counts = file_pr.groupby('user')['pid'].count()

# Print the top 3 developers
author_counts.nlargest(3)

#%% 8. Who made the last ten pull requests on a given file?

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'

# Select the pull requests that changed the target file
file_pr = pull_files[pull_files['file'] == file]

# Merge the obtained results with the pulls DataFrame
joined_pr = file_pr.merge(pulls, on='pid')

# Find the users of the last 10 most recent pull requests
users_last_10 = set(joined_pr.nlargest(10, 'date')['user'])

# Printing the results
users_last_10