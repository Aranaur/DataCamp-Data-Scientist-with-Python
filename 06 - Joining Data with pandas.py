import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 4)

# %%  1. Inner join
census = pd.read_pickle('data/06/census.p')
wards = pd.read_pickle('data/06/ward.p')

census.head()
census.shape

wards.head()
wards.shape

wards_census = wards.merge(census, on="ward")
wards_census.head()
wards_census.shape

wards_census.columns

wards_census = wards.merge(census, on="ward", suffixes=('_ward', '_cen'))
wards_census.head()
wards_census.shape

# %%  1.1 Inner join - taxi
taxi_owners = pd.read_pickle('data/06/taxi_owners.p')
taxi_veh = pd.read_pickle('data/06/taxi_vehicles.p')

taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid')
taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid', suffixes=('_own', '_veh'))

taxi_own_veh['fuel_type'].value_counts()

# %%  2. One-to-many relationships

licenses = pd.read_pickle('data/06/licenses.p')
ward_licenses = wards.merge(licenses, on='ward', suffixes=('_ward', '_lic'))
ward_licenses.head()
ward_licenses.shape

# %%  2.1 One-to-many relationships

biz_owners = pd.read_pickle('data/06/business_owners.p')

# Merge the licenses and biz_owners table on account
licenses_owners = licenses.merge(biz_owners, on='account')

# Group the results by title then count the number of accounts
counted_df = licenses_owners.groupby('title').agg({'account': 'count'})

# Sort the counted_df in desending order
sorted_df = counted_df.sort_values('account', ascending=False)

# Use .head() method to print the first few rows of sorted_df
print(sorted_df.head())

# %% 3. Merging multiple DataFrames

# grants_licenses_ward = grants.merge(licenses, on=['address', 'zip']) \
#    .merge(wards, on='ward', suffixes=('_bis', '_ward'))
# grants_licenses_ward.head()

# grants_licenses_ward.groupby('ward').agg('sum').plot(kind='bar', y='grant')
# plt.show()

# Merge the ridership, cal, and stations tables
# ridership_cal_stations = ridership.merge(cal, on=['year','month','day']) \
# 							.merge(stations, on='station_id')

# Create a filter to filter ridership_cal_stations
# filter_criteria = ((ridership_cal_stations['month'] == 7)
#                   & (ridership_cal_stations['day_type'] == 'Weekday')
#                   & (ridership_cal_stations['station_name'] == 'Wilson'))

# Use .loc and the filter to select for rides
# print(ridership_cal_stations.loc[filter_criteria, 'rides'].sum())

zip_demo = pd.read_pickle('data/06/zip_demo.p')

# Merge licenses and zip_demo, on zip; and merge the wards on ward
licenses_zip_ward = licenses.merge(zip_demo, on='zip') \
    .merge(wards, on='ward')

# Print the results by alderman and show median income
print(licenses_zip_ward.groupby('alderman').agg({'income': 'median'}))

# %% 3.1 One-to-many merge with multiple tables
land_use = pd.read_pickle('data/06/land_use.p')

land_cen_lic = land_use.merge(census, on='ward') \
    .merge(licenses, on='ward', suffixes=('_cen', '_lic'))

pop_vac_lic = land_cen_lic.groupby(['ward', 'pop_2010', 'vacant'],
                                   as_index=False).agg({'account': 'count'})

sorted_pop_vac_lic = pop_vac_lic.sort_values(['vacant', 'account', 'pop_2010'],
                                             ascending=[False, True, False])

#%% 4. Left join
movies = pd.read_pickle('data/06/movies.p')
taglines = pd.read_pickle('data/06/taglines.p')
financials = pd.read_pickle('data/06/financials.p')

movies.head()
movies.shape

taglines.head()
taglines.shape

financials.head()

movies_taglines = movies.merge(taglines, on='id', how='left')
movies_taglines.head()

movies_financials = movies.merge(financials, on='id', how='left')

number_of_missing_fin = movies_financials['budget'].isnull().sum()

number_of_missing_fin

#%%

# Merge the toy_story and taglines tables with a left join
# toystory_tag = toy_story.merge(taglines, on='id', how='left')

# Print the rows and shape of toystory_tag
# print(toystory_tag)
# print(toystory_tag.shape)

# Merge the toy_story and taglines tables with a inner join
# toystory_tag = toy_story.merge(taglines, on='id', how='inner')

# Print the rows and shape of toystory_tag
# print(toystory_tag)
# print(toystory_tag.shape)

#%% 5. Other joins

movie_to_genres = pd.read_pickle('data/06/movie_to_genres.p')
tv_genre = movie_to_genres[movie_to_genres['genre'] == 'TV Movie']
tv_genre

m = movie_to_genres['genre'] == 'TV Movie'
tv_genre = movie_to_genres[m]
tv_genre

movies

#%% 5.1 Right join
tv_movies = movies.merge(tv_genre, how='right', left_on='id', right_on='movie_id')
tv_movies

#%% 5.2 Outer join
family = movie_to_genres[movie_to_genres['genre'] == 'Family']
comedy = movie_to_genres[movie_to_genres['genre'] == 'Comedy']

family
comedy

family_comedy = family.merge(comedy, on='movie_id', how='outer',
                             suffixes=('_fam', '_com'))

family_comedy

#%%
# Merge action_movies to scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, on='movie_id', how='right')

# Merge action_movies to scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, on='movie_id', how='right',
                                   suffixes=('_act', '_sci'))

# Merge action_movies to the scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, on='movie_id', how='right',
                                   suffixes=('_act','_sci'))

# From action_scifi, select only the rows where the genre_act column is null
scifi_only = action_scifi[action_scifi['genre_act'].isnull()]

# Merge the movies and scifi_only tables with an inner join
movies_and_scifi_only = movies.merge(scifi_only, left_on='id', right_on='movie_id', how='inner')

# Print the first few rows and shape of movies_and_scifi_only
print(movies_and_scifi_only.head())
print(movies_and_scifi_only.shape)

#%%
# Use right join to merge the movie_to_genres and pop_movies tables
genres_movies = movie_to_genres.merge(pop_movies, how='right',
                                      left_on='movie_id',
                                      right_on='id')

# Count the number of genres
genre_count = genres_movies.groupby('genre').agg({'id':'count'})

# Plot a bar chart of the genre_count
genre_count.plot(kind='bar')
plt.show()

#%%
# Merge iron_1_actors to iron_2_actors on id with outer join using suffixes
iron_1_and_2 = iron_1_actors.merge(iron_2_actors,
                                     how='outer',
                                     on='id',
                                     suffixes=('_1','_2'))

# Create an index that returns true if name_1 or name_2 are null
m = ((iron_1_and_2['name_1'].isnull()) |
     (iron_1_and_2['name_2'].isnull()))

# Print the first few rows of iron_1_and_2
print(iron_1_and_2[m].head())

#%% 6. Merging a table to itself

sequals = pd.read_pickle('data/06/sequels.p')
sequals

original_sequals = sequals.merge(sequals, left_on='sequel', right_on='id', suffixes=('_org', '_seq'))
original_sequals

original_sequals[['title_org', 'title_seq']].head()

original_sequals = sequals.merge(sequals, left_on='sequel', right_on='id', how='left', suffixes=('_org', '_seq'))
original_sequals

#%%

# Merge the crews table to itself
crews_self_merged = crews.merge(crews, how='inner', on='id', suffixes=('_dir', '_crew'))

# Create a Boolean index to select the appropriate
boolean_filter = ((crews_self_merged['job_dir'] == 'Director') &
     (crews_self_merged['job_crew'] != 'Director'))
direct_crews = crews_self_merged[boolean_filter]

print(direct_crews.head())

#%% 7. Merging on indexes

movies = movies.set_index('id')
taglines = taglines.set_index('id')

movies_taglines = movies.merge(taglines, on='id', how='left')
movies_taglines

movies_genres = movies.merge(movie_to_genres, left_on='id', left_index=True, right_on='movie_id', right_index=True)

#%%
# Merge to the movies table the ratings table on the index
movies_ratings = movies.merge(ratings, on='id')

# Print the first few rows of movies_ratings
print(movies_ratings.head())

# Merge sequels and financials on index id
sequels_fin = sequels.merge(financials, on='id', how='left')

# Self merge with suffixes as inner join with left on sequel and right on id
orig_seq = sequels_fin.merge(sequels_fin, how='inner', left_on='sequel',
                             right_on='id', right_index=True,
                             suffixes=('_org','_seq'))

# Add calculation to subtract revenue_org from revenue_seq
orig_seq['diff'] = orig_seq['revenue_seq'] - orig_seq['revenue_org']

# Select the title_org, title_seq, and diff
print(titles_diff.sort_values('diff', ascending=False).head())

#%% 8. Filtering joins

# Merge employees and top_cust
empl_cust = employees.merge(top_cust, on='srid',
                            how='left', indicator=True)

# Select the srid column where _merge is left_only
srid_list = empl_cust.loc[empl_cust['_merge'] == 'left_only', 'srid']

# Get employees not working with top customers
print(employees[employees['srid'].isin(srid_list)])

# Merge the non_mus_tck and top_invoices tables on tid
tracks_invoices = non_mus_tcks.merge(top_invoices, on='tid')

# Use .isin() to subset non_mus_tcsk to rows with tid in tracks_invoices
top_tracks = non_mus_tcks[non_mus_tcks['tid'].isin(tracks_invoices['tid'])]

# Group the top_tracks by gid and count the tid rows
cnt_by_gid = top_tracks.groupby(['gid'], as_index=False).agg({'tid':'count'})

# Merge the genres table to cnt_by_gid on gid and print
print(cnt_by_gid.merge(genres, on='gid'))

#%% 9. Concatenate DataFrames together vertically
# .concat([], ignore_index=True, keys=[], sort=True, join='inner')
# .append(ignore_index=True, sort=True)

# Concatenate the tracks
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               sort=True)
print(tracks_from_albums)

# Concatenate the tracks so the index goes from 0 to n-1
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               ignore_index=True,
                               sort=True)
print(tracks_from_albums)

# Concatenate the tracks, show only columns names that are in all tables
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               join='inner',
                               sort=True)
print(tracks_from_albums)

# Concatenate the tables and add keys
inv_jul_thr_sep = pd.concat([inv_jul, inv_aug, inv_sep],
                            keys=['7Jul', '8Aug', '9Sep'])

# Group the invoices by the index keys and find avg of the total column
avg_inv_by_month = inv_jul_thr_sep.groupby(level=0).agg({'total': 'mean'})

# Bar plot of avg_inv_by_month
avg_inv_by_month.plot(kind='bar')
plt.show()

# Use the .append() method to combine the tracks tables
metallica_tracks = tracks_ride.append([tracks_master, tracks_st], sort=False)

# Merge metallica_tracks and invoice_items
tracks_invoices = metallica_tracks.merge(invoice_items)

# For each tid and name sum the quantity sold
tracks_sold = tracks_invoices.groupby(['tid', 'name']).agg({'quantity': 'count'})

# Sort in decending order by quantity and print the results
print(tracks_sold.sort_values('quantity', ascending=False))

#%% 10. Verifying integrity
#%%
# .merge(validate=None)
# 'one_to_one'
# 'one_to_many'
# 'many_to_one'
# 'many_to_many'

# tracks.merge(specs, on='tid', validate='one_to_one')
# albums.merge(tracks, on='aid', validate='one_to_many')

#%%
# pd.concat([inv_feb, inv_mar], verify_intengrity=True)

#%%
# Concatenate the classic tables vertically
classic_18_19 = pd.concat([classic_18, classic_19], ignore_index=True)

# Concatenate the pop tables vertically
pop_18_19 = pd.concat([pop_18, pop_19], ignore_index=True)

# Merge classic_18_19 with pop_18_19
classic_pop = classic_18_19.merge(pop_18_19, on='tid', how='inner')

# Using .isin(), filter classic_18_19 rows where tid is in classic_pop
popular_classic = classic_18_19[classic_18_19['tid'].isin(classic_pop['tid'])]

# Print popular chart
print(popular_classic)

# %% Using merge_ordered()
# pd.merge_ordered(df1, df2, on='', left_on=, right_on=, how=, suffixes)


