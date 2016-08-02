import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

hotels = pd.read_table('hotels.txt', index_col=0)
users = pd.read_table('users.txt', index_col=0)
activity = pd.read_table('activity.txt')

users.loc[users['gender'] == 'female', 'gender'] = 0
users.loc[users['gender'] == 'male', 'gender'] = 1
views = pd.DataFrame(0, index=users.index, columns=hotels.index)

# build vectorized hotel views for each user
for user in users.index:
    user_hotels = activity.loc[activity['user'] == user]['hotel']
    for hotel in user_hotels:
        views.loc[user, hotel] = 1
        # views.loc[user, hotel] = hotels['star_rating'][hotel]

print(views.shape)
print(pd.value_counts(views.values.ravel()))

# add gender and home features to views table
views['gender'] = users['gender']
views['home'] = users['home continent']
cols = list(views)
cols.insert(0, cols.pop(cols.index('home')))
cols.insert(0, cols.pop(cols.index('gender')))
views = views.ix[:, cols]
views.to_csv('views.txt', sep='\t')

views = pd.read_table('views.txt', index_col=0)

# k-means clustering model
kmeans_model = KMeans(n_clusters=10, random_state=1).fit(views.iloc[:, 2:])
labels = kmeans_model.labels_

print(pd.crosstab(labels, views['gender']))
print(pd.crosstab(labels, views['home']))

# PCA to compress views for 2D visualization
pca = PCA(3)
plot_columns = pca.fit_transform(views.iloc[:, 2:])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], z=plot_columns[:, 2], c=labels)
ax.scatter(plot_columns[:, 0], plot_columns[:, 1], plot_columns[:, 2], c=labels)
plt.legend(labels)
plt.show()

users['cluster'] = labels+1
users.to_csv('users_clusters_v2.txt', sep='\t')

# mysql shenanigans
ranks = pd.read_table('ranks_v2.txt', index_col=0)

# extract training and test sets
test = []
train = pd.DataFrame()

for user in users.index:
    test.append(activity.loc[activity['user'] == user]['hotel'].iloc[-1])
    train = train.append(activity.loc[activity['user'] == user].iloc[:-1])

test_set = pd.DataFrame(test, index=np.arange(1, len(test)+1))

# go back and re-cluster for your validation. too lazy right now.
predictions = pd.DataFrame(index=np.arange(1, len(users)+1), columns=['hotel'])

for user in users.index:
    print 'user', user
    cluster = 'c' + str(users.loc[users.index == user]['cluster'].iloc[0])
    print 'cluster', cluster
    # user_hotels = activity.loc[activity['user'] == user]['hotel']
    user_hotels = train.loc[activity['user'] == user]['hotel']
    ranked_hotels = ranks[cluster]
    for hotel in ranked_hotels:
        if hotel not in user_hotels:
            predictions['hotel'][user] = hotel
            break

# first method (0 1 1 0) accuracy = 22%