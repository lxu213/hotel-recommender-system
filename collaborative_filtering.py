# collaborative filtering for predicting user's next hotel viewing

import pandas as pd
from scipy.spatial.distance import cosine

hotels = pd.read_table('hotels.txt', index_col=0)
users = pd.read_table('users.txt', index_col=0)
activity = pd.read_table('activity.txt')

users.loc[users['gender'] == 'female', 'gender'] = 0
users.loc[users['gender'] == 'male', 'gender'] = 1
views = pd.DataFrame(0, index=users.index, columns=hotels.index)

# build binary vectorized hotel views for each user
for user in users.index:
    user_hotels = activity.loc[activity['user'] == user]['hotel']
    for hotel in user_hotels:
        views.loc[user, hotel] = 1
views.to_csv('views.txt', sep='\t')

# build similarity table using cosine distance calculation
data_cos = pd.DataFrame(index=users.index, columns=users.index)
for i in range(1, len(data_cos.index)+1):
    if i % 10 == 1:
        print i
    for j in range(1, len(data_cos.index)+1):
        data_cos.ix[i,j] = 1-cosine(views.ix[i,:], views.ix[j,:])
data_cos.to_csv('data_cos.txt', sep='\t')

# build user's nearest neighbors table
user_neighbor = pd.DataFrame(index=data_cos.index, columns=range(1, 21))
for i in range(1, len(data_cos.index)+1):
    user_neighbor.ix[i,:20] = data_cos.ix[0:,i].order(ascending=False)[:20].index
user_neighbor.to_csv('data_neighbor.txt', sep='\t')

# scripts above are memory intensive - ran once and read in
data = pd.read_table('views.txt', index_col=0)
data_cos = pd.read_table('data_cos.txt', index_col=0)
user_neighbor = pd.read_table('data_neighbor.txt', index_col=0)

# build matrices of scores and predict next hotel view for each user
scores = pd.DataFrame(index=data.columns, columns=['sum'])
prediction = pd.DataFrame(index=data.index, columns=['hotel'])

for user in range(1, len(data.index)+1):
    if user % 10 == 1:
        print user
    for hotel in range(0, len(data.columns)):
        running_score = 0
        for nb in user_neighbor.ix[user]:
            running_score += (data_cos.ix[user, nb-1] * data.ix[nb, hotel])
        scores.ix[hotel, 'sum'] = running_score
    rec_list = scores.sort(columns=['sum'], ascending=False).index
    for rec in rec_list:
        viewed_list = activity.loc[activity['user'] == user]['hotel'].values
        if int(rec) not in viewed_list:
            prediction.ix[user]['hotel'] = rec
            break
prediction.to_csv('prediction.txt', sep='\t')
