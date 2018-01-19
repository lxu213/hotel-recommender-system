# validation
# create training set (activity minus the last hotel view) and results set (last hotel view)
import numpy as np
import pandas as pd

users = pd.read_table('users.txt', index_col=0)
activity = pd.read_table('activity.txt')

# create a new activity
test = []
train = pd.DataFrame()

for user in users.index:
    test.append(activity.loc[activity['user'] == user]['hotel'].iloc[-1])
    train = train.append(activity.loc[activity['user'] == user].iloc[:-1])
test_set = pd.DataFrame(test, index=np.arange(1, len(test)+1))

# same similarity matrix as original data due to time constraints
data_cos = pd.read_table('data_cos.txt', index_col=0)
