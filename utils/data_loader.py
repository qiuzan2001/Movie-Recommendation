import pandas as pd
import numpy as np
import os


def load_data(data_path='data'):
    movies = pd.read_csv(
        os.path.join(data_path, 'movies.dat'),
        sep='::',
        header=None,
        names=['MovieID', 'Title', 'Genres'],
        engine='python',
        encoding='latin-1'
    )

    ratings = pd.read_csv(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python',
        encoding='latin-1'
    )

    users = pd.read_csv(
        os.path.join(data_path, 'users.dat'),
        sep='::',
        header=None,
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        engine='python',
        encoding='latin-1'
    )

    max_user_id = ratings['UserID'].max()
    max_movie_id = ratings['MovieID'].max()

    NUM_USERS = max_user_id
    NUM_MOVIES = max_movie_id

    R = np.full((NUM_USERS, NUM_MOVIES), np.nan)

    for row in ratings.itertuples():
        u = row.UserID - 1
        m = row.MovieID - 1
        R[u, m] = row.Rating

    R_df = pd.DataFrame(
        R,
        index=['u' + str(i) for i in range(1, NUM_USERS + 1)],
        columns=['m' + str(j) for j in range(1, NUM_MOVIES + 1)]
    )

    return movies, users, R_df
