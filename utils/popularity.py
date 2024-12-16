# utils/popularity.py
import numpy as np

def define_popularity(R_df, method='num_ratings'):
    count_ratings = R_df.notna().sum(axis=0)
    avg_ratings = R_df.mean(axis=0, skipna=True)
    popularity_score = count_ratings * avg_ratings
    popularity_ranking = popularity_score.sort_values(ascending=False)
    return popularity_ranking

def get_top_popular_movies(R_df, movies, top_n=10):
    popularity_ranking = define_popularity(R_df)
    top_movies = popularity_ranking.head(top_n).index
    top_info = movies.set_index('MovieID').loc[[int(x[1:]) for x in top_movies]]
    top_info = top_info[['Title']]
    top_info['MovieID_str'] = top_movies
    return top_info
