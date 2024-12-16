# app/app.py
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.append('..')
from utils.data_loader import load_data
from utils.ibcf import myIBCF
from utils.popularity import define_popularity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommendation System (System II)")

# Load data and precomputed similarity matrix
movies, users, R_df = load_data()
popularity_ranking = define_popularity(R_df)
S = np.load(os.path.join('..', 'data', 'similarity_matrix.npy'))

# Sample movies to show for rating
sample_movies = movies.sample(10, random_state=42)
ratings_input = {}

st.write(
    "Please rate the following movies (1–5 stars). If you don't want to rate a movie, leave it as NA. After rating, click 'Get Recommendations' at the bottom.")

# Display sample movies in a grid
cols = st.columns(5)  # Adjust number of columns as desired
sample_movies_list = list(sample_movies.itertuples())
for i, row in enumerate(sample_movies_list):
    col = cols[i % 5]
    # Construct image URL
    movie_id_num = row.MovieID
    image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id_num}.jpg"
    with col:
        st.image(image_url, width=150, caption=row.Title)
        rating = st.selectbox(
            f"Rate '{row.Title}' (m{movie_id_num})",
            options=["NA", "1", "2", "3", "4", "5"],
            index=0,
            key=f"rate_{row.MovieID}"
        )
        if rating != "NA":
            ratings_input[f"m{movie_id_num}"] = float(rating)

st.markdown("---")

if st.button("Get Recommendations"):
    # Create the new user rating vector
    newuser = np.full(R_df.shape[1], np.nan)
    for mid_str, rating_val in ratings_input.items():
        mid = int(mid_str[1:]) - 1
        newuser[mid] = rating_val

    recommended = myIBCF(newuser, S, R_df, popularity_ranking)

    st.write("### Top 10 Recommended Movies for You:")
    rec_ids = [int(x[1:]) for x in recommended]
    rec_info = movies.set_index('MovieID').loc[rec_ids]

    rec_cols = st.columns(5)
    for i, (rid, info) in enumerate(rec_info.iterrows()):
        image_url = f"https://liangfgithub.github.io/MovieImages/{rid}.jpg"
        col = rec_cols[i % 5]
        with col:
            st.image(image_url, width=150, caption=info['Title'])
