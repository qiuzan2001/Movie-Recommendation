# utils/ibcf.py
import numpy as np
import pandas as pd
from utils.popularity import define_popularity


def myIBCF(newuser, S, R_df, popularity_ranking, top_n=10):
    w = newuser.copy()
    NUM_MOVIES = R_df.shape[1]
    rated_mask = ~np.isnan(w)

    predictions = np.full(NUM_MOVIES, np.nan)

    for i in range(NUM_MOVIES):
        if rated_mask[i]:
            continue
        sim_movies = np.where(~np.isnan(S[i, :]))[0]
        rated_and_similar = sim_movies[rated_mask[sim_movies]]
        if len(rated_and_similar) == 0:
            continue
        sim_vals = S[i, rated_and_similar]
        user_ratings = w[rated_and_similar]
        denom = np.nansum(sim_vals)
        if denom > 0:
            pred = np.nansum(sim_vals * user_ratings) / denom
            predictions[i] = pred

    non_na_preds = np.where(~np.isnan(predictions))[0]
    if len(non_na_preds) < top_n:
        sorted_preds_idx = non_na_preds[np.argsort(predictions[non_na_preds])[::-1]]
        top_already = set(sorted_preds_idx)
        needed = top_n - len(sorted_preds_idx)
        all_movies_pop_order = popularity_ranking.index
        rated_movies = set(np.where(rated_mask)[0])
        candidate_pop = [int(x[1:]) - 1 for x in all_movies_pop_order
                         if (int(x[1:]) - 1 not in rated_movies and int(x[1:]) - 1 not in top_already)]
        fill_movies = candidate_pop[:needed]
        final_indices = list(sorted_preds_idx) + fill_movies
    else:
        sorted_preds_idx = non_na_preds[np.argsort(predictions[non_na_preds])[::-1]]
        final_indices = sorted_preds_idx[:top_n]

    recommended = ["m" + str(i + 1) for i in final_indices]
    return recommended
