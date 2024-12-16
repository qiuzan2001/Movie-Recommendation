# utils/similarity.py
import numpy as np
import pandas as pd
from utils.constants import MIN_COMMON_RATINGS, TOP_K


def center_rows(R):
    R_centered = R.copy()
    means = np.nanmean(R_centered, axis=1)
    for i in range(R_centered.shape[0]):
        row_mask = ~np.isnan(R_centered[i, :])
        R_centered[i, row_mask] = R_centered[i, row_mask] - means[i]
    return R_centered, means


def compute_similarity(R_centered):
    num_users, num_movies = R_centered.shape
    norms = np.sqrt(np.nansum(R_centered ** 2, axis=0))
    S = np.full((num_movies, num_movies), np.nan)

    for i in range(num_movies):
        Ri = R_centered[:, i]
        valid_i = ~np.isnan(Ri)
        for j in range(num_movies):
            if j == i:
                continue
            Rj = R_centered[:, j]
            valid_j = ~np.isnan(Rj)
            both = valid_i & valid_j
            if np.sum(both) >= MIN_COMMON_RATINGS:
                num = np.nansum(Ri[both] * Rj[both])
                den = norms[i] * norms[j]
                if den > 0:
                    cos_ij = num / den
                    sij = (1 + cos_ij) / 2
                    S[i, j] = sij

    for i in range(num_movies):
        row = S[i, :]
        non_na_idx = np.where(~np.isnan(row))[0]
        if len(non_na_idx) > TOP_K:
            top_idx = non_na_idx[np.argsort(row[non_na_idx])[-TOP_K:]]
            mask = np.ones(len(row), dtype=bool)
            mask[top_idx] = False
            S[i, mask] = np.nan
    return S
