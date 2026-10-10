"""Recency-weighted pitcher features for whiff prediction.

For every swing, attach the pitcher's historical whiff rates computed
strictly from games BEFORE the current game_date (no leakage):

  whiff_recent   -- whiff rate over the pitcher's most recent <=5 games
  n_recent       -- swing count in that window (lets the model learn
                    how much to trust whiff_recent)
  whiff_prior    -- whiff rate over ALL earlier games (season-to-date;
                    doubles as the career baseline in this dataset)
  n_prior        -- swing count behind whiff_prior
  whiff_recent_shrunk -- Bayesian shrinkage of whiff_recent toward
                    whiff_prior: (n*r + k*prior) / (n + k).
                    Small samples get pulled toward the prior, which is
                    itself a form of regularization against overfitting
                    on noisy recent windows.

Rationale: a pitcher's last 5 games capture real form changes (new pitch,
mechanics tweak, fatigue) that season/career averages wash out, but a
5-game window is noisy -- hence the shrinkage and the explicit n_recent.
"""
import numpy as np
import pandas as pd

RECENCY_FEATURES = [
    "whiff_recent",
    "n_recent",
    "whiff_prior",
    "n_prior",
    "whiff_recent_shrunk",
]

SHRINKAGE_K = 200  # pseudo-counts of prior weight; larger = more shrinkage


def _pitcher_history(g: pd.DataFrame, pitcher, window: int = 5) -> pd.DataFrame:
    """Per-game history aggregates for one pitcher, leakage-free.

    For game i (0-indexed, sorted by date), all aggregates cover games
    [max(0, i-window), i) for 'recent' and [0, i) for 'prior'.
    """
    g = g.sort_values("game_date")
    n = g["n_swings"].to_numpy(dtype=float)
    w = g["n_whiffs"].to_numpy(dtype=float)
    m = len(g)
    cs_n = np.concatenate([[0.0], np.cumsum(n)])  # cs_n[i] = sum of games < i
    cs_w = np.concatenate([[0.0], np.cumsum(w)])

    idx = np.arange(m)
    lo = np.maximum(0, idx - window)
    recent_n = cs_n[idx] - cs_n[lo]
    recent_w = cs_w[idx] - cs_w[lo]
    prior_n = cs_n[idx]
    prior_w = cs_w[idx]

    out = pd.DataFrame({
        "pitcher": pitcher,
        "game_date": g["game_date"].to_numpy(),
    })
    out["n_recent"] = recent_n
    out["n_prior"] = prior_n
    out["whiff_recent"] = np.where(recent_n > 0, recent_w / recent_n, np.nan)
    out["whiff_prior"] = np.where(prior_n > 0, prior_w / prior_n, np.nan)
    return out


def add_recency_features(d: pd.DataFrame, window: int = 5,
                         shrinkage_k: float = SHRINKAGE_K) -> pd.DataFrame:
    """Attach recency features to a swing-level dataframe.

    Requires columns: pitcher, game_date, whiff. Returns a copy with
    RECENCY_FEATURES added. Rows from a pitcher's first game in the
    dataset get NaN (no history) -- handled downstream by imputation.
    """
    d = d.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])

    per_game = (
        d.groupby(["pitcher", "game_date"], observed=True)["whiff"]
        .agg(n_swings="size", n_whiffs="sum")
        .reset_index()
    )
    frames = [
        _pitcher_history(g, pitcher, window=window)
        for pitcher, g in per_game.groupby("pitcher", observed=True)
    ]
    hist = pd.concat(frames, ignore_index=True)

    d = d.merge(hist, on=["pitcher", "game_date"], how="left", validate="m:1")
    pr = d["whiff_prior"]
    nr = d["n_recent"]
    rr = d["whiff_recent"]
    d["whiff_recent_shrunk"] = np.where(
        nr > 0, (nr * rr + shrinkage_k * pr) / (nr + shrinkage_k), np.nan
    )
    return d
