"""Recency-weighting experiment: does recent form beat season averages?

Compares, under walk-forward validation (train on past games, test on
future games -- no leakage):
  baseline  : original pitch-level features only
  +recency  : baseline + whiff_recent / n_recent / whiff_prior / n_prior
  +shrunk   : baseline + whiff_recent_shrunk only (Bayesian shrinkage
              toward the prior; the regularized single-feature version)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data import build_whiff_dataset, fetch_statcast
from model import evaluate_walkforward
from recency import RECENCY_FEATURES, add_recency_features

START, END = "2026-09-01", "2026-09-21"
SPLIT = "2026-09-15"  # train Sep 1-14, test Sep 15-21

df = fetch_statcast(START, END, cache_name="statcast_sep2026.parquet")
d = build_whiff_dataset(df)
print(f"Swings: {len(d):,} | whiff rate: {d['whiff'].mean():.3f}")

d = add_recency_features(d, window=5)
cov = d["whiff_recent"].notna().mean()
print(f"Rows with recent history: {cov:.1%} "
      f"(rest are first-game-in-window, median-imputed)")

feature_sets = {
    "baseline": [],
    "+recency": RECENCY_FEATURES,
    "+shrunk": ["whiff_recent_shrunk", "n_recent"],
}
results = evaluate_walkforward(d, feature_sets, split_date=SPLIT)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "recency_results.json")
with open(out, "w") as f:
    json.dump({"split": SPLIT, "window": 5, "results": results}, f, indent=2)
print(f"\nResults saved to {out}")
