"""End-to-end run: fetch Statcast, build whiff dataset, train, plot, report."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data import build_whiff_dataset, fetch_statcast
from model import evaluate, permutation_importance_top
from plots import feature_importance_bar, whiff_by_pitch_type, whiff_heatmap

START, END = "2026-09-01", "2026-09-21"

df = fetch_statcast(START, END, cache_name="statcast_sep2026.parquet")
d = build_whiff_dataset(df)
print(f"Swings: {len(d):,} | whiff rate: {d['whiff'].mean():.3f}")

results, (Xte, yte) = evaluate(d)
imp = permutation_importance_top(results["gradboost"]["pipeline"], Xte, yte)
print("\nTop features:")
print(imp.round(4).to_string())

whiff_by_pitch_type(d)
whiff_heatmap(d)
feature_importance_bar(imp)
print("\nFigures saved to figures/")
