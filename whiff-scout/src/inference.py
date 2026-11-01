"""Inference tool: run the saved whiff GBM on a pitcher's swings.

The model artifact lives in the parent repo (../models/) and is
referenced here by relative path, never copied.
"""
import os

import joblib
import pandas as pd

from src.features import (FEATURES_CATEGORICAL, FEATURES_NUMERIC,
                          build_whiff_dataset)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(os.path.dirname(HERE), "models",
                          "whiff_gbm_strict75.joblib")

PITCH_NAMES = {
    "FF": "Four-Seam Fastball",
    "FC": "Cutter",
    "SI": "Sinker",
    "CH": "Changeup",
    "CU": "Curveball",
    "SL": "Slider",
    "ST": "Sweeper",
    "FS": "Splitter",
    "FO": "Forkball",
    "KC": "Knuckle Curve",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "SC": "Screwball",
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def run_inference(model, pitches: pd.DataFrame) -> dict:
    """Return per-pitch-type actual vs predicted whiff + overall totals."""
    swings = build_whiff_dataset(pitches)
    X = swings[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    swings = swings.assign(pred_whiff=model.predict_proba(X)[:, 1])

    rows = []
    for code, g in swings.groupby("pitch_type"):
        rows.append({
            "code": code,
            "name": PITCH_NAMES.get(code, code),
            "n": int(len(g)),
            "actual": float(g["whiff"].mean()),
            "predicted": float(g["pred_whiff"].mean()),
        })
    rows.sort(key=lambda r: r["n"], reverse=True)

    # Matchup context: whiff vs left- / right-handed batters.
    stand = {}
    for s, g in swings.groupby("stand"):
        stand[s] = {
            "n": int(len(g)),
            "actual": float(g["whiff"].mean()),
            "predicted": float(g["pred_whiff"].mean()),
        }

    # Pitch mix over ALL pitches (not just swings).
    mix = (pitches["pitch_type"].value_counts(normalize=True) * 100).to_dict()

    return {
        "n_pitches": int(len(pitches)),
        "n_swings": int(len(swings)),
        "overall_actual": float(swings["whiff"].mean()),
        "overall_predicted": float(swings["pred_whiff"].mean()),
        "by_pitch": rows,
        "by_stand": stand,
        "pitch_mix_pct": {k: float(v) for k, v in mix.items()},
    }
