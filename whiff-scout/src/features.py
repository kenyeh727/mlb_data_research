"""Feature definitions and strict whiff labeling for Whiff Scout.

Copied unchanged from ken's own MLB project (../src/data.py in this
repo) so the saved model (../models/whiff_gbm_strict75.joblib) sees
exactly the inputs it was trained on. Do not edit the feature lists or
the label definition without retraining the model.
"""
import numpy as np
import pandas as pd

# Pitch descriptions that indicate the batter swung
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}
# ... of which these are whiffs (swings with no contact).
# NOTE (2026-10-13, external review): foul_tip is EXCLUDED from whiffs --
# Savant's handling of foul tips is disputed, and counting them as whiffs
# hurt discrimination (strict target: test AUC 0.780 vs 0.769 legacy).
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}

FEATURES_NUMERIC = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "dist_from_center",
    "balls",
    "strikes",
    "release_pos_x",
    "release_pos_z",
    "spin_axis",
    # review additions (2026-10-13): zone-relative location + sequencing
    "zone_height",
    "zone_edge",
    "inside_out",
    "pitch_num_in_pa",
    "velo_delta",
]
FEATURES_CATEGORICAL = ["pitch_type", "stand", "p_throws", "prev_pitch_type"]


def build_whiff_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pitches where the batter swung; label whiffs as 1.

    Also engineers the review-recommended features (2026-10-13):
    zone-relative location (zone_height, zone_edge, inside_out) and
    sequencing context (pitch_num_in_pa, velo_delta, prev_pitch_type).
    """
    d = df[df["description"].isin(SWING_DESCRIPTIONS)].copy()
    d["whiff"] = d["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    # Edge of the strike zone distance (approx, in feet from zone center)
    d["dist_from_center"] = ((d["plate_x"]) ** 2 + (d["plate_z"] - 2.5) ** 2) ** 0.5

    # sort first so all subsequent index-aligned ops are safe
    d = d.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"]
                      ).reset_index(drop=True)

    # --- zone-relative location (each pitch's own sz_top/sz_bot) ---
    d["zone_height"] = d["plate_z"] - (d["sz_bot"] + d["sz_top"]) / 2
    d["zone_edge"] = np.hypot(d["plate_x"].abs() - 0.83,
                              d["zone_height"].abs() - 0.96)
    d["stand"] = d["stand"].fillna("U").astype(str)
    # inside (+) / outside (-) from the batter's perspective
    d["inside_out"] = np.sign(d["plate_x"]) * np.where(d["stand"] == "R", -1.0, 1.0)

    # --- sequencing context within each plate appearance ---
    g = d.groupby(["game_pk", "batter", "at_bat_number"])
    d["pitch_num_in_pa"] = g.cumcount() + 1
    d["velo_delta"] = (d["release_speed"] - g["release_speed"].shift(1)).fillna(0.0)
    d["prev_pitch_type"] = g["pitch_type"].shift(1).fillna("NONE").astype(str)

    d["p_throws"] = d["p_throws"].fillna("U").astype(str)
    return d.reset_index(drop=True)
