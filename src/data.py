"""Fetch and cache MLB Statcast pitch-tracking data via pybaseball."""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Pitch descriptions that indicate the batter swung
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}
# ... of which these are whiffs (swings with no contact)
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
}


def fetch_statcast(start_dt: str, end_dt: str, cache_name: str = "statcast.parquet") -> pd.DataFrame:
    """Download Statcast data for a date range, caching to parquet."""
    from pybaseball import statcast

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, cache_name)
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    print(f"Downloading Statcast {start_dt} -> {end_dt} ...")
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    df.to_parquet(cache_path)
    print(f"Saved {len(df):,} pitches to {cache_path}")
    return df


def build_whiff_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pitches where the batter swung; label whiffs as 1."""
    d = df[df["description"].isin(SWING_DESCRIPTIONS)].copy()
    d["whiff"] = d["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    # Edge of the strike zone distance (approx, in feet from zone center)
    d["dist_from_center"] = ((d["plate_x"]) ** 2 + (d["plate_z"] - 2.5) ** 2) ** 0.5
    return d.reset_index(drop=True)


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
]
FEATURES_CATEGORICAL = ["pitch_type", "stand"]
