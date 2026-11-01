"""Data tool: resolve a pitcher name to a Statcast ID and load pitches.

Demo path is ALWAYS the local cache in the parent repo's data/ dir
(../data/statcast_<id>_2026.parquet). pybaseball fetch is only a
fallback for pitchers without a cache file, never the demo path.
The report must state the sample window honestly: the 2026 caches
span 2026-03-01 -> 10-03 and mix regular season, spring training,
and postseason games -- never claim pure regular season.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Parent repo's data dir (caches live here; nothing is copied).
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

# Demo pitchers with verified local caches (2026 season).
PITCHER_REGISTRY = {
    "drew rasmussen": 656876,
    "jacob misiorowski": 694819,
    # Eval set (ken's rule: Cy Young top 5 + rookie SP top 5).
    # Misiorowski (694819) and Messick (800048) have local caches;
    # the rest are fetched via pybaseball on first run and cached.
    "parker messick": 800048,
    "chris sale": 519242,
    "cristopher sánchez": 650911,
    "cam schlittler": 693645,
    "dylan cease": 656302,
    "logan henderson": 701656,
    "nolan mclean": 690997,
    "payton tolle": 801139,
    "walbert ureña": 700712,
}

# Same window the WS G1 preview used.
FETCH_START = "2026-03-01"
FETCH_END = "2026-10-20"


def resolve_pitcher(name: str) -> int:
    """Resolve a pitcher name to a Statcast player ID.

    Tries the local registry first (exact full name, then unique
    last-name match), then pybaseball playerid_lookup as a fallback.
    """
    key = " ".join(name.strip().lower().split())
    if key in PITCHER_REGISTRY:
        return PITCHER_REGISTRY[key]
    last = key.split()[-1]
    matches = [pid for full, pid in PITCHER_REGISTRY.items()
               if full.split()[-1] == last]
    if len(matches) == 1:
        return matches[0]
    # Fallback: look the player up (network). Never used for demo pitchers.
    from pybaseball import playerid_lookup
    parts = key.split()
    first, last = parts[0], parts[-1]
    lookup = playerid_lookup(last, first)
    if len(lookup) == 0:
        raise ValueError(f"Could not resolve pitcher name: {name!r}")
    return int(lookup.iloc[0]["key_mlbam"])


def cache_path(pid: int) -> str:
    return os.path.join(DATA_DIR, f"statcast_{pid}_2026.parquet")


def sample_window(df: pd.DataFrame) -> dict:
    """Describe the sample honestly: date span + game_type mix."""
    dates = pd.to_datetime(df["game_date"])
    types = df["game_type"].value_counts().to_dict() if "game_type" in df else {}
    return {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "game_types": {str(k): int(v) for k, v in types.items()},
        "n_pitches": int(len(df)),
    }


def load_pitches(name: str) -> tuple:
    """Return (pitcher_id, pitches DataFrame, window dict).

    Uses the local cache when present; fetches via pybaseball only
    when no cache exists, and saves the fetch as a new cache file.
    """
    pid = resolve_pitcher(name)
    path = cache_path(pid)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        source = "cache"
    else:
        from pybaseball import statcast_pitcher
        df = statcast_pitcher(FETCH_START, FETCH_END, pid)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_parquet(path)
        source = "pybaseball fetch (saved to cache)"
    window = sample_window(df)
    window["source"] = source
    return pid, df, window
