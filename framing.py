"""Catcher pitch framing: how many extra strikes does each catcher 'steal'?

Method: train P(called strike | location, count, pitch) on all *taken* pitches,
then compare each catcher's actual called strikes vs expected. The gap is the
framing effect. Converted to runs at ~0.14 runs per extra strike.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data import DATA_DIR, build_whiff_dataset  # noqa: F401  (reuse DATA_DIR)

TAKEN = {"called_strike", "ball", "blocked_ball"}
NUM = ["plate_x", "plate_z", "dist_from_center", "balls", "strikes",
       "release_speed", "pfx_x", "pfx_z"]
CAT = ["pitch_type"]
RUNS_PER_STRIKE = 0.14


def load_taken() -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(DATA_DIR, "statcast_sep2026.parquet"))
    t = d[d["description"].isin(TAKEN)].copy()
    t["called_strike"] = (t["description"] == "called_strike").astype(int)
    t["dist_from_center"] = ((t["plate_x"]) ** 2 + (t["plate_z"] - 2.5) ** 2) ** 0.5
    return t.reset_index(drop=True)


def main():
    t = load_taken()
    print(f"Taken pitches: {len(t):,} | called-strike rate: {t['called_strike'].mean():.3f}")

    X = t[NUM + CAT]
    y = t["called_strike"].values
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, t.index, test_size=0.25, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(max_iter=300,
                                                            learning_rate=0.06,
                                                            random_state=42))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    print(f"Called-strike model ROC-AUC: {roc_auc_score(yte, p):.3f}")

    te = t.loc[ite, ["fielder_2", "called_strike"]].copy()
    te["expected"] = p
    g = te.groupby("fielder_2").agg(n=("called_strike", "size"),
                                    actual=("called_strike", "sum"),
                                    expected=("expected", "sum"))
    g = g[g["n"] >= 150].copy()
    g["strikes_added"] = g["actual"] - g["expected"]
    g["framing_runs"] = g["strikes_added"] * RUNS_PER_STRIKE
    g = g.sort_values("framing_runs", ascending=False)

    # map catcher ids -> names
    from pybaseball import playerid_lookup
    ids = [int(i) for i in g.index]
    names = {}
    for i in ids:
        try:
            lk = playerid_lookup(i, fuzzy=False)
            if len(lk):
                r = lk.iloc[0]
                names[i] = f"{r['name_first']} {r['name_last']}"
        except Exception:
            pass
    g["catcher"] = [names.get(int(i), str(int(i))) for i in g.index]

    print("\nTop 10 framers (Sept 2026, min 150 taken pitches):")
    print(g[["catcher", "n", "actual", "expected", "strikes_added",
             "framing_runs"]].head(10).round(1).to_string(index=False))
    print("\nBottom 5:")
    print(g[["catcher", "n", "actual", "expected", "strikes_added",
             "framing_runs"]].tail(5).round(1).to_string(index=False))
    g.to_csv(os.path.join(DATA_DIR, "catcher_framing.csv"))


if __name__ == "__main__":
    main()
