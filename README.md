# MLB Whiff Prediction

Predicting whether a batter's swing will whiff, using MLB Statcast pitch-tracking data.
End-to-end data science project: data ingestion → feature engineering → modeling → visualization.

## Problem definition

Given a pitch the batter **swung at**, predict whether it will be a whiff (swinging strike / foul tip = 1).

## Data

- Source: MLB Statcast (via `pybaseball`, free official data)
- Range: 2026-09-01 to 2026-09-21 — 80,851 pitches, 38,429 swings
- Base whiff rate: 25.1%

## Method

Features: release velocity, spin rate, movement (pfx_x/pfx_z), pitch location (plate_x/plate_z),
distance from zone center, count, pitch type, batter handedness, etc.

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression | 0.721 | 0.534 |
| Gradient Boosting | **0.747** | **0.555** |

Most important features: distance from zone center, pitch height, vertical movement, count, velocity.

## Key findings

- Splitter (35.6%) and slider (32.8%) generate the most whiffs; sinker (13.4%) the fewest (designed for ground balls)
- Pitches on the zone edge and out of the zone whiff most — "chasing" drives whiffs
- Whiff rate rises sharply with two strikes

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py
```

## Project layout

```
src/
  data.py      # Statcast fetch + caching + whiff dataset construction
  model.py     # feature pipeline + training + evaluation
  plots.py     # visualization
  recency.py   # recency-weighted pitcher features (see docs/recency-weighting.md)
run_experiment.py          # one-command end-to-end run
run_recency_experiment.py  # recency-weighting ablation (walk-forward validation)
run_inning_analysis.py     # whiff by inning / times-through-order / count
docs/          # write-ups: recency-weighting.md, inning-whiff.md
figures/       # output charts
data/          # cached parquet data
```

## Possible extensions

- Pitcher/batter-level random effects (mixed-effects model)
- GAM surfaces for nonlinear whiff curves by pitch type and location
- Predict exit velocity / xwOBA
- Pitcher clustering: find pitchers with similar arsenals
