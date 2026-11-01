# Whiff Scout

Input a pitcher name → agent fetches Statcast data → runs ken's
whiff-prediction GBM (strict labels, AUC 0.777, 75 trees) → outputs a
self-contained HTML scouting report: pitch mix, actual vs predicted
whiff by pitch type with sample sizes, matchup advice, and uncertainty
stated honestly.

## Run

From this `whiff-scout/` subfolder, using the repo's venv:

```
../.venv/bin/python -m src.report "Drew Rasmussen"
../.venv/bin/python -m src.report "Jacob Misiorowski"
```

Writes `reports/<Pitcher_Name>.html`. The demo path is always the
local cache in the parent repo's `data/` dir
(`statcast_<id>_2026.parquet`); pybaseball fetch is only a fallback
for pitchers without a cache file.

## Sample window honesty

The 2026 caches span 2026-03-01 → 10-03 and mix regular season,
spring training, and postseason games. Every report states its window
and game-type mix; it never claims a pure regular-season sample.

## Meta Model API (live prose)

Report prose is written live by Muse Spark when a key is present,
otherwise template prose is used and the report flags it. To enable
live prose, create a `.env` file in this subfolder (it is gitignored
at the repo root — never commit it, never paste the key anywhere):

```
META_API_KEY=your_key_here
```

The client reads only that file (or the environment). No key is ever
hardcoded, logged, or printed.

## Layout

- `src/features.py` — feature lists + strict whiff label, copied
  unchanged from ken's MLB project (`../src/data.py`) so the saved
  model sees identical inputs.
- `src/data_tool.py` — pitcher name → Statcast ID → cached pitches.
- `src/inference.py` — loads `../models/whiff_gbm_strict75.joblib`
  (referenced, not copied) and scores swings.
- `src/meta_client.py` — OpenAI-compatible client for Meta Model API
  (`https://api.meta.ai/v1`, model `muse-spark-1.3`).
- `src/report.py` — three visible roles (Scout coordinates, Data
  fetches, Analyst runs the model + writes prose) + HTML rendering.

## Web demo (optional, local only)

```
../.venv/bin/python -m src.server
```

Open http://127.0.0.1:8000, type a pitcher name, get the HTML report. Binds loopback only.
