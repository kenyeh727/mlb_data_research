# Recency Weighting: Implementation Notes & Thinking

Date: 2026-10-10

## 1. Motivation: the question you asked

> "What about weighting the last 5 games vs. the pitcher's season vs. career numbers?"

This is the classic forecasting trade-off: **recent form (sensitive, noisy) vs. large-sample averages (stable, slow to react)**.
A pitcher may have changed their grip or added a pitch last week — career averages won't capture that, but 5 games alone might just be noise.

## 2. What I did

### New `src/recency.py`: three layers of historical features

For every swing, compute the pitcher's historical whiff rates using **only games strictly before the current game** (`game_date` strictly less than current — zero leakage):

| Feature | Definition |
|---|---|
| `whiff_recent` | Whiff rate over the most recent ≤5 games (recent form) |
| `n_recent` | Swing count in that window (lets the model learn how much to trust it) |
| `whiff_prior` | Whiff rate over ALL earlier games (season-to-date; doubles as the career baseline here) |
| `n_prior` | Swing count behind the prior |
| `whiff_recent_shrunk` | Bayesian shrinkage: `(n·recent + k·prior) / (n + k)`, k=200 |

Shrinkage intuition: small samples get pulled toward the career mean; large samples trust recent form.
**Shrinkage is itself regularization** — it directly answers the "won't small samples overfit?" concern.

### New walk-forward validation (`model.py::evaluate_walkforward`)

The original `evaluate()` used a random `train_test_split`. You can't do that with time-dependent features —
random splits leak future games into training, and validation AUC looks unrealistically good.
Switched to: **train on 9/1–9/14, test on 9/15–9/21**. The model only ever sees the past.

### Experiment (`run_recency_experiment.py`)

Three feature sets compared with GradBoost under walk-forward validation:

| Feature set | ROC-AUC | AP |
|---|---|---|
| baseline (original features) | **0.7567** | 0.5816 |
| +recency (all 5 added) | 0.7550 | 0.5808 |
| +shrunk (shrinkage only) | 0.7534 | 0.5793 |

Permutation importance: all 5 recency features rank dead last (≈0) — the model ignores them.
What actually matters: `dist_from_center` (0.159), `plate_z`, `release_speed` — pitch location and velocity.

## 3. My thinking: why didn't it help? (an honest negative result)

**The data window is too short.** We only have three weeks (9/1–9/21). A starter makes ~3–4 starts in three weeks,
so "last 5 games" is nearly identical to "all history" — `whiff_recent` and `whiff_prior` compute almost the same
number, the features are highly collinear, and there's no marginal signal for the model to learn.
The method isn't wrong; the data isn't long enough.

Also, 27% of swings are the pitcher's first game in the window (no history), median-imputed, which further dilutes the signal.

In other words: **on three weeks of data, the recent-vs-career distinction doesn't exist**.
With a full season (~30 starts), the three layers would properly separate and the experiment would be meaningful.

## 4. Next steps (if you want to continue)

1. **Longer data**: pull the full 2026 season; split `whiff_prior` into true "season" and "career (prior years)" layers —
   the three-layer architecture you originally had in mind.
2. **Opponent adjustment**: a high recent whiff rate might just mean weak opponents; add the opposing lineup's
   average whiff rate as an offset.
3. **EWMA instead of a hard 5-game cutoff**: `w = λ^d` exponential decay, tune λ as a hyperparameter —
   smoother than a window where game #6 abruptly drops out.
4. **Test the "ace premium" hypothesis**: the model systematically underestimates aces by ~4 points.
   With real recency features (full-season data), check whether that gap shrinks. That would be a great
   portfolio story arc: *baseline → discovered systematic bias → hypothesis (model can't keep up with form)
   → added features → gap shrinks*.

## 5. Files changed

- New `src/recency.py` (feature engineering, zero leakage)
- `src/model.py`: new `build_pipeline_for()`, `evaluate_walkforward()`
- New `run_recency_experiment.py`, `recency_results.json`
- This file `docs/recency-weighting.md`
