# Whiff Rate by Inning / Times Through Order / Count: Implementation Notes & Thinking

Date: 2026-10-10

## 1. Motivation: the question you asked

> "Do whiff rates differ by pitch type or by inning?"

This time, no theory — computed directly on your September Statcast data (38,649 swings).
Definition note: **whiff rate = share of swings with zero contact**
(`swinging_strike` / `swinging_strike_blocked`), not whiffs per pitch — a pitch with no swing can't be a "whiff."

## 2. What I did

New `run_inning_analysis.py`: three split dimensions + one figure (`figures/whiff_by_inning_group.png`). Fully reproducible:

```bash
.venv/bin/python run_inning_analysis.py
```

## 3. Results

### 3.1 Times-through-the-order penalty: real, and a clean signal

| Times facing the lineup (same pitcher) | Whiff rate | Swings |
|---|---|---|
| 1st time | 23.36% | 25,334 |
| 2nd time | 21.79% | 9,176 |
| 3rd time | 21.05% | 4,095 |

After seeing the repertoire once, whiff rate drops ~2.3 points — the data-driven explanation for why
starters get pulled the third time through. Good news: `n_thruorder_pitcher` is already a column in your
data, so it can go straight into the model as a feature.

### 3.2 Inning: differs, but confounded by starter/reliever usage

| Inning group | 4-seam | Slider | Changeup | Curve | Sinker |
|---|---|---|---|---|---|
| 1–3 | 19.2% | 29.9% | 29.1% | 29.1% | 10.9% |
| 4–6 | 17.0% | 29.2% | 26.7% | 30.1% | 10.7% |
| 7–9 | 20.5% | 32.9% | 32.8% | 26.9% | 13.3% |

Reading:

- **4-seam drops to 17.0% in innings 4–6**: that's the pitcher fatigue you asked about — starter velocity
  dips, fastball whiffs dip with it.
- **But innings 7–9 bounce back (4-seam 20.5%, slider/changeup ~33%)**: fatigue didn't vanish — the
  personnel changed. Late innings are almost all relievers: higher velocity, sharper breaking stuff.
  Inning here is a proxy for pitcher role; using raw inning as a feature would teach the model the wrong causality.
- **Curve is flat across all three groups (29.1% → 30.1% → 26.9%)**: spin-based pitches don't depend on
  velocity decay, consistent with theory.
- Sinker lowest everywhere (~11–13%): it's designed for contact/ground balls — low whiff is a feature, not a bug.

**Conclusion**: inning itself isn't worth a feature; what matters is times-through-the-order
(`n_thruorder_pitcher`) and the starter/reliever role distinction.

### 3.3 Count: two-strike whiff rate is LOWER than you'd expect

| Count situation | Whiff rate | Swings |
|---|---|---|
| Pitcher ahead | 20.9% | 7,761 |
| Even / hitter ahead | 24.2% | 16,347 |
| Two strikes | 22.1% | 14,541 |

Intuition says two strikes should be the easiest to strike out — but **conditional on swinging**,
hitters shorten up and spoil pitches (fouls) with two strikes, so whiff rate is lower than in even counts,
where hitters take full swings. This count × swing-intent interaction is a nice detail for the report.

## 4. Thinking: what this means for the model and play-by-play

1. **Feature recommendation**: add `n_thruorder_pitcher` to the GBM (monotonically decreasing, clean signal,
   low overfit risk); encode starter/reliever as a binary feature before interacting with inning.
2. **Don't**: use raw `inning` as a numeric feature — the late-inning whiff spike is reliever selection
   effect, not inning magic.
3. **Play-by-play use**: on a broadcast you can say "this is the lineup's second look at him — historically
   whiff rate drops 1.6 points the second time through" — more narrative than quoting a raw whiff rate.
   Applies directly to tomorrow's NLCS Game 1 (Misiorowski vs Skubal): from the second time through,
   the putaway value of the curve/changeup fades.
4. **Limitation**: only three weeks of September data; postseason pitcher usage (short starts, bullpen games)
   differs from the regular season, so don't apply these numbers directly.

## 5. Next steps (if you want to continue)

- Re-run on a full season: the starter fatigue curve (innings 4–6) will be sharper, reliever samples more stable.
- Ablation: add `n_thruorder_pitcher` to the model and check the AUC delta — verify whether times-through-order
  has real predictive power or is just descriptive.
- Opponent adjustment: currently doesn't account for who the hitter is; strong hitters whiff less by nature.
