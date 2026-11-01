# Whiff Scout eval summary — 10 pitchers

Eval set (ken rule: Cy Young top 5 + rookie SP top 5). Model: GBM whiff predictor, strict labels (foul tips excluded), 75 trees, validation AUC 0.777 (95% CI 0.766–0.789). All samples come from the report pipeline (cache-first; 8 pitchers fetched via pybaseball on first run and saved to ../data/statcast_<id>_2026.parquet). No numbers invented.

## Per-pitcher overall

| Pitcher | Statcast ID | n pitches | n swings | Actual whiff | Predicted whiff | Gap (pp) | Window | Game-type mix | Source |
|---|---|---|---|---|---|---|---|---|---|
| Jacob Misiorowski | 694819 | 3,046 | 1,538 | 33.6% | 30.1% | +3.6 | 2026-03-08 → 2026-10-11 | D 88, L 94, R 2649, S 215 | pre-existing cache |
| Chris Sale | 519242 | 2,997 | 1,505 | 28.1% | 25.8% | +2.3 | 2026-03-05 → 2026-10-06 | D 95, F 116, R 2575, S 211 | fetched 2026-10-26, now cached |
| Cristopher Sánchez | 650911 | 3,332 | 1,657 | 28.8% | 23.6% | +5.2 | 2026-03-20 → 2026-09-30 | F 124, R 3158, S 50 | fetched 2026-10-26, now cached |
| Cam Schlittler | 693645 | 3,384 | 1,747 | 26.9% | 23.7% | +3.2 | 2026-03-06 → 2026-10-05 | D 76, F 117, R 3044, S 147 | fetched 2026-10-26, now cached |
| Dylan Cease | 656302 | 3,033 | 1,396 | 30.4% | 27.6% | +2.8 | 2026-03-10 → 2026-09-18 | R 2896, S 137 | fetched 2026-10-26, now cached |
| Logan Henderson | 701656 | 1,706 | 901 | 23.2% | 22.0% | +1.2 | 2026-03-04 → 2026-10-12 | D 72, L 64, R 1505, S 65 | fetched 2026-10-26, now cached |
| Nolan McLean | 690997 | 2,915 | 1,285 | 21.6% | 19.4% | +2.3 | 2026-03-29 → 2026-09-23 | R 2915 | fetched 2026-10-26, now cached |
| Parker Messick | 800048 | 3,335 | 1,608 | 23.2% | 22.0% | +1.2 | 2026-03-09 → 2026-10-08 | D 152, R 2954, S 229 | pre-existing cache |
| Payton Tolle | 801139 | 2,471 | 1,354 | 25.0% | 22.2% | +2.8 | 2026-03-04 → 2026-09-29 | F 83, R 2286, S 102 | fetched 2026-10-26, now cached |
| Walbert Ureña | 700712 | 2,715 | 1,253 | 26.7% | 22.1% | +4.6 | 2026-03-05 → 2026-09-23 | R 2490, S 225 | fetched 2026-10-26, now cached |

Mean absolute overall gap across the 10 pitchers: **2.9pp**. The model underpredicts for all 10 pitchers (predicted < actual in every case) — a consistent conservative bias on 2026 data, not random noise around the truth.

## Per-pitch-type actual vs predicted (sorted by n, top pitches)

### Jacob Misiorowski (overall actual 33.6% vs predicted 30.1%, 1,538 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 953 | 35.0% | 31.2% | +3.8 |  |
| Slider (SL) | 204 | 31.4% | 27.5% | +3.9 |  |
| Cutter (FC) | 190 | 20.5% | 22.4% | -1.9 |  |
| Curveball (CU) | 172 | 43.0% | 35.8% | +7.3 |  |
| Changeup (CH) | 19 | 31.6% | 23.4% | +8.1 | small sample (n<50) |

### Chris Sale (overall actual 28.1% vs predicted 25.8%, 1,505 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 648 | 23.9% | 22.0% | +2.0 |  |
| Slider (SL) | 569 | 38.1% | 32.6% | +5.5 |  |
| Changeup (CH) | 165 | 26.1% | 27.4% | -1.3 |  |
| Sinker (SI) | 123 | 6.5% | 12.3% | -5.8 |  |

### Cristopher Sánchez (overall actual 28.8% vs predicted 23.6%, 1,657 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Sinker (SI) | 694 | 9.8% | 11.7% | -1.9 |  |
| Changeup (CH) | 663 | 43.6% | 32.3% | +11.3 |  |
| Slider (SL) | 300 | 40.0% | 31.8% | +8.2 |  |

### Cam Schlittler (overall actual 26.9% vs predicted 23.7%, 1,747 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 827 | 31.8% | 26.2% | +5.6 |  |
| Cutter (FC) | 459 | 21.8% | 18.6% | +3.2 |  |
| Sinker (SI) | 299 | 13.7% | 16.1% | -2.4 |  |
| Curveball (CU) | 150 | 42.0% | 39.8% | +2.2 |  |
| Slider (SL) | 12 | 25.0% | 36.9% | -11.9 | small sample (n<50) |

### Dylan Cease (overall actual 30.4% vs predicted 27.6%, 1,396 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 541 | 20.9% | 22.6% | -1.7 |  |
| Slider (SL) | 497 | 41.2% | 36.7% | +4.5 |  |
| Changeup (CH) | 131 | 45.8% | 22.8% | +23.0 |  |
| Sinker (SI) | 114 | 13.2% | 13.4% | -0.3 |  |
| Knuckle Curve (KC) | 77 | 28.6% | 30.9% | -2.4 |  |
| Sweeper (ST) | 36 | 25.0% | 33.7% | -8.7 | small sample (n<50) |

### Logan Henderson (overall actual 23.2% vs predicted 22.0%, 901 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 433 | 17.8% | 18.1% | -0.4 |  |
| Changeup (CH) | 316 | 34.2% | 29.0% | +5.2 |  |
| Cutter (FC) | 128 | 16.4% | 18.8% | -2.4 |  |
| Sweeper (ST) | 24 | 12.5% | 18.4% | -5.9 | small sample (n<50) |

### Nolan McLean (overall actual 21.6% vs predicted 19.4%, 1,285 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Sinker (SI) | 444 | 10.8% | 11.3% | -0.5 |  |
| Four-Seam Fastball (FF) | 252 | 25.8% | 19.5% | +6.3 |  |
| Curveball (CU) | 201 | 34.8% | 27.9% | +6.9 |  |
| Cutter (FC) | 157 | 22.3% | 20.4% | +1.9 |  |
| Changeup (CH) | 94 | 25.5% | 26.3% | -0.8 |  |
| Sweeper (ST) | 92 | 21.7% | 27.5% | -5.7 |  |
| Slider (SL) | 45 | 35.6% | 26.0% | +9.5 | small sample (n<50) |

### Parker Messick (overall actual 23.2% vs predicted 22.0%, 1,608 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 698 | 22.1% | 18.4% | +3.7 |  |
| Changeup (CH) | 419 | 35.3% | 33.2% | +2.1 |  |
| Sinker (SI) | 251 | 9.6% | 11.2% | -1.6 |  |
| Slider (SL) | 98 | 23.5% | 26.6% | -3.1 |  |
| Curveball (CU) | 74 | 12.2% | 26.2% | -14.0 |  |
| Cutter (FC) | 68 | 22.1% | 17.3% | +4.7 |  |

### Payton Tolle (overall actual 25.0% vs predicted 22.2%, 1,354 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Four-Seam Fastball (FF) | 613 | 26.8% | 23.8% | +2.9 |  |
| Cutter (FC) | 298 | 26.2% | 21.4% | +4.8 |  |
| Sinker (SI) | 291 | 13.7% | 14.1% | -0.4 |  |
| Curveball (CU) | 122 | 41.0% | 35.7% | +5.3 |  |
| Changeup (CH) | 30 | 20.0% | 20.7% | -0.7 | small sample (n<50) |

### Walbert Ureña (overall actual 26.7% vs predicted 22.1%, 1,253 swings)

| Pitch | n | Actual | Predicted | Gap (pp) | Flag |
|---|---|---|---|---|---|
| Changeup (CH) | 449 | 34.7% | 27.5% | +7.3 |  |
| Sinker (SI) | 419 | 18.1% | 16.4% | +1.8 |  |
| Four-Seam Fastball (FF) | 214 | 19.6% | 16.5% | +3.1 |  |
| Sweeper (ST) | 142 | 35.2% | 32.1% | +3.1 |  |

## Small-sample flags (n<50)

- Jacob Misiorowski: Changeup (CH, n=19)
- Cam Schlittler: Slider (SL, n=12)
- Dylan Cease: Sweeper (ST, n=36)
- Logan Henderson: Sweeper (ST, n=24)
- Nolan McLean: Slider (SL, n=45)
- Payton Tolle: Changeup (CH, n=30)

## Notes

- **Largest overall misses**: Cristopher Sánchez (actual 28.8% vs predicted 23.6%, gap +5.2pp) and Walbert Ureña (26.7% vs 22.1%, +4.6pp). Smallest: Logan Henderson and Parker Messick (both +1.2pp).
- **Systematic underprediction**: predicted < actual for all 10 pitchers. Plausible causes: the model was trained on September 2026 data only, while these samples span March–October including spring training and postseason; and 2026 pitcher pools differ from the training pool. The reports should keep stating this, not hide it.
- **Window honesty**: none of these samples is pure regular season. Fetched windows run 2026-03-01 → 2026-10-20 and mix game types (R regular season, S spring training, F Wild Card, D Division Series, L League Championship Series; no World Series games in these samples). Every report states its own window and game-type mix; do not describe any of them as regular-season-only.
- **No fetch failures**: all 10 pitchers resolved and ran; 2 from pre-existing cache (Misiorowski, Messick), 8 fetched via pybaseball and saved to ../data/ for reuse.
- **Small-sample rule**: per-pitch-type rates with n<50 are flagged in the reports and should be quoted with wide uncertainty, never as precise as the main pitches.
