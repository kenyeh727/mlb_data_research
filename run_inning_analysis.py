"""Whiff rate by pitch type x inning: answers ken's question on inning-level differences.
Usage: .venv/bin/python run_inning_analysis.py
Outputs: figures/whiff_by_inning_group.png, results printed to stdout."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWINGS = {"foul","foul_tip","hit_into_play","hit_into_play_no_out","hit_into_play_score",
          "swinging_strike","swinging_strike_blocked","foul_bunt","missed_bunt"}
WHIFF = {"swinging_strike","swinging_strike_blocked"}
PITCH_EN = {'FF':'Four-seam','SI':'Sinker','FC':'Cutter','CU':'Curve','KC':'Knuckle curve',
            'SL':'Slider','ST':'Sweeper','CH':'Changeup','FS':'Splitter'}

df = pd.read_parquet('data/statcast_sep2026.parquet')
d = df[df['description'].isin(SWINGS)].copy()
d['whiff'] = d['description'].isin(WHIFF).astype(int)
d['inn_grp'] = pd.cut(d['inning'], bins=[0,3,6,9,20], labels=['Inn 1-3','Inn 4-6','Inn 7-9','Inn 10+'])
d['pt'] = d['pitch_type'].map(PITCH_EN).fillna(d['pitch_type'])

t = d.groupby(['pt','inn_grp'], observed=True)['whiff'].agg(['mean','count'])
t = t[t['count'] >= 150]['mean'].unstack('inn_grp')
order = ['Inn 1-3','Inn 4-6','Inn 7-9']
t = t[[c for c in order if c in t.columns]]
t = t.reindex(t['Inn 1-3'].sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(t))
w = 0.22
for i, col in enumerate(t.columns):
    ax.bar(x + (i-1)*w, t[col].values, width=w, label=col)
ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=20, ha='right')
ax.set_ylabel('Whiff rate (among swings)')
ax.set_title('Whiff rate by pitch type × inning (Sep 2026 Statcast, n=38,649 swings)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for xi, vals in zip(x, t.values):
    for j, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(xi+(j-1)*w, v+0.004, f'{v:.1%}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig('figures/whiff_by_inning_group.png', dpi=150)
print(t.round(4).to_string())
print("\nFigure -> figures/whiff_by_inning_group.png")

# times through the order
tto = d.groupby('n_thruorder_pitcher')['whiff'].agg(['mean','count'])
print("\n== times through order ==")
print(tto.round(4).to_string())

# count situation
d['count_sit'] = np.where(d['strikes']==2,'2 strikes', np.where(d['balls']>d['strikes'],'pitcher ahead','even/behind'))
cs = d.groupby('count_sit')['whiff'].agg(['mean','count'])
print("\n== count situation ==")
print(cs.round(4).to_string())
