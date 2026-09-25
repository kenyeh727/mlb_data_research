"""Figures for the whiff model report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def whiff_by_pitch_type(d: pd.DataFrame, path="whiff_by_pitch_type.png"):
    g = d.groupby("pitch_type")["whiff"].agg(["mean", "count"])
    g = g[g["count"] >= 200].sort_values("mean")
    plt.figure(figsize=(8, 5))
    plt.barh(g.index, g["mean"] * 100, color="#1f77b4")
    for i, (v, c) in enumerate(zip(g["mean"] * 100, g["count"])):
        plt.text(v + 0.3, i, f"{v:.1f}% (n={c:,})", va="center", fontsize=9)
    plt.xlabel("Whiff rate (%)")
    plt.title("Whiff rate by pitch type (on swings)")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, path), dpi=150)
    plt.close()


def whiff_heatmap(d: pd.DataFrame, path="whiff_zone_heatmap.png"):
    sub = d.dropna(subset=["plate_x", "plate_z"])
    x = sub["plate_x"].clip(-2, 2)
    z = sub["plate_z"].clip(0.5, 4.5)
    bins = 24
    whiff_sum, xe, ze = np.histogram2d(x, z, bins=bins,
                                       weights=sub["whiff"].values)
    cnt, _, _ = np.histogram2d(x, z, bins=[xe, ze])
    rate = np.divide(whiff_sum, cnt, out=np.full_like(whiff_sum, np.nan),
                     where=cnt >= 30)
    plt.figure(figsize=(7, 7))
    plt.imshow(rate.T, origin="lower", extent=[-2, 2, 0.5, 4.5],
               aspect="auto", cmap="hot", vmin=0, vmax=0.6)
    plt.colorbar(label="Whiff rate")
    # strike zone box
    plt.plot([-0.83, 0.83, 0.83, -0.83, -0.83],
             [1.5, 1.5, 3.5, 3.5, 1.5], "w--", lw=1.5)
    plt.xlabel("Horizontal location (ft)")
    plt.ylabel("Vertical location (ft)")
    plt.title("Whiff rate by pitch location (catcher view)")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, path), dpi=150)
    plt.close()


def feature_importance_bar(imp: pd.Series, path="feature_importance.png"):
    plt.figure(figsize=(8, 5))
    imp.sort_values().plot.barh(color="#2ca02c")
    plt.xlabel("Permutation importance (ROC-AUC drop)")
    plt.title("Top features driving whiff prediction")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, path), dpi=150)
    plt.close()
