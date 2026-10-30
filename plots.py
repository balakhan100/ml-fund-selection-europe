"""Figures (saved to outputs/figures)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

NAVY, BLUE, TEAL, GOLD, SLATE, CRIMSON, GRID = (
    "#0B1F3A", "#3E5C76", "#2A9D8F", "#C9A227", "#6C7A89", "#A4243B", "#D9E1EA")
PALETTE = [NAVY, TEAL, GOLD, BLUE, CRIMSON, SLATE]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": SLATE,
        "axes.labelcolor": NAVY, "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 11, "xtick.color": SLATE, "ytick.color": SLATE, "grid.color": GRID,
        "grid.linewidth": 0.8, "font.size": 10, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    d = C.OUTPUT_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / name, dpi=200, bbox_inches="tight")


def correlation_matrix(panel: pd.DataFrame, cols: list[str], name="corr_matrix.png"):
    corr = panel[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(np.ma.masked_where(mask, corr.values), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("_z_lag1", "") for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=9)
    ax.set_yticks(range(len(cols)), labels, fontsize=9)
    ax.set_title("Correlation of lagged standardised predictors", color=NAVY)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, name)
    return fig


def feature_importance(summary: pd.DataFrame, name="feature_importance.png"):
    lin = summary[summary["kind"] == "coef"].pivot(index="feature", columns="model", values="mean")
    tree = summary[summary["kind"] == "importance"].pivot(index="feature", columns="model",
                                                         values="mean")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, df, title in [(axes[0], lin, "Mean signed coefficient (linear models)"),
                          (axes[1], tree, "Mean feature importance (tree models)")]:
        if df.empty:
            ax.set_visible(False)
            continue
        df = df.loc[df.abs().mean(axis=1).sort_values().index]
        df.index = [i.replace("_z_lag1", "") for i in df.index]
        df.plot.barh(ax=ax, width=0.75, color=PALETTE[:df.shape[1]])
        ax.set_title(title)
        ax.axvline(0, color=SLATE, lw=1)
        ax.grid(axis="x", alpha=0.6)
        ax.legend(loc="lower right", title="Model")
    fig.tight_layout()
    _save(fig, name)
    return fig


def misallocation(dec: pd.DataFrame, name="capital_misallocation.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(dec.index, dec["mean_size"], color=NAVY)
    axes[0].set(title="Fund size at formation by predicted-alpha decile",
                xlabel="Predicted-alpha decile", ylabel="Mean TNA (EUR m)")
    axes[1].bar(dec.index, dec["realised_alpha_pct"], color=TEAL)
    axes[1].axhline(0, color=SLATE, lw=1)
    axes[1].set(title="Realised alpha by predicted-alpha decile",
                xlabel="Predicted-alpha decile", ylabel="Annual alpha (%)")
    for ax in axes:
        ax.set_xticks(dec.index)
        ax.grid(axis="y", alpha=0.6)
    fig.tight_layout()
    _save(fig, name)
    return fig


def cumulative_alpha(cum: pd.DataFrame, title: str, name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(cum.columns):
        ax.plot(cum.index, cum[c] * 100, lw=2.2, color=PALETTE[i % len(PALETTE)], label=c)
    ax.axhline(0, color=SLATE, lw=0.8, ls="--")
    ax.set(title=title, ylabel="Cumulative alpha (%)")
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, name)
    return fig
