#!/usr/bin/env python3
"""
Plot training metrics across epochs for generative model evaluation.
Academic style suitable for MICCAI / NeurIPS.

Usage:
    python plot_metrics.py [--metrics-root METRICS_ROOT] [--models MODEL [MODEL ...]]

Outputs per model (saved to metrics/<model>/plots/):
    fig_fid_2p5d.pdf      — FID 2.5D per plane + average (log scale)
    fig_fid_3d.pdf        — FID 3D
    fig_prdc.pdf          — Precision / Recall / Density / Coverage (2×2)
    fig_waddiv.pdf        — WAD-DIV: empirical ref (left) + zero ref (right)
    fig_hu_wasserstein.pdf— HU Wasserstein distances (foreground)
    fig_summary.pdf       — 2×3 overview of key metrics

Multi-model output (metrics/_comparison/):
    fig_comparison.pdf    — key metrics overlaid per model
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
ACADEMIC_RC = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
    "legend.borderpad": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Monotonic blue palette — light to dark (5 tones)
BLUES = [
    "#BDD7EE",  # B0 — very light
    "#9DC3E6",  # B1 — light
    "#5B9BD5",  # B2 — medium
    "#2E75B6",  # B3 — dark
    "#1F4E79",  # B4 — very dark
]

# Consistent k-value encoding across all plots (k=3,5,10)
K_STYLES = [
    (3,  BLUES[2], "o",  "-"),   # medium, solid, circle
    (5,  BLUES[3], "s",  "--"),  # dark, dashed, square
    (10, BLUES[4], "^",  "-."),  # very dark, dash-dot, triangle
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*"]


_COMP_LINESTYLES = ["-", "--", "-."]


def _model_styles(models: list[str]) -> list[tuple]:
    """(color, marker, linestyle) per model — blue ramp + cycling line styles."""
    n = len(models)
    colors = [plt.cm.Blues(v) for v in np.linspace(0.35, 0.92, max(n, 2))]
    return [
        (colors[i], MARKERS[i % len(MARKERS)], _COMP_LINESTYLES[i % len(_COMP_LINESTYLES)])
        for i in range(n)
    ]


def _add_fig_legend(fig, ax, models: list[str], ncol: int = 3) -> None:
    """Shared legend below figure using lines from ax."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=ncol,
               fontsize=7, framealpha=0.9, edgecolor="0.8")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_model_data(metrics_root: Path, model: str) -> dict:
    """Return dict with epoch list and per-epoch DataFrames for each metric family."""
    model_dir = metrics_root / model
    epoch_dirs = sorted(
        model_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    epochs = [int(p.name.split("_")[1]) for p in epoch_dirs]

    records: dict[str, list[pd.DataFrame]] = {
        k: [] for k in ("fid_2p5d", "fid_3d", "prdc", "waddiv", "hu")
    }
    for ep in epoch_dirs:
        records["fid_2p5d"].append(_load_tsv(ep / "fid_2p5d" / "fid_results.tsv"))
        records["fid_3d"].append(_load_tsv(ep / "fid_3d" / "fid_results.tsv"))
        records["prdc"].append(_load_tsv(ep / "prdc" / "prdc_results.tsv"))
        records["waddiv"].append(_load_tsv(ep / "waddiv" / "waddiv_results.tsv"))
        records["hu"].append(_load_tsv(ep / "hu_distribution" / "hu_distribution_summary.tsv"))

    return {"epochs": epochs, **records}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _get(df: pd.DataFrame, mask, col: str = "value") -> float:
    rows = df[mask]
    return float(rows[col].iloc[0]) if not rows.empty else np.nan


def fid2p5d_series(frames, plane: str, metric: str = "FID") -> list[float]:
    return [_get(df, (df["plane"] == plane) & (df["metric"] == metric)) for df in frames]


def fid3d_series(frames, metric: str = "FID") -> list[float]:
    return [_get(df, (df["plane"] == "3d") & (df["metric"] == metric)) for df in frames]


def prdc_series(frames, k: int, metric: str) -> list[float]:
    return [_get(df, (df["k"] == k) & (df["metric"] == metric)) for df in frames]


def waddiv_series(frames, ref: str, k: int) -> list[float]:
    return [_get(df, (df["reference"] == ref) & (df["k"] == k), col="wad_div") for df in frames]


def hu_series(frames, metric: str) -> list[float]:
    return [_get(df, df["metric"] == metric) for df in frames]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _better_tag(ax, direction: str = "down") -> None:
    """Bottom-right annotation for optimisation direction (avoids legend overlap)."""
    sym = "↓" if direction == "down" else "↑"
    ax.text(0.98, 0.03, f"{sym} better",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.5", style="italic")


def _epoch_ticks(ax, epochs: list[int]) -> None:
    step = epochs[1] - epochs[0] if len(epochs) > 1 else 300
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.set_xlim(epochs[0] - step * 0.15, epochs[-1] + step * 0.15)


def _epoch_ticks_multi(ax, all_epochs: list[int]) -> None:
    """Epoch ticks for comparison plots spanning multiple models' epoch ranges."""
    if len(all_epochs) < 2:
        return
    step = all_epochs[1] - all_epochs[0]
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.set_xlim(all_epochs[0] - step * 0.15, all_epochs[-1] + step * 0.15)


def _save(fig, out_dir: Path, stem: str, tag: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}  ({tag})")


# ---------------------------------------------------------------------------
# Figure: FID 2.5D
# ---------------------------------------------------------------------------

def plot_fid_2p5d(data: dict, out_dir: Path, label: str) -> None:
    epochs = data["epochs"]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    plane_cfg = [
        ("xy",       "XY plane", BLUES[1], "o",  "-",  1.4),
        ("yz",       "YZ plane", BLUES[2], "s",  "-",  1.4),
        ("zx",       "ZX plane", BLUES[3], "^",  "-",  1.4),
        ("avg_2p5d", "Average",  BLUES[4], "D",  "--", 2.0),
    ]
    for plane, plabel, color, marker, ls, lw in plane_cfg:
        vals = fid2p5d_series(data["fid_2p5d"], plane)
        ax.plot(epochs, vals, color=color, marker=marker, ls=ls, lw=lw,
                label=plabel, zorder=3 if plane == "avg_2p5d" else 2)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("FID — 2.5D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID (log scale)")
    ax.legend(loc="upper right")
    _better_tag(ax)
    _epoch_ticks(ax, epochs)

    _save(fig, out_dir, "fig_fid_2p5d", label)


# ---------------------------------------------------------------------------
# Figure: FID 3D
# ---------------------------------------------------------------------------

def plot_fid_3d(data: dict, out_dir: Path, label: str) -> None:
    epochs = data["epochs"]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    vals = fid3d_series(data["fid_3d"])
    ax.plot(epochs, vals, color=BLUES[3], marker="o", lw=1.6)

    ax.set_title("FID — 3D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _better_tag(ax)
    _epoch_ticks(ax, epochs)

    _save(fig, out_dir, "fig_fid_3d", label)


# ---------------------------------------------------------------------------
# Figure: PRDC (2×2 subfigures)
# ---------------------------------------------------------------------------

def plot_prdc(data: dict, out_dir: Path, label: str) -> None:
    epochs = data["epochs"]
    metrics_cfg = [
        ("precision", "Precision", "upper left"),
        ("recall",    "Recall",    "upper left"),
        ("density",   "Density",   "upper left"),
        ("coverage",  "Coverage",  "upper left"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, leg_loc) in zip(axes, metrics_cfg):
        for k, color, marker, ls in K_STYLES:
            vals = prdc_series(data["prdc"], k, metric)
            ax.plot(epochs, vals, color=color, marker=marker, ls=ls,
                    lw=1.6, label=f"$k={k}$")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(loc=leg_loc)
        _better_tag(ax, "up")
        _epoch_ticks(ax, epochs)
        ax.set_ylim(bottom=0.0)

    _save(fig, out_dir, "fig_prdc", label)


# ---------------------------------------------------------------------------
# Figure: WAD-DIV (empirical left, zero right)
# ---------------------------------------------------------------------------

def plot_waddiv(data: dict, out_dir: Path, label: str) -> None:
    epochs = data["epochs"]
    refs = [
        ("empirical", "Empirical reference"),
        ("zero",      "Zero reference"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             constrained_layout=True)

    for ax, (ref, ref_label) in zip(axes, refs):
        for k, color, marker, ls in K_STYLES:
            vals = waddiv_series(data["waddiv"], ref=ref, k=k)
            ax.plot(epochs, vals, color=color, marker=marker, ls=ls,
                    lw=1.6, label=f"$k={k}$")
        ax.set_title(f"WAD-DIV — {ref_label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("WAD-DIV")
        ax.legend(loc="upper right")
        _better_tag(ax, "down")
        _epoch_ticks(ax, epochs)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    _save(fig, out_dir, "fig_waddiv", label)


# ---------------------------------------------------------------------------
# Figure: HU Wasserstein (foreground)
# ---------------------------------------------------------------------------

def plot_hu_wasserstein(data: dict, out_dir: Path, label: str) -> None:
    epochs = data["epochs"]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    vals_pop  = hu_series(data["hu"], "wasserstein_population_foreground")
    vals_mean = hu_series(data["hu"], "wasserstein_per_volume_fg_mean")
    vals_p25  = hu_series(data["hu"], "wasserstein_per_volume_fg_p25")
    vals_p75  = hu_series(data["hu"], "wasserstein_per_volume_fg_p75")

    ax.plot(epochs, vals_pop,  color=BLUES[4], marker="o",  ls="-",  lw=1.6,
            label="Pop. foreground")
    ax.plot(epochs, vals_mean, color=BLUES[2], marker="s",  ls="--", lw=1.6,
            label="Per-vol. fg mean")
    ax.fill_between(epochs, vals_p25, vals_p75,
                    color=BLUES[2], alpha=0.18, label="Per-vol. fg IQR")

    ax.set_title("HU Wasserstein — Foreground")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wasserstein Distance (HU)")
    ax.legend(loc="upper right")
    _better_tag(ax, "down")
    _epoch_ticks(ax, epochs)

    _save(fig, out_dir, "fig_hu_wasserstein", label)


# ---------------------------------------------------------------------------
# Figure: Summary (2×3)
# ---------------------------------------------------------------------------

def plot_summary(data: dict, out_dir: Path, label: str) -> None:
    """Compact 2×3 overview of key metrics for at-a-glance training progress."""
    epochs = data["epochs"]

    panels = [
        # (title, ylabel, direction, yscale, draw_fn)
        (
            "FID — 2.5D", "FID (log scale)", "down", "log",
            lambda ax: [
                ax.plot(epochs, fid2p5d_series(data["fid_2p5d"], pl), color=c,
                        marker=m, ls=ls, lw=lw, label=lbl)
                for pl, lbl, c, m, ls, lw in [
                    ("xy",       "XY",  BLUES[1], "o", "-",  1.3),
                    ("yz",       "YZ",  BLUES[2], "s", "-",  1.3),
                    ("zx",       "ZX",  BLUES[3], "^", "-",  1.3),
                    ("avg_2p5d", "Avg", BLUES[4], "D", "--", 2.0),
                ]
            ],
        ),
        (
            "FID — 3D", "FID", "down", "linear",
            lambda ax: ax.plot(epochs, fid3d_series(data["fid_3d"]),
                               color=BLUES[3], marker="o", lw=1.6),
        ),
        (
            "Precision & Recall", "Score", "up", "linear",
            lambda ax: [
                ax.plot(epochs, prdc_series(data["prdc"], 5, m), color=c,
                        marker=mk, ls=ls, lw=1.6, label=lbl)
                for m, lbl, c, mk, ls in [
                    ("precision", "Precision $k\!=\!5$", BLUES[4], "o", "-"),
                    ("recall",    "Recall $k\!=\!5$",    BLUES[2], "s", "--"),
                ]
            ],
        ),
        (
            "Density & Coverage", "Score", "up", "linear",
            lambda ax: [
                ax.plot(epochs, prdc_series(data["prdc"], 5, m), color=c,
                        marker=mk, ls=ls, lw=1.6, label=lbl)
                for m, lbl, c, mk, ls in [
                    ("density",  "Density $k\!=\!5$",  BLUES[4], "o", "-"),
                    ("coverage", "Coverage $k\!=\!5$", BLUES[2], "s", "--"),
                ]
            ],
        ),
        (
            "WAD-DIV (empirical ref.)", "WAD-DIV", "down", "linear",
            lambda ax: [
                ax.plot(epochs, waddiv_series(data["waddiv"], "empirical", k),
                        color=color, marker=marker, ls=ls, lw=1.6, label=f"$k={k}$")
                for k, color, marker, ls in K_STYLES
            ],
        ),
        (
            "HU Wasserstein — fg", "Wasserstein (HU)", "down", "linear",
            lambda ax: [
                ax.plot(epochs, hu_series(data["hu"], "wasserstein_population_foreground"),
                        color=BLUES[4], marker="o", ls="-",  lw=1.6, label="Pop. fg"),
                ax.plot(epochs, hu_series(data["hu"], "wasserstein_per_volume_fg_mean"),
                        color=BLUES[2], marker="s", ls="--", lw=1.6, label="Per-vol. fg"),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, ylabel, direction, yscale, draw) in zip(axes, panels):
        draw(ax)
        ax.set_yscale(yscale)
        if yscale == "log":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        _better_tag(ax, direction)
        _epoch_ticks(ax, epochs)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right" if direction == "down" else "lower right")

    fig.suptitle(f"Training Metrics — {label}", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_summary", label)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def _all_epochs(all_data: dict) -> list[int]:
    return sorted({e for mdata in all_data.values() for e in mdata["epochs"]})


def plot_comparison(all_data: dict[str, dict], out_dir: Path) -> None:
    """2×3 summary comparison — key metrics overlaid per model."""
    models = list(all_data.keys())
    if len(models) < 2:
        return

    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    extractors = [
        ("FID — 2.5D Avg",        "FID (log)",    "down", "log",
         lambda d: fid2p5d_series(d["fid_2p5d"], "avg_2p5d")),
        ("FID — 3D",              "FID",           "down", "linear",
         lambda d: fid3d_series(d["fid_3d"])),
        ("Precision ($k=5$)",     "Precision",     "up",   "linear",
         lambda d: prdc_series(d["prdc"], 5, "precision")),
        ("Recall ($k=5$)",        "Recall",        "up",   "linear",
         lambda d: prdc_series(d["prdc"], 5, "recall")),
        ("WAD-DIV ($k=5$, emp.)", "WAD-DIV",       "down", "linear",
         lambda d: waddiv_series(d["waddiv"], "empirical", 5)),
        ("HU Wasserstein — fg",   "W-dist (HU)",   "down", "linear",
         lambda d: hu_series(d["hu"], "wasserstein_population_foreground")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, ylabel, direction, yscale, extractor) in zip(axes, extractors):
        for i, (model_name, mdata) in enumerate(all_data.items()):
            color, marker, ls = styles[i]
            ax.plot(mdata["epochs"], extractor(mdata),
                    color=color, marker=marker, ls=ls, lw=1.4, label=model_name)
        ax.set_yscale(yscale)
        if yscale == "log":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        _better_tag(ax, direction)
        _epoch_ticks_multi(ax, merged_epochs)

    _add_fig_legend(fig, axes[0], models, ncol=3)
    fig.suptitle("Model Comparison — Summary", fontsize=11, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir, "fig_comparison", "comparison")


def plot_comparison_fid_2p5d(all_data: dict[str, dict], out_dir: Path) -> None:
    """One subplot per 2.5D plane, all models overlaid."""
    models = list(all_data.keys())
    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    plane_cfg = [
        ("xy",       "XY Plane"),
        ("yz",       "YZ Plane"),
        ("zx",       "ZX Plane"),
        ("avg_2p5d", "Average"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (plane, plane_label) in zip(axes, plane_cfg):
        for i, (model_name, mdata) in enumerate(all_data.items()):
            color, marker, ls = styles[i]
            vals = fid2p5d_series(mdata["fid_2p5d"], plane)
            ax.plot(mdata["epochs"], vals, color=color, marker=marker,
                    ls=ls, lw=1.4, label=model_name)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_title(f"FID 2.5D — {plane_label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID (log scale)")
        _better_tag(ax)
        _epoch_ticks_multi(ax, merged_epochs)

    _add_fig_legend(fig, axes[0], models, ncol=3)
    fig.suptitle("Model Comparison — FID 2.5D", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_comparison_fid_2p5d", "comparison")


def plot_comparison_fid_3d(all_data: dict[str, dict], out_dir: Path) -> None:
    """FID 3D for all models."""
    models = list(all_data.keys())
    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)

    for i, (model_name, mdata) in enumerate(all_data.items()):
        color, marker, ls = styles[i]
        vals = fid3d_series(mdata["fid_3d"])
        ax.plot(mdata["epochs"], vals, color=color, marker=marker,
                ls=ls, lw=1.4, label=model_name)

    ax.set_title("FID — 3D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=7, loc="upper right")
    _better_tag(ax)
    _epoch_ticks_multi(ax, merged_epochs)

    fig.suptitle("Model Comparison — FID 3D", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_comparison_fid_3d", "comparison")


def plot_comparison_prdc(all_data: dict[str, dict], out_dir: Path) -> None:
    """Precision / Recall / Density / Coverage (k=5) for all models."""
    models = list(all_data.keys())
    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    metrics_cfg = [
        ("precision", "Precision", "up"),
        ("recall",    "Recall",    "up"),
        ("density",   "Density",   "up"),
        ("coverage",  "Coverage",  "up"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, direction) in zip(axes, metrics_cfg):
        for i, (model_name, mdata) in enumerate(all_data.items()):
            color, marker, ls = styles[i]
            vals = prdc_series(mdata["prdc"], 5, metric)
            ax.plot(mdata["epochs"], vals, color=color, marker=marker,
                    ls=ls, lw=1.4, label=model_name)
        ax.set_title(f"{title} ($k=5$)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0.0)
        _better_tag(ax, direction)
        _epoch_ticks_multi(ax, merged_epochs)

    _add_fig_legend(fig, axes[0], models, ncol=3)
    fig.suptitle("Model Comparison — PRDC", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_comparison_prdc", "comparison")


def plot_comparison_waddiv(all_data: dict[str, dict], out_dir: Path) -> None:
    """WAD-DIV (k=5, empirical & zero reference) for all models."""
    models = list(all_data.keys())
    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    refs = [
        ("empirical", "Empirical Reference"),
        ("zero",      "Zero Reference"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    for ax, (ref, ref_label) in zip(axes, refs):
        for i, (model_name, mdata) in enumerate(all_data.items()):
            color, marker, ls = styles[i]
            vals = waddiv_series(mdata["waddiv"], ref=ref, k=5)
            ax.plot(mdata["epochs"], vals, color=color, marker=marker,
                    ls=ls, lw=1.4, label=model_name)
        ax.set_title(f"WAD-DIV — {ref_label} ($k=5$)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("WAD-DIV")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        _better_tag(ax)
        _epoch_ticks_multi(ax, merged_epochs)

    _add_fig_legend(fig, axes[0], models, ncol=3)
    fig.suptitle("Model Comparison — WAD-DIV", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_comparison_waddiv", "comparison")


def plot_comparison_hu(all_data: dict[str, dict], out_dir: Path) -> None:
    """HU Wasserstein (population fg + per-volume fg mean) for all models."""
    models = list(all_data.keys())
    styles = _model_styles(models)
    merged_epochs = _all_epochs(all_data)

    metrics_cfg = [
        ("wasserstein_population_foreground", "Population Foreground"),
        ("wasserstein_per_volume_fg_mean",    "Per-Volume FG Mean"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    for ax, (metric, title) in zip(axes, metrics_cfg):
        for i, (model_name, mdata) in enumerate(all_data.items()):
            color, marker, ls = styles[i]
            vals = hu_series(mdata["hu"], metric)
            ax.plot(mdata["epochs"], vals, color=color, marker=marker,
                    ls=ls, lw=1.4, label=model_name)
        ax.set_title(f"HU Wasserstein — {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Wasserstein Distance (HU)")
        _better_tag(ax)
        _epoch_ticks_multi(ax, merged_epochs)

    _add_fig_legend(fig, axes[0], models, ncol=3)
    fig.suptitle("Model Comparison — HU Wasserstein", fontsize=11, fontweight="bold")
    _save(fig, out_dir, "fig_comparison_hu", "comparison")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-root", type=Path,
                   required=True,
                   help="Root of the metrics directory tree.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model folder names to process (default: all subdirs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_root: Path = args.metrics_root

    model_names = args.models or sorted(
        p.name for p in metrics_root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )
    if not model_names:
        raise SystemExit(f"No model directories found in {metrics_root}")

    all_data: dict[str, dict] = {}

    with plt.rc_context(ACADEMIC_RC):
        for model in model_names:
            print(f"\nModel: {model}")
            data = load_model_data(metrics_root, model)
            all_data[model] = data

            out_dir = metrics_root / model / "plots"
            out_dir.mkdir(exist_ok=True)
            label = model.replace("_", " ").title()

            plot_fid_2p5d(data, out_dir, label)
            plot_fid_3d(data, out_dir, label)
            plot_prdc(data, out_dir, label)
            plot_waddiv(data, out_dir, label)
            plot_hu_wasserstein(data, out_dir, label)
            plot_summary(data, out_dir, label)

        if len(model_names) > 1:
            print("\nGenerating comparison figures …")
            cmp_dir = metrics_root / "_comparison"
            cmp_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison(all_data, out_dir=cmp_dir)
            plot_comparison_fid_2p5d(all_data, out_dir=cmp_dir)
            plot_comparison_fid_3d(all_data, out_dir=cmp_dir)
            plot_comparison_prdc(all_data, out_dir=cmp_dir)
            plot_comparison_waddiv(all_data, out_dir=cmp_dir)
            plot_comparison_hu(all_data, out_dir=cmp_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
