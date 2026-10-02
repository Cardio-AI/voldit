#!/usr/bin/env python3
"""
Select the best epoch per model from the metrics tree.

The default score is a balanced 0-100 composite over five metric families:
2.5D FID average, 3D FID, PRDC mean at k=5, empirical WAD-DIV at k=5, and
HU foreground Wasserstein. Lower-is-better metrics are inverted before scoring.

Outputs are written to metrics/_comparison/:
    model_selection_all_epochs.tsv
    model_selection_summary.tsv
    model_selection_summary.md
    fig_model_selection_scores.{png,pdf}
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreMetric:
    column: str
    label: str
    direction: str
    weight: float = 1.0


SCORE_METRICS = [
    ScoreMetric("fid_2p5d_avg", "FID 2.5D avg", "lower"),
    ScoreMetric("fid_3d", "FID 3D", "lower"),
    ScoreMetric("prdc_mean_k5", "PRDC mean k=5", "higher"),
    ScoreMetric("waddiv_empirical_k5", "WAD-DIV empirical k=5", "lower"),
    ScoreMetric("hu_wasserstein_fg", "HU W-dist foreground", "lower"),
]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def single_value(df: pd.DataFrame, mask: pd.Series, column: str = "value") -> float:
    values = df.loc[mask, column]
    return float(values.iloc[0]) if len(values) else np.nan


def epoch_record(model_dir: Path, epoch_dir: Path) -> dict[str, float | int | str]:
    epoch = int(epoch_dir.name.split("_", maxsplit=1)[1])

    fid_2p5d = read_tsv(epoch_dir / "fid_2p5d" / "fid_results.tsv")
    fid_3d = read_tsv(epoch_dir / "fid_3d" / "fid_results.tsv")
    prdc = read_tsv(epoch_dir / "prdc" / "prdc_results.tsv")
    waddiv = read_tsv(epoch_dir / "waddiv" / "waddiv_results.tsv")
    hu = read_tsv(epoch_dir / "hu_distribution" / "hu_distribution_summary.tsv")

    record: dict[str, float | int | str] = {
        "model": model_dir.name,
        "epoch": epoch,
        "fid_2p5d_avg": single_value(
            fid_2p5d,
            (fid_2p5d["plane"] == "avg_2p5d") & (fid_2p5d["metric"] == "FID"),
        ),
        "fid_3d": single_value(
            fid_3d,
            (fid_3d["plane"] == "3d") & (fid_3d["metric"] == "FID"),
        ),
        "waddiv_empirical_k5": single_value(
            waddiv,
            (waddiv["reference"] == "empirical") & (waddiv["k"] == 5),
            column="wad_div",
        ),
        "hu_wasserstein_fg": single_value(
            hu,
            hu["metric"] == "wasserstein_population_foreground",
        ),
    }

    prdc_values = []
    for metric in ("precision", "recall", "density", "coverage"):
        value = single_value(prdc, (prdc["k"] == 5) & (prdc["metric"] == metric))
        record[f"prdc_{metric}_k5"] = value
        prdc_values.append(value)
    record["prdc_mean_k5"] = float(np.nanmean(prdc_values))

    return record


def collect_metrics(metrics_root: Path, models: list[str] | None) -> pd.DataFrame:
    if models:
        model_dirs = [metrics_root / model for model in models]
    else:
        model_dirs = sorted(
            path
            for path in metrics_root.iterdir()
            if path.is_dir() and not path.name.startswith((".", "_"))
        )

    rows = []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        epoch_dirs = sorted(
            model_dir.glob("epoch_*"),
            key=lambda path: int(path.name.split("_", maxsplit=1)[1]),
        )
        if not epoch_dirs:
            raise FileNotFoundError(f"No epoch directories found in: {model_dir}")
        for epoch_dir in epoch_dirs:
            rows.append(epoch_record(model_dir, epoch_dir))

    return pd.DataFrame(rows).sort_values(["model", "epoch"]).reset_index(drop=True)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    weighted_score_columns = []

    for spec in SCORE_METRICS:
        values = scored[spec.column].astype(float)
        best_col = f"score_{spec.column}"

        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            scored[best_col] = np.nan
        elif max_value == min_value:
            scored[best_col] = 1.0
        elif spec.direction == "lower":
            scored[best_col] = (max_value - values) / (max_value - min_value)
        elif spec.direction == "higher":
            scored[best_col] = (values - min_value) / (max_value - min_value)
        else:
            raise ValueError(f"Unknown direction for {spec.column}: {spec.direction}")

        weighted_col = f"weighted_{spec.column}"
        scored[weighted_col] = scored[best_col] * spec.weight
        weighted_score_columns.append(weighted_col)

    total_weight = sum(spec.weight for spec in SCORE_METRICS)
    scored["selection_score"] = scored[weighted_score_columns].sum(axis=1) / total_weight * 100.0
    return scored


def select_best_epochs(scored: pd.DataFrame) -> pd.DataFrame:
    selected = (
        scored.sort_values(["model", "selection_score", "epoch"], ascending=[True, False, True])
        .groupby("model", as_index=False)
        .first()
        .sort_values("selection_score", ascending=False)
        .reset_index(drop=True)
    )
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["n_epochs"] = selected["model"].map(scored.groupby("model")["epoch"].nunique())
    return selected


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    formatted = table.astype(str)
    headers = list(formatted.columns)
    rows = formatted.values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def write_markdown(selected: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "rank",
        "model",
        "epoch",
        "selection_score",
        "fid_2p5d_avg",
        "fid_3d",
        "prdc_mean_k5",
        "waddiv_empirical_k5",
        "hu_wasserstein_fg",
    ]
    table = selected[cols].rename(columns={"epoch": "best_epoch"}).copy()
    table["selection_score"] = table["selection_score"].map(lambda value: f"{value:.2f}")
    table["fid_2p5d_avg"] = table["fid_2p5d_avg"].map(lambda value: f"{value:.1f}")
    table["fid_3d"] = table["fid_3d"].map(lambda value: f"{value:.5f}")
    table["prdc_mean_k5"] = table["prdc_mean_k5"].map(lambda value: f"{value:.4f}")
    table["waddiv_empirical_k5"] = table["waddiv_empirical_k5"].map(lambda value: f"{value:.5f}")
    table["hu_wasserstein_fg"] = table["hu_wasserstein_fg"].map(lambda value: f"{value:.2f}")

    lines = [
        "# Model Selection Summary",
        "",
        "Selection score is an equal-weight 0-100 min-max composite over FID 2.5D avg, "
        "FID 3D, PRDC mean at k=5, empirical WAD-DIV at k=5, and HU foreground "
        "Wasserstein. Higher selection score is better.",
        "",
        dataframe_to_markdown(table),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_scores(selected: pd.DataFrame, output_dir: Path) -> None:
    selected_plot = selected.sort_values("selection_score", ascending=True)
    labels = [
        f"{row.model}\nepoch {int(row.epoch)}"
        for row in selected_plot.itertuples(index=False)
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    fig_height = max(4.0, 0.52 * len(selected_plot) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    colors = plt.cm.Blues(np.linspace(0.38, 0.88, len(selected_plot)))
    bars = ax.barh(labels, selected_plot["selection_score"], color=colors)

    ax.set_title("Model Selection Scores")
    ax.set_xlabel("Selection score (0-100, higher is better)")
    ax.set_xlim(0, 100)

    for bar, row in zip(bars, selected_plot.itertuples(index=False)):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{row.selection_score:.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_model_selection_scores.png")
    fig.savefig(output_dir / "fig_model_selection_scores.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        required=True,
        help="Root of the metrics directory tree.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model directory names to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.metrics_root / "_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_epochs = add_scores(collect_metrics(args.metrics_root, args.models))
    selected = select_best_epochs(all_epochs)

    all_epochs.to_csv(output_dir / "model_selection_all_epochs.tsv", sep="\t", index=False)
    selected.to_csv(output_dir / "model_selection_summary.tsv", sep="\t", index=False)
    write_markdown(selected, output_dir / "model_selection_summary.md")
    plot_scores(selected, output_dir)

    display_cols = [
        "rank",
        "model",
        "epoch",
        "selection_score",
        "fid_2p5d_avg",
        "fid_3d",
        "prdc_mean_k5",
        "waddiv_empirical_k5",
        "hu_wasserstein_fg",
    ]
    print(selected[display_cols].rename(columns={"epoch": "best_epoch"}).to_string(index=False))
    print(f"\nWrote model selection outputs to {output_dir}")


if __name__ == "__main__":
    main()
