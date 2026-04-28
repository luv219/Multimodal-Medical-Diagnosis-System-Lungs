"""
dashboard.py — Multi-model comparison dashboard for NIH ChestX-ray14.

Loads evaluation JSON reports produced by evaluate.py --save-report and
renders a 2×2 panel figure:
  Top-left  : AUC heatmap (14 diseases × 3 models)
  Top-right : F1 / precision / recall heatmap
  Bot-left  : Macro AUC bar chart  +  per-disease AUC line plot
  Bot-right : Summary table with Best-Model column

Usage:
    python dashboard.py
    python dashboard.py --results-dir results/ --output results/dashboard.png
    python dashboard.py --metric precision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

# Consistent per-model colour palette used in every panel
MODEL_COLORS = {
    "densenet": "steelblue",
    "swin":     "coral",
    "hybrid":   "goldenrod",
}


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def load_reports(
    results_dir: str,
    models: list[str] | None = None,
) -> dict[str, dict]:
    """Load the most recent evaluation report for each model.

    Scans *results_dir* for JSON files whose names start with ``{model}_``
    and picks the lexicographically latest (ISO timestamps sort correctly).

    Args:
        results_dir: Directory that contains report JSON files.
        models:      Model names to load; defaults to all three.

    Returns:
        ``{model_name: report_dict}`` where each report_dict is the parsed
        JSON from evaluate.py.

    Raises:
        FileNotFoundError: If any model's report is absent.
    """
    if models is None:
        models = ["densenet", "swin", "hybrid"]

    results_path = Path(results_dir)
    reports: dict[str, dict] = {}

    print(f"Loading reports from: {results_path.resolve()}")
    for model_name in models:
        matches = sorted(results_path.glob(f"{model_name}_*.json"))
        if not matches:
            raise FileNotFoundError(
                f"No report found for model '{model_name}' in '{results_path}'. "
                "Run  python evaluate.py --model {model} ... --save-report  first."
            )
        latest = matches[-1]          # ISO timestamp → lex-sort == time-sort
        with open(latest) as fh:
            reports[model_name] = json.load(fh)
        print(f"  [{model_name:8s}]  {latest.name}")

    return reports


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_auc_dataframe(
    reports: dict[str, dict],
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a (14 × n_models) AUC DataFrame and a macro-AUC summary.

    Returns:
        df         : rows=diseases, columns=model names, values=per-class AUC.
        macro_aucs : {model_name: macro_auc}.
    """
    data = {
        model: [
            reports[model]["per_class"].get(cls, {}).get("auc", None)
            for cls in class_names
        ]
        for model in reports
    }
    df = pd.DataFrame(data, index=class_names)
    # None → np.nan so pandas / seaborn handle them uniformly
    df = df.where(df.notna()).astype(float)

    macro_aucs = {model: reports[model]["macro_auc"] for model in reports}
    return df, macro_aucs


def build_metrics_dataframe(
    reports: dict[str, dict],
    class_names: list[str],
    metric: str = "f1",
) -> pd.DataFrame:
    """Build a (14 × n_models) DataFrame for *metric* ("f1", "precision", "recall").

    Returns:
        DataFrame with rows=diseases, columns=model names, values=metric value.
    """
    data = {
        model: [
            reports[model]["per_class"].get(cls, {}).get(metric, None)
            for cls in class_names
        ]
        for model in reports
    }
    df = pd.DataFrame(data, index=class_names)
    return df.where(df.notna()).astype(float)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def plot_auc_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> None:
    """Seaborn heatmap with NaN cells rendered as gray with 'N/A' text.

    Rows are sorted by mean metric value descending so the best-performing
    diseases appear at the top.
    """
    # Sort rows by mean value descending (NaN ignored)
    row_means = df.mean(axis=1, skipna=True)
    df = df.loc[row_means.sort_values(ascending=False).index]

    nan_mask = df.isna()

    # Annotation strings: formatted value or "N/A"
    annot = pd.DataFrame(
        np.where(
            nan_mask,
            "N/A",
            df.map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"),
        ),
        index=df.index,
        columns=df.columns,
    )

    # Plot with NaN filled to vmin (will be overlaid by gray patches)
    sns.heatmap(
        df.fillna(vmin),
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Score"},
        mask=nan_mask,
    )

    # Gray overlay + "N/A" text for NaN cells
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if nan_mask.iloc[i, j]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, i), 1, 1, fill=True, color="lightgray", zorder=3
                    )
                )
                ax.text(
                    j + 0.5, i + 0.5, "N/A",
                    ha="center", va="center", fontsize=9,
                    color="dimgray", zorder=4,
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def plot_macro_bar(ax: plt.Axes, macro_aucs: dict[str, float]) -> None:
    """Horizontal bar chart comparing macro AUC across models."""
    models = list(macro_aucs.keys())
    values = [macro_aucs[m] for m in models]
    colors = [MODEL_COLORS.get(m, "gray") for m in models]

    bars = ax.barh(models, values, color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Macro AUC", fontsize=9)
    ax.set_title("Macro AUC comparison", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)


def plot_per_disease_lines(ax: plt.Axes, auc_df: pd.DataFrame) -> None:
    """Line plot of per-disease AUC with one line per model."""
    # Use the same row order as the heatmap (mean AUC descending)
    row_means = auc_df.mean(axis=1, skipna=True)
    df = auc_df.loc[row_means.sort_values(ascending=False).index]

    x = np.arange(len(df.index))
    for model in df.columns:
        color = MODEL_COLORS.get(model, "gray")
        ax.plot(
            x, df[model].values,
            marker="o", markersize=5,
            label=model, color=color, linewidth=1.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Per-disease AUC by model", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_summary_table(
    ax: plt.Axes,
    reports: dict[str, dict],
    class_names: list[str],
) -> None:
    """Matplotlib table: Disease | DenseNet AUC | Swin AUC | Hybrid AUC | Best Model.

    Rows sorted by mean AUC descending.  A 'Macro' footer row is appended.
    The 'Best Model' cell is colour-coded to match the bar chart palette.
    """
    models = list(reports.keys())

    def _auc(model: str, cls: str) -> float | None:
        return reports[model]["per_class"].get(cls, {}).get("auc", None)

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    # Sort diseases by mean AUC descending
    mean_auc = {
        cls: np.nanmean([v for m in models if (v := _auc(m, cls)) is not None] or [np.nan])
        for cls in class_names
    }
    sorted_classes = sorted(class_names, key=lambda c: mean_auc[c], reverse=True)

    # Build cell text and track best-model per row
    col_labels  = ["Disease"] + [m.capitalize() for m in models] + ["Best Model"]
    cell_text   = []
    best_models = []   # parallel list for colour-coding

    for cls in sorted_classes:
        aucs       = {m: _auc(m, cls) for m in models}
        valid_aucs = {m: v for m, v in aucs.items() if v is not None}

        if valid_aucs:
            best_m = max(valid_aucs, key=lambda m: valid_aucs[m])
        else:
            best_m = None

        row = [cls] + [_fmt(aucs[m]) for m in models] + [best_m or "N/A"]
        cell_text.append(row)
        best_models.append(best_m)

    # Macro footer
    macro_row = (
        ["Macro AUC"]
        + [f"{reports[m]['macro_auc']:.3f}" for m in models]
        + [""]
    )
    cell_text.append(macro_row)
    best_models.append(None)

    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    # Style header row
    n_cols = len(col_labels)
    for j in range(n_cols):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour-code Best Model column and zebra-stripe data rows
    best_col_idx = n_cols - 1
    for i, best_m in enumerate(best_models):
        row_idx = i + 1  # +1 for header
        # Zebra stripe
        stripe = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[row_idx, j].set_facecolor(stripe)

        if best_m and best_m in MODEL_COLORS:
            cell = table[row_idx, best_col_idx]
            cell.set_facecolor(MODEL_COLORS[best_m])
            cell.set_text_props(color="white", fontweight="bold")

    # Footer (Macro row) — slightly darker stripe
    footer_idx = len(best_models)   # last row
    for j in range(n_cols):
        table[footer_idx, j].set_facecolor("#dfe6e9")
        table[footer_idx, j].set_text_props(fontweight="bold")

    ax.set_title("Per-Class Results Summary", fontsize=12, fontweight="bold", pad=12)


# ---------------------------------------------------------------------------
# Dashboard assembly
# ---------------------------------------------------------------------------

def create_dashboard(
    reports: dict[str, dict],
    class_names: list[str],
    metric: str = "f1",
    output_path: str = "results/dashboard.png",
) -> None:
    """Assemble and save the 2×2 comparison dashboard.

    Layout
    ------
    [AUC heatmap]        [F1/metric heatmap]
    [bar + line plots]   [summary table]

    Args:
        reports:     Loaded report dicts keyed by model name.
        class_names: Canonical list of 14 disease class names.
        metric:      Second-heatmap metric — "f1", "precision", or "recall".
        output_path: Where to save the PNG file.
    """
    auc_df, macro_aucs = build_auc_dataframe(reports, class_names)
    metric_df           = build_metrics_dataframe(reports, class_names, metric)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "NIH ChestX-ray14 — Model Comparison Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs_outer = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # Top row
    ax_auc  = fig.add_subplot(gs_outer[0, 0])
    ax_f1   = fig.add_subplot(gs_outer[0, 1])

    # Bottom-right: summary table
    ax_tbl  = fig.add_subplot(gs_outer[1, 1])

    # Bottom-left: bar chart stacked above line plot
    gs_bl   = gs_outer[1, 0].subgridspec(2, 1, hspace=0.55, height_ratios=[1, 2])
    ax_bar  = fig.add_subplot(gs_bl[0])
    ax_line = fig.add_subplot(gs_bl[1])

    # ---- Panels ----------------------------------------------------------
    plot_auc_heatmap(
        ax_auc, auc_df,
        title="Per-class AUC-ROC  (sorted by mean AUC)",
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
    )
    plot_auc_heatmap(
        ax_f1, metric_df,
        title=f"Per-class {metric.upper()}  (sorted by mean {metric.upper()})",
        cmap="Blues", vmin=0.0, vmax=1.0,
    )
    plot_macro_bar(ax_bar, macro_aucs)
    plot_per_disease_lines(ax_line, auc_df)
    plot_summary_table(ax_tbl, reports, class_names)

    # ---- Save ------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved → {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIH ChestX-ray14 — multi-model comparison dashboard"
    )
    p.add_argument("--results-dir", type=str, default="results/",
                   help="Directory containing evaluate.py JSON reports")
    p.add_argument("--output",      type=str, default="results/dashboard.png",
                   help="Output PNG path")
    p.add_argument("--models",      nargs="+",
                   default=["densenet", "swin", "hybrid"],
                   help="Model names to include")
    p.add_argument("--metric",      choices=["f1", "precision", "recall"],
                   default="f1",
                   help="Metric for the second heatmap panel")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = parse_args()
    class_names = cfg.DATASET["class_names"]

    reports = load_reports(args.results_dir, models=args.models)
    create_dashboard(
        reports=reports,
        class_names=class_names,
        metric=args.metric,
        output_path=args.output,
    )
    print(f"Done. Dashboard written to: {args.output}")


if __name__ == "__main__":
    main()
