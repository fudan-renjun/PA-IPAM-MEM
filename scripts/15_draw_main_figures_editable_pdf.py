from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "13_figure_excel_data_package"
OUT = ROOT / "投稿" / "main_figures_editable_pdf"

DATE = "2026-06-05"
COMBINED_PDF: PdfPages | None = None


COLORS = {
    "IPM": "#177E89",
    "MEM": "#D95F02",
    "warning": "#C43C39",
    "trust": "#2A9D8F",
    "explore": "#7B8FA1",
    "neutral": "#6C757D",
    "light": "#E9ECEF",
    "dark": "#243447",
    "grid": "#D9DEE3",
    "oprD": "#2A9D8F",
    "composite": "#7C3AED",
    "ampc": "#C77D1A",
    "none": "#8D99AE",
}

SUBTYPE_COLORS = {
    "OprD-loss": COLORS["oprD"],
    "High-confidence composite": COLORS["composite"],
    "AmpC-axis disruptive": COLORS["ampc"],
    "No high-confidence driver": COLORS["none"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def labelize(value: object) -> str:
    text = str(value)
    return (
        text.replace("_", " ")
        .replace("locked local validation", "Local validation")
        .replace("locked public external", "Public external")
        .replace("public training model development", "Public training")
        .replace("public BV-BRC IPM candidate", "Public IPM candidates")
    )


def read_panel(fig: int, filename: str) -> pd.DataFrame:
    folder = DATA / f"Main_Figure_{fig}"
    path = folder / filename
    if not path.exists():
        parts = filename.split("_", 2)
        if len(parts) >= 2:
            matches = sorted(folder.glob(f"{parts[0]}_{parts[1]}_*.xlsx"))
            if len(matches) == 1:
                path = matches[0]
    return pd.read_excel(path)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if COMBINED_PDF is not None:
        COMBINED_PDF.savefig(fig, bbox_inches="tight")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_flow(ax: plt.Axes) -> None:
    nodes = read_panel(1, "Fig_1A_Study_design_flow.xlsx")
    edges = read_panel(1, "Fig_1A_edges_Study_design_flow_edges.xlsx")
    ax.set_axis_off()
    xpos = {
        "public_training": 0.08,
        "unified_model": 0.38,
        "locked_public_external": 0.68,
        "locked_local_validation": 0.68,
        "prediction_error_table": 0.38,
        "mechanism_predictability": 0.68,
    }
    ypos = {
        "public_training": 0.72,
        "unified_model": 0.72,
        "locked_public_external": 0.86,
        "locked_local_validation": 0.58,
        "prediction_error_table": 0.32,
        "mechanism_predictability": 0.32,
    }
    for _, row in nodes.iterrows():
        node = row["node_id"]
        x, y = xpos.get(node, 0.1), ypos.get(node, 0.5)
        face = "#F5FAFA" if row["stage"] in {"training", "model"} else "#FBF7F2"
        rect = Rectangle((x, y - 0.065), 0.23, 0.13, facecolor=face, edgecolor=COLORS["dark"], linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.115, y, str(row["label"]).replace(" genomes", "\ngenomes").replace(" mechanism", "\nmechanism"), ha="center", va="center", fontsize=7.5)
    for _, row in edges.iterrows():
        s, t = row["source"], row["target"]
        x1, y1 = xpos.get(s, 0.1) + 0.23, ypos.get(s, 0.5)
        x2, y2 = xpos.get(t, 0.1), ypos.get(t, 0.5)
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, color=COLORS["neutral"], linewidth=0.8)
        ax.add_patch(arrow)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, str(row["edge_label"]), ha="center", va="center", fontsize=6.8, color=COLORS["neutral"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 1)
    ax.set_title("Locked public-training model and evaluation flow", loc="left", pad=8)


def figure1() -> Path:
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    draw_flow(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    df = read_panel(1, "Fig_1B_Dataset_readiness.xlsx")
    if "cohort_order" in df.columns:
        df = df.sort_values("cohort_order")
    y = np.arange(len(df))
    metrics = [
        ("isolates", "genomes/isolates", "#444444"),
        ("assemblies_present", "assemblies/features", "#B8B8B8"),
        ("ipm_mic_and_sir", "IPM MIC/SIR", COLORS["IPM"]),
        ("mem_mic_and_sir", "MEM MIC/SIR", COLORS["MEM"]),
    ]
    offsets = np.linspace(-0.27, 0.27, len(metrics))
    height = 0.16
    for offset, (col, label, color) in zip(offsets, metrics):
        bars = ax.barh(y + offset, df[col], height=height, color=color, edgecolor="white", linewidth=0.35, label=label)
        for bar in bars:
            value = int(round(bar.get_width()))
            if value <= 0:
                continue
            ax.text(bar.get_width() + max(df["isolates"].max() * 0.01, 8), bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=6.5, color=COLORS["neutral"])
    labels = df["display_label"] if "display_label" in df.columns else df["data_origin"]
    ax.set_yticks(y, [labelize(v) for v in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Genome / phenotype count")
    ax.set_title("Model-development and locked-evaluation cohorts", loc="left")
    clean_axes(ax, "x")
    ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    df = read_panel(1, "Fig_1C_Locked_model_performance.xlsx")
    metrics = [
        ("within_1_dilution_pct", "EA ±1"),
        ("categorical_agreement_pct", "CA"),
        ("very_major_error_rate_pct", "VME"),
    ]
    cohorts = list(df["cohort"].drop_duplicates())
    x = np.arange(len(cohorts))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for m_idx, (col, lab) in enumerate(metrics):
        for d_idx, drug in enumerate(["IPM", "MEM"]):
            vals = []
            for cohort in cohorts:
                sub = df[(df["cohort"] == cohort) & (df["drug"] == drug)]
                vals.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            pos = x + (m_idx - 1) * 0.24 + (d_idx - 0.5) * width
            ax.bar(pos, vals, width=width, color=COLORS[drug], alpha=0.95 if m_idx < 2 else 0.55, label=f"{drug} {lab}")
    ax.set_xticks(x, [labelize(v) for v in cohorts], rotation=10, ha="right")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Locked model performance", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    df = read_panel(1, "Fig_1D_Local_IPM_MEM_phenotype_groups.xlsx").sort_values("pair_group_order")
    bars = ax.barh(np.arange(len(df)), df["isolates"], color="#9FBBC1", edgecolor=COLORS["dark"])
    ax.set_yticks(np.arange(len(df)), [labelize(v).replace(" or I", "/I") for v in df["ipm_mem_pair_group"]])
    ax.invert_yaxis()
    ax.set_xlabel("Local paired isolates")
    ax.set_title("Paired IPM/MEM phenotype composition", loc="left")
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=7)
    clean_axes(ax, "x")
    fig.suptitle("Figure 1. Unified IPM/MEM modelling and evaluation cohorts", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return save(fig, f"Figure_1_unified_model_and_cohorts_{DATE}.pdf")


def figure2() -> Path:
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.15], hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    ax.set_axis_off()
    boxes = [
        ("High-confidence driver classes", 0.03, 0.72, 0.92, 0.18, "#F6FBFA"),
        ("OprD severe loss\nAcquired carbapenemase\nAmpC-axis disruptive\nEfflux-driver disruptive", 0.03, 0.38, 0.42, 0.25, "#FFFFFF"),
        ("Priority subtype assignment\nOprD-loss -> composite -> AmpC-axis -> unresolved", 0.53, 0.38, 0.42, 0.25, "#FFFFFF"),
        ("Purpose: separate predictable MIC phenotypes from mechanism-dependent failure modes", 0.03, 0.09, 0.92, 0.18, "#FBF7F2"),
    ]
    for text, x, y, w, h, face in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=COLORS["dark"], linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)
    ax.add_patch(FancyArrowPatch((0.45, 0.505), (0.53, 0.505), arrowstyle="-|>", mutation_scale=9, color=COLORS["neutral"]))
    ax.set_title("Mechanism framework", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    df = read_panel(2, "Fig_2B_High_confidence_mechanism_counts.xlsx").sort_values("high_conf_order")
    y = np.arange(len(df))
    colors = [SUBTYPE_COLORS.get(v, COLORS["neutral"]) for v in df["mechanism_subtype_high_confidence"]]
    ax.barh(y, df["isolates"], color=colors, edgecolor=COLORS["dark"])
    ax.set_yticks(y, [labelize(v) for v in df["mechanism_subtype_high_confidence"]])
    ax.invert_yaxis()
    ax.set_xlabel("Isolates")
    ax.set_title("High-confidence subtype counts", loc="left")
    for yy, val in zip(y, df["isolates"]):
        ax.text(val + 1, yy, str(int(val)), va="center", fontsize=7)
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    df = read_panel(2, "Fig_2C_High_confidence_mechanism_evidence_matrix.xlsx").sort_values(["high_conf_order", "genome_id"])
    flags = [
        "oprd_deep_disruptive",
        "acquired_carbapenemase_strict",
        "ampc_core_driver_disruptive_any",
        "efflux_strict_driver_disruptive_any",
    ]
    mat = df[flags].astype(int).to_numpy().T
    for yy in range(mat.shape[0]):
        for xx in range(mat.shape[1]):
            face = COLORS["warning"] if mat[yy, xx] else "#F2F4F6"
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=0.08))
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_yticks(np.arange(len(flags)), ["OprD loss", "Carbapenemase", "AmpC axis", "Efflux driver"])
    ax.set_xticks([])
    boundaries = df.groupby("mechanism_subtype_high_confidence").size().cumsum().to_numpy()[:-1]
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", linewidth=1.2)
    ax.set_xlabel("Local isolates sorted by high-confidence subtype")
    ax.set_title("Binary mechanism evidence matrix", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    df = read_panel(2, "Fig_2D_Target_gene_PAO1_relative_variation_burden.xlsx").sort_values(
        "isolates_with_non_syn_or_regulatory_pct", ascending=True
    )
    y = np.arange(len(df))
    colors = np.where(df["strict_driver"], COLORS["warning"], "#9FBBC1")
    ax.barh(y, df["isolates_with_non_syn_or_regulatory_pct"], color=colors, edgecolor=COLORS["dark"], linewidth=0.5)
    ax.set_yticks(y, df["gene"])
    ax.set_xlabel("Isolates with non-synonymous/regulatory variants (%)")
    ax.set_xlim(0, 105)
    ax.set_title("PAO1-relative target-gene variation burden", loc="left")
    clean_axes(ax, "x")
    fig.suptitle("Figure 2. High-confidence mechanism annotation for IPM/MEM interpretability", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return save(fig, f"Figure_2_high_confidence_mechanism_framework_{DATE}.pdf")


def grouped_bar(ax: plt.Axes, df: pd.DataFrame, value: str, ylabel: str, title: str, pct: bool = False) -> None:
    df = df.sort_values(["high_conf_order", "drug_order"])
    subtypes = list(df["mechanism_subtype_high_confidence"].drop_duplicates())
    x = np.arange(len(subtypes))
    width = 0.34
    for i, drug in enumerate(["IPM", "MEM"]):
        vals = []
        ns = []
        for subtype in subtypes:
            row = df[(df["mechanism_subtype_high_confidence"] == subtype) & (df["drug"] == drug)]
            vals.append(float(row[value].iloc[0]) if len(row) else np.nan)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)
        if pct:
            vals = [v * 100 for v in vals]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[drug], label=drug)
        for bar, n in zip(bars, ns):
            if not math.isnan(bar.get_height()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (2 if pct else 0.08), f"n={n}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    ax.set_xticks(x, [labelize(v) for v in subtypes], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)


def figure3() -> Path:
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.35)
    metrics = read_panel(3, "Fig_3A_B_High_confidence_predictability_metrics.xlsx")
    points = read_panel(3, "Fig_3C_D_High_confidence_error_points.xlsx")

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    grouped_bar(ax, metrics, "within_1_dilution", "Essential agreement ±1 (%)", "Mechanism-stratified MIC agreement", pct=True)
    ax.set_ylim(0, 112)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    grouped_bar(ax, metrics, "mae_log2", "MAE (log2 dilution)", "Mechanism-stratified absolute error")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    subtypes = metrics.sort_values("high_conf_order")["mechanism_subtype_high_confidence"].drop_duplicates().tolist()
    positions = {s: i for i, s in enumerate(subtypes)}
    rng = np.random.default_rng(4)
    for drug in ["IPM", "MEM"]:
        sub = points[points["drug"] == drug]
        xs = sub["mechanism_subtype_high_confidence"].map(positions).astype(float).to_numpy()
        xs = xs + (-0.18 if drug == "IPM" else 0.18) + rng.normal(0, 0.045, len(xs))
        ax.scatter(xs, sub["signed_log2_error"], s=16, alpha=0.72, color=COLORS[drug], edgecolor="white", linewidth=0.25, label=drug)
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axhline(-2, color=COLORS["warning"], linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(subtypes)), [labelize(v) for v in subtypes], rotation=25, ha="right")
    ax.set_ylabel("Signed prediction error (predicted - observed log2 MIC)")
    ax.set_title("Signed error reveals MEM underprediction", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    for drug in ["IPM", "MEM"]:
        sub = points[points["drug"] == drug]
        ax.scatter(sub["actual_mic_log2"], sub["pred_mic_log2"], s=18, alpha=0.68, color=COLORS[drug], edgecolor="white", linewidth=0.25, label=drug)
    lo = min(points["actual_mic_log2"].min(), points["pred_mic_log2"].min()) - 0.5
    hi = max(points["actual_mic_log2"].max(), points["pred_mic_log2"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=COLORS["dark"], linewidth=0.8)
    ax.plot([lo, hi], [lo - 1, hi - 1], color=COLORS["grid"], linewidth=0.7, linestyle="--")
    ax.plot([lo, hi], [lo + 1, hi + 1], color=COLORS["grid"], linewidth=0.7, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed MIC (log2)")
    ax.set_ylabel("Predicted MIC (log2)")
    ax.set_title("Observed versus predicted MIC", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)
    fig.suptitle("Figure 3. Mechanism-dependent predictability of IPM and MEM MICs", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return save(fig, f"Figure_3_mechanism_dependent_predictability_{DATE}.pdf")


def figure4() -> Path:
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.42)
    paired = read_panel(4, "Fig_4A_Paired_high_confidence_IPM_MEM_error_points.xlsx")
    summary = read_panel(4, "Fig_4B_Paired_high_confidence_error_summary.xlsx").sort_values("high_conf_order")

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    for subtype, sub in paired.groupby("mechanism_subtype_high_confidence"):
        ax.scatter(sub["abs_log2_error_IPM"], sub["abs_log2_error_MEM"], s=22, color=SUBTYPE_COLORS.get(subtype, COLORS["neutral"]), alpha=0.78, edgecolor="white", linewidth=0.25, label=labelize(subtype))
    lim = max(paired["abs_log2_error_IPM"].max(), paired["abs_log2_error_MEM"].max()) + 0.5
    ax.plot([0, lim], [0, lim], color=COLORS["dark"], linewidth=0.8)
    ax.set_xlim(-0.1, lim)
    ax.set_ylim(-0.1, lim)
    ax.set_xlabel("IPM abs error (log2)")
    ax.set_ylabel("MEM abs error (log2)")
    ax.set_title("Paired isolate errors", loc="left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    y = np.arange(len(summary))
    ax.barh(y, summary["MEM_worse_fraction"] * 100, color=[SUBTYPE_COLORS.get(v, COLORS["neutral"]) for v in summary["mechanism_subtype_high_confidence"]], edgecolor=COLORS["dark"])
    ax.set_yticks(y, [labelize(v) for v in summary["mechanism_subtype_high_confidence"]])
    ax.invert_yaxis()
    ax.set_xlabel("MEM worse than IPM (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Direction of paired error", loc="left")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "C")
    ax.barh(y, summary["mean_MEM_minus_IPM_abs_error"], color=[SUBTYPE_COLORS.get(v, COLORS["neutral"]) for v in summary["mechanism_subtype_high_confidence"]], edgecolor=COLORS["dark"])
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set_xlabel("Mean MEM-IPM abs error")
    ax.set_title("Magnitude of paired gap", loc="left")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "D")
    st = read_panel(4, "Fig_4C_Local_ST_counts.xlsx").sort_values("isolates", ascending=False).head(12)
    ax.barh(np.arange(len(st)), st["isolates"], color="#9FBBC1", edgecolor=COLORS["dark"])
    ax.set_yticks(np.arange(len(st)), st["ST_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Isolates")
    ax.set_title("Most common local STs", loc="left")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "E")
    loo = read_panel(4, "Fig_4D_Leave_one_ST_out_sensitivity.xlsx")
    ax.barh(np.arange(len(loo)), loo["remaining_MEM_worse_fraction"] * 100, color="#B6C8A9", edgecolor=COLORS["dark"])
    ax.set_yticks(np.arange(len(loo)), ["Exclude " + str(v) for v in loo["excluded_ST_label"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Remaining MEM-worse fraction (%)")
    ax.set_title("Leave-one-ST-out sensitivity", loc="left")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "F")
    models = read_panel(4, "Fig_4E_ST_adjusted_models.xlsx")
    labels = ["Mechanism subtype", "ST group", "Mechanism + ST"]
    ax.bar(np.arange(len(models)), models["r_squared"], color=[COLORS["oprD"], COLORS["none"], COLORS["composite"]], edgecolor=COLORS["dark"])
    ax.set_xticks(np.arange(len(models)), labels, rotation=25, ha="right")
    ax.set_ylabel("R-squared")
    ax.set_title("Mechanism explains more than sparse ST", loc="left")
    clean_axes(ax)
    fig.suptitle("Figure 4. Paired IPM/MEM error structure and ST sensitivity", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return save(fig, f"Figure_4_paired_errors_and_ST_sensitivity_{DATE}.pdf")


def figure5() -> Path:
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.62, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    rules = read_panel(5, "Fig_5A_Clinical_warning_rule_table.xlsx").sort_values(["rule_order", "drug_order"])
    table_df = rules[["drug", "mechanism_subtype_high_confidence", "clinical_rule", "n", "within_1_dilution", "very_major_error_rate"]].copy()
    table_df["EA"] = (table_df["within_1_dilution"] * 100).round(0).astype(int).astype(str) + "%"
    table_df["VME"] = (table_df["very_major_error_rate"] * 100).round(0).astype(int).astype(str) + "%"
    table_df = table_df[["drug", "mechanism_subtype_high_confidence", "clinical_rule", "n", "EA", "VME"]]
    ax.set_axis_off()
    cell_text = []
    for _, r in table_df.iterrows():
        cell_text.append([r["drug"], labelize(r["mechanism_subtype_high_confidence"])[:22], labelize(r["clinical_rule"])[:24], int(r["n"]), r["EA"], r["VME"]])
    tab = ax.table(cellText=cell_text, colLabels=["Drug", "Subtype", "Rule", "n", "EA", "VME"], cellLoc="left", colLoc="left", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(5.7)
    tab.scale(1, 1.2)
    for (row, col), cell in tab.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#EDF2F4")
            cell.set_text_props(weight="bold")
    ax.set_title("Clinical interpretation strata", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    perf = read_panel(5, "Fig_5B_Rule_level_performance.xlsx").sort_values("rule_order")
    y = np.arange(len(perf))
    colors = [COLORS["trust"] if "trust" in r else COLORS["warning"] if "warning" in r else COLORS["explore"] for r in perf["clinical_rule"]]
    ax.barh(y, perf["within_1_dilution"] * 100, color=colors, alpha=0.95, edgecolor=COLORS["dark"], label="EA ±1")
    ax.scatter(perf["very_major_error_rate"] * 100, y, color=COLORS["warning"], s=28, zorder=3, label="VME")
    ax.set_yticks(y, [f"{d} {labelize(r)}" for d, r in zip(perf["drug"], perf["clinical_rule"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent")
    ax.set_title("Rule-level performance", loc="left")
    clean_axes(ax, "x")
    ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    enr = read_panel(5, "Fig_5C_MEM_large_underprediction_enrichment.xlsx")
    flags = enr["flag"].tolist()
    x = np.arange(len(flags))
    width = 0.35
    ax.bar(
        x - width / 2,
        enr["flagged_large_underprediction_gt2_rate"] * 100,
        width,
        color=COLORS["warning"],
        label="Flagged large underprediction",
    )
    ax.bar(
        x + width / 2,
        enr["unflagged_large_underprediction_gt2_rate"] * 100,
        width,
        color="#BFC7CF",
        label="Unflagged large underprediction",
    )
    ax.set_xticks(x, flags, rotation=25, ha="right")
    ax.set_ylabel("MEM large underprediction (%)")
    ax.set_title("Mechanism flags enrich MEM underprediction", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    bp = read_panel(5, "Fig_5D_Breakpoint_zone_errors.xlsx")
    bp = bp[bp["breakpoint_zone"].astype(str).str.contains("far_NS")].sort_values(["high_conf_order", "drug_order"])
    subtypes = list(bp["mechanism_subtype_high_confidence"].drop_duplicates())
    x = np.arange(len(subtypes))
    width = 0.34
    for i, drug in enumerate(["IPM", "MEM"]):
        vals = []
        for subtype in subtypes:
            row = bp[(bp["drug"] == drug) & (bp["mechanism_subtype_high_confidence"] == subtype)]
            vals.append(float(row["vme_rate"].iloc[0]) * 100 if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[drug], label=drug)
    ax.set_xticks(x, [labelize(v) for v in subtypes], rotation=25, ha="right")
    ax.set_ylabel("VME rate in far-NS zone (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Errors far from breakpoint", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "E")
    tax = read_panel(5, "Fig_5E_Error_taxonomy.xlsx")
    danger = tax[tax["error_taxonomy"] == "dangerous_false_susceptible_far_from_breakpoint"].copy()
    danger = danger.sort_values(["high_conf_order", "drug_order"])
    ylabels = [f"{r.drug} {labelize(r.mechanism_subtype_high_confidence)}" for r in danger.itertuples()]
    ax.barh(np.arange(len(danger)), danger["n"], color=[COLORS["MEM"] if d == "MEM" else COLORS["IPM"] for d in danger["drug"]], edgecolor=COLORS["dark"])
    ax.set_yticks(np.arange(len(danger)), ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Dangerous false-susceptible cases")
    ax.set_title("Error taxonomy: clinical safety failures", loc="left")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "F")
    gate = read_panel(5, "Fig_5F_optional_Safety_gate_evaluation.xlsx")
    gate = gate[gate["drug"] == "MEM"].copy()
    x = np.arange(len(gate))
    ax.bar(x - 0.18, gate["coverage"] * 100, 0.36, color="#9FBBC1", label="Coverage")
    ax.bar(x + 0.18, gate["released_vme_rate"] * 100, 0.36, color=COLORS["warning"], label="Released VME")
    ax.set_xticks(x, [labelize(v).replace(" gate", "") for v in gate["gate_scenario"]], rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Prototype safety gate trade-off for MEM", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False)

    fig.suptitle("Figure 5. Clinical safety interpretation of mechanism-dependent prediction errors", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return save(fig, f"Figure_5_clinical_safety_rules_{DATE}.pdf")


def main() -> None:
    global COMBINED_PDF
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    merged = OUT / f"IPM-GPT_main_figures_combined_preview_{DATE}.pdf"
    with PdfPages(merged) as pdf:
        COMBINED_PDF = pdf
        paths = [figure1(), figure2(), figure3(), figure4(), figure5()]
        index_fig = plt.figure(figsize=(7.2, 5.0))
        ax = index_fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.02, 0.95, "IPM-GPT main figure editable PDF draft set", fontsize=13, weight="bold", va="top")
        y = 0.82
        for i, path in enumerate(paths, start=1):
            ax.text(0.05, y, f"Figure {i}: {path.name}", fontsize=9, va="top")
            y -= 0.11
        ax.text(0.02, 0.12, "Use the individual PDFs for editing in Illustrator, Inkscape, Affinity Designer, or PowerPoint PDF import.", fontsize=8)
        pdf.savefig(index_fig, bbox_inches="tight")
        plt.close(index_fig)
        COMBINED_PDF = None
    manifest = OUT / f"main_figure_pdf_manifest_{DATE}.csv"
    pd.DataFrame(
        [{"figure": f"Figure {i}", "pdf_file": p.name, "absolute_path": str(p)} for i, p in enumerate(paths, 1)]
        + [{"figure": "Combined multi-page preview", "pdf_file": merged.name, "absolute_path": str(merged)}]
    ).to_csv(manifest, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(paths)} editable PDFs to {OUT}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
