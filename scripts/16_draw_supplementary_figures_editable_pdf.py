from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "13_figure_excel_data_package"
OUT = ROOT / "\u6295\u7a3f" / "supplementary_figures_editable_pdf"
DATE = "2026-06-07"

IPM = "#0077B6"
MEM = "#D95F02"
DARK = "#2F4154"
TEXT = "#333333"
GRID = "#E6E6E6"
LIGHT = "#F5F5F7"
GREY = "#BFBFBF"
MIDGREY = "#777777"
WARNING = "#D95F02"
RED = "#C43C39"
GREEN = "#1B9E77"
YELLOW = "#F0B000"
PURPLE = "#7C3AED"

SUBTYPE_COLORS = {
    "OprD-loss": "#1B9E77",
    "Composite": "#7C3AED",
    "High-confidence composite": "#7C3AED",
    "AmpC-axis disruptive": "#C77D1A",
    "Efflux-associated genotype": "#4C78A8",
    "No strict mechanism": "#9E9E9E",
    "No high-confidence driver": "#9E9E9E",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#404040",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def panel_file(folder: str, prefix: str) -> Path:
    matches = sorted((DATA / folder).glob(f"{prefix}_*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No panel file for {folder}/{prefix}")
    return matches[0]


def read_panel(folder: str, prefix: str) -> pd.DataFrame:
    return pd.read_excel(panel_file(folder, prefix))


def title(ax: plt.Axes, main: str, subtitle: str | None = None) -> None:
    ax.set_title("")
    ax.text(
        0,
        1.13,
        main,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=DARK,
    )
    if subtitle:
        ax.text(
            0,
            1.065,
            "\n".join(textwrap.wrap(subtitle, width=92)),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=MIDGREY,
            style="italic",
        )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.02, 0.012, "\n".join(textwrap.wrap(text, width=155)), ha="left", va="bottom", fontsize=8.5, color=MIDGREY, style="italic")


def clean(ax: plt.Axes, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def short(text: object) -> str:
    value = str(text)
    value = value.replace("High-confidence composite", "HC composite")
    value = value.replace("No high-confidence driver", "No HC driver")
    value = value.replace("Efflux-associated genotype", "Efflux genotype")
    value = value.replace("No strict mechanism", "No strict mech")
    value = value.replace("IPM-R/MEM-S_or_I", "IPM-R/MEM-S/I")
    value = value.replace("_", " ")
    return value


def wrapped(values: list[object], width: int = 14) -> list[str]:
    return ["\n".join(textwrap.wrap(str(v), width=width, break_long_words=False)) for v in values]


def add_y_bands(ax: plt.Axes, n: int, color: str = "#F6F6F8") -> None:
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=color, zorder=0)


def save(fig: plt.Figure, filename: str, pdf: PdfPages | None, paths: list[Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    try:
        fig.tight_layout(rect=[0.01, 0.08, 0.99, 0.88])
    except Exception:
        fig.subplots_adjust(top=0.80, bottom=0.22)
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)


def table_ax(ax: plt.Axes, frame: pd.DataFrame, col_widths: list[float] | None = None, fontsize: float = 8.5) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#EEF2F6")
            cell.set_text_props(weight="bold", color=DARK)


def s1a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S1", "S1A")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Drug": row["drug"],
                "Breakpoint log2": row["breakpoint_log2"],
                "Gate": row["gate_model_name"],
                "Features": row["gate_feature_set"],
                "Threshold": row["threshold"],
                "Approach": row["approach"],
                "Cap": row["cap_policy"],
            }
        )
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    title(ax, "Unified model policy", "same model class, feature policy, threshold and MIC snapping for IPM and MEM")
    table_ax(ax, pd.DataFrame(rows), fontsize=9)
    footer(fig, "Public-training-only unified policy; locked public external and local validation were not used for training or policy selection.")
    save(fig, "S1A_unified_model_policy.pdf", pdf, paths)


def s1b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S1", "S1B")
    metrics = [
        ("mic_ea_pm1_apparent", "EA +/-1"),
        ("mic_exact_pm0_5_apparent", "Exact +/-0.5"),
        ("mic_sns_ca", "CA"),
        ("mic_sns_balanced_accuracy", "Balanced acc."),
        ("mic_sns_vme", "VME"),
        ("mic_sns_me", "ME"),
        ("gate_auc_apparent", "Gate AUC"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(metrics))
    width = 0.34
    for i, drug in enumerate(["IPM", "MEM"]):
        sub = df[df["drug"].eq(drug)].iloc[0]
        values = [float(sub[m]) * 100 for m, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=IPM if drug == "IPM" else MEM, label=drug)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x, [lab for _, lab in metrics], rotation=25, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    title(ax, "Public-training apparent performance", "smoke-check only; not used as locked validation evidence")
    clean(ax)
    ax.legend(frameon=False, loc="upper right")
    footer(fig, "Training apparent n: IPM 312, MEM 3,065; apparent metrics are shown only to document model behavior during development.")
    save(fig, "S1B_public_training_apparent_performance.pdf", pdf, paths)


def s2a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S2", "S2A").sort_values("subtype_order")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(df))
    add_y_bands(ax, len(df))
    colors = [SUBTYPE_COLORS.get(v, MIDGREY) for v in df["mechanism_subtype_strict"]]
    bars = ax.barh(y, df["isolates"], color=colors)
    ax.set_yticks(y, [short(v) for v in df["mechanism_subtype_strict"]])
    ax.invert_yaxis()
    ax.set_xlabel("Isolates")
    title(ax, "First-pass strict mechanism subtype counts", "strict rules before high-confidence refinement")
    clean(ax, "x")
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=9)
    footer(fig, "Strict first-pass labels are shown as auditable intermediate annotations.")
    save(fig, "S2A_first_pass_strict_subtype_counts.pdf", pdf, paths)


def s2b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S2", "S2B").sort_values("pair_group_order")
    subtypes = [c for c in df.columns if c not in {"ipm_mem_pair_group", "pair_group_order"}]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for subtype in subtypes:
        vals = df[subtype].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=short(subtype), color=SUBTYPE_COLORS.get(subtype, MIDGREY))
        bottom += vals
    ax.set_xticks(x, wrapped([short(v) for v in df["ipm_mem_pair_group"]], width=12), rotation=0, ha="center")
    ax.set_ylabel("Isolates")
    title(ax, "Phenotype groups by strict subtype", "paired local IPM/MEM phenotype categories")
    clean(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    footer(fig, "Only local isolates contribute paired IPM/MEM phenotype groups.")
    save(fig, "S2B_pair_group_by_strict_subtype.pdf", pdf, paths)


def draw_binary_matrix(ax: plt.Axes, df: pd.DataFrame, flags: list[tuple[str, str]], order_col: str, title_text: str, subtitle: str) -> None:
    df = df.sort_values([order_col, "genome_id"]).reset_index(drop=True)
    for yy, (col, _lab) in enumerate(flags):
        vals = df[col].fillna(False).astype(bool).to_numpy()
        for xx, val in enumerate(vals):
            face = RED if val else "#F3F3F3"
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=0.08))
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(len(flags) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(flags)), [lab for _, lab in flags])
    for b in df.groupby(order_col).size().cumsum().to_numpy()[:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, title_text, subtitle)


def s2c(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S2", "S2C")
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    flags = [
        ("oprd_severe_loss", "OprD severe loss"),
        ("acquired_carbapenemase_strict", "Carbapenemase"),
        ("ampc_associated_strict", "AmpC strict"),
        ("efflux_regulator_strict", "Efflux regulator"),
        ("pdc_present_broad", "PDC broad"),
        ("efflux_regulator_broad", "Efflux broad"),
    ]
    draw_binary_matrix(ax, df, flags, "subtype_order", "Strict mechanism evidence matrix", "147 local isolates sorted by first-pass subtype")
    footer(fig, "Red tiles indicate positive evidence; broad-context signals are shown for transparency but do not define final high-confidence labels.")
    save(fig, "S2C_strict_evidence_matrix.pdf", pdf, paths)


def s3a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S3", "S3A")
    severity_order = ["none", "synonymous_only", "missense", "promoter_or_upstream", "inframe_indel", "coding_indel_uncertain", "disruptive"]
    genes = (
        df[["axis", "gene"]]
        .drop_duplicates()
        .assign(axis_order=lambda x: x["axis"].map({"oprd": 1, "efflux": 2, "ampc_axis": 3, "ampc_axis_context": 4}).fillna(9))
        .sort_values(["axis_order", "gene"])["gene"]
        .tolist()
    )
    mat = []
    for gene in genes:
        sub = df[df["gene"].eq(gene)]
        counts = sub["severity"].value_counts()
        mat.append([counts.get(s, 0) / len(sub) * 100 for s in severity_order])
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("grey_red", ["#F7F7F7", "#F0B000", RED])
    arr = np.array(mat)
    for yy in range(arr.shape[0]):
        for xx in range(arr.shape[1]):
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=cmap(arr[yy, xx] / 100), edgecolor="white", linewidth=0.7))
            if arr[yy, xx] >= 20:
                ax.text(xx, yy, f"{arr[yy, xx]:.0f}", ha="center", va="center", fontsize=7, color=TEXT)
    ax.set_xlim(-0.5, len(severity_order) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    pretty_severity = {
        "none": "none",
        "synonymous_only": "syn.\nonly",
        "missense": "missense",
        "promoter_or_upstream": "promoter /\nupstream",
        "inframe_indel": "inframe\nindel",
        "coding_indel_uncertain": "coding indel\nuncertain",
        "disruptive": "disruptive",
    }
    ax.set_xticks(np.arange(len(severity_order)), [pretty_severity[v] for v in severity_order], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(genes)), genes)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "Target-gene variant severity landscape", "cell values show percent of local isolates per gene and severity class")
    footer(fig, "High PAO1-relative missense burden is common across several genes; disruptive calls are treated more conservatively.")
    save(fig, "S3A_target_gene_variant_severity_landscape.pdf", pdf, paths)


def s3b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S3", "S3B")
    effect_order = ["synonymous", "missense", "promoter_or_upstream", "disruptive", "coding_indel_uncertain", "inframe_indel", "other_annotated"]
    counts = df.groupby(["axis", "effect_class"]).size().reset_index(name="n")
    pivot = counts.pivot(index="axis", columns="effect_class", values="n").fillna(0)
    pivot = pivot.reindex(["oprd", "efflux", "ampc_axis", "ampc_axis_context"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    palette = {
        "synonymous": GREY,
        "missense": IPM,
        "promoter_or_upstream": YELLOW,
        "disruptive": RED,
        "coding_indel_uncertain": PURPLE,
        "inframe_indel": GREEN,
        "other_annotated": MIDGREY,
    }
    for effect in effect_order:
        vals = pivot[effect].to_numpy() if effect in pivot.columns else np.zeros(len(pivot))
        ax.bar(x, vals, bottom=bottom, color=palette.get(effect, MIDGREY), label=short(effect))
        bottom += vals
    ax.set_yscale("log")
    ax.set_xticks(x, wrapped([short(v) for v in pivot.index], width=12), rotation=0, ha="center")
    ax.set_ylabel("Variant rows (log scale)")
    title(ax, "Target-region variant-row composition", "effect-class distribution across mechanism axes")
    clean(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    footer(fig, "Counts are variant rows, not isolate counts; the full variant table is retained in the Excel package.")
    save(fig, "S3B_target_region_variant_row_composition.pdf", pdf, paths)


def strip_error_plot(df: pd.DataFrame, subtype_col: str, order_col: str, main: str, filename: str, pdf: PdfPages | None, paths: list[Path]) -> None:
    subtypes = df[[subtype_col, order_col]].drop_duplicates().sort_values(order_col)[subtype_col].tolist()
    pos = {s: i for i, s in enumerate(subtypes)}
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for i in range(len(subtypes)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#F6F6F8", zorder=0)
    for drug, color, offset in [("IPM", IPM, -0.18), ("MEM", MEM, 0.18)]:
        sub = df[df["drug"].eq(drug)].copy()
        xs = sub[subtype_col].map(pos).to_numpy(dtype=float) + offset + rng.normal(0, 0.045, len(sub))
        ax.scatter(xs, sub["signed_log2_error"], s=24, color=color, alpha=0.75, edgecolor="white", linewidth=0.4, label=drug)
    ax.axhline(0, color=DARK, linewidth=1.0)
    ax.axhline(-2, color=RED, linestyle="--", linewidth=0.9)
    ax.set_xticks(np.arange(len(subtypes)), wrapped([short(v) for v in subtypes], width=13), rotation=0, ha="center")
    ax.set_ylabel("Signed error (predicted - observed log2 MIC)")
    title(ax, main, "negative values indicate underprediction")
    clean(ax)
    ax.legend(frameon=False, loc="lower left")
    footer(fig, "Dashed line marks large underprediction greater than two log2 dilutions.")
    save(fig, filename, pdf, paths)


def s4a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S4", "S4A")
    strip_error_plot(df, "mechanism_subtype_strict", "subtype_order", "Strict subtype point-level prediction errors", "S4A_strict_subtype_error_points.pdf", pdf, paths)


def s4b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S4", "S4B")
    strip_error_plot(df, "mechanism_subtype_high_confidence", "high_conf_order", "High-confidence subtype point-level prediction errors", "S4B_high_confidence_error_points.pdf", pdf, paths)


def s5a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S5", "S5A").sort_values("isolates", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    add_y_bands(ax, len(df))
    colors = np.where(df["mlst_call_status"].eq("ST_exact_profile"), IPM, GREY)
    bars = ax.barh(np.arange(len(df)), df["isolates"], color=colors)
    ax.set_yticks(np.arange(len(df)), df["ST_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Isolates")
    title(ax, "Top local MLST groups", "largest 20 ST labels or unassigned profiles")
    clean(ax, "x")
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())}", va="center", fontsize=8)
    footer(fig, "Exact PubMLST profiles are blue; novel/unprofiled or incomplete calls are grey.")
    save(fig, "S5A_local_ST_counts.pdf", pdf, paths)


def s5b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S5", "S5B")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(df))
    add_y_bands(ax, len(df))
    ax.scatter(df["remaining_MEM_worse_fraction"] * 100, y, s=100, color=MEM)
    for yy, row in zip(y, df.itertuples()):
        ax.text(row.remaining_MEM_worse_fraction * 100 + 1.5, yy, f"n={row.remaining_n}; p={row.remaining_paired_wilcoxon_abs_error_p:.1e}", va="center", fontsize=9)
    ax.set_yticks(y, ["Exclude " + str(v) for v in df["excluded_ST_label"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Remaining MEM-worse fraction (%)")
    title(ax, "Leave-one-ST-out sensitivity", "paired IPM/MEM error signal after excluding common ST groups")
    clean(ax, "x")
    footer(fig, "MEM-worse fraction remains high after excluding each tested ST group.")
    save(fig, "S5B_leave_one_ST_out_sensitivity.pdf", pdf, paths)


def s5c(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S5", "S5C")
    labels = ["Mechanism", "ST group", "Mechanism + ST"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(np.arange(len(df)), df["r_squared"], color=[GREEN, GREY, PURPLE])
    ax.set_xticks(np.arange(len(df)), labels, rotation=20, ha="right")
    ax.set_ylabel("R-squared")
    title(ax, "Mechanism vs ST explanatory models", "variance explained for MEM-minus-IPM absolute error")
    clean(ax)
    for bar, p in zip(bars, df["model_f_pvalue"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"p={p:.1e}", ha="center", va="bottom", fontsize=8)
    footer(fig, "Sparse ST grouping explains substantially less variance than mechanism subtype.")
    save(fig, "S5C_ST_adjusted_models.pdf", pdf, paths)


def s6a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S6", "S6A").sort_values(["high_conf_order", "drug_order"])
    subtypes = df["mechanism_subtype_high_confidence"].drop_duplicates().tolist()
    y = np.arange(len(subtypes))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for i, subtype in enumerate(subtypes):
        for drug, color, marker in [("IPM", IPM, "o"), ("MEM", MEM, "o")]:
            row = df[(df["mechanism_subtype_high_confidence"].eq(subtype)) & (df["drug"].eq(drug))]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.plot([row["mean_prob_ns"] * 100, row["observed_ns_rate"] * 100], [i, i], color="#CFCFCF", linewidth=3)
            ax.scatter(row["mean_prob_ns"] * 100, i + (-0.07 if drug == "IPM" else 0.07), color=color, s=55, label=f"{drug} mean prob" if i == 0 else None)
            ax.scatter(row["observed_ns_rate"] * 100, i + (-0.07 if drug == "IPM" else 0.07), facecolor="white", edgecolor=color, s=65, linewidth=1.8, label=f"{drug} observed" if i == 0 else None)
    ax.set_yticks(y, [short(v) for v in subtypes])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Non-susceptible probability / observed rate (%)")
    title(ax, "Subtype-level calibration summary", "filled dots: mean predicted NS probability; open dots: observed NS rate")
    clean(ax, "x")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    footer(fig, "Calibration is shown descriptively by high-confidence subtype and drug.")
    save(fig, "S6A_calibration_summary.pdf", pdf, paths)


def s6b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S6", "S6B")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for drug, color in [("IPM", IPM), ("MEM", MEM)]:
        sub = df[df["drug"].eq(drug)].copy()
        ax.scatter(sub["mean_prob_ns"] * 100, sub["observed_ns_rate"] * 100, s=np.maximum(sub["n"], 1) * 22, color=color, alpha=0.7, edgecolor="white", linewidth=0.5, label=drug)
    ax.plot([0, 100], [0, 100], color=DARK, linewidth=1)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Mean predicted NS probability (%)")
    ax.set_ylabel("Observed NS rate (%)")
    title(ax, "Calibration by probability bin", "point size reflects bin sample size")
    clean(ax)
    ax.legend(frameon=False, loc="lower right")
    footer(fig, "Bins are pooled across high-confidence subtypes where rows were available.")
    save(fig, "S6B_calibration_by_probability_bin.pdf", pdf, paths)


def s6c(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S6", "S6C").sort_values(["high_conf_order", "drug_order"])
    confidence_label = {
        "confident_S": "confident S",
        "low_margin_S": "low-margin S",
    }
    labels = [
        f"{r.drug} | {short(r.mechanism_subtype_high_confidence)} | {confidence_label.get(str(r.susceptible_call_confidence), short(r.susceptible_call_confidence))}"
        for r in df.itertuples()
    ]
    colors = [IPM if d == "IPM" else MEM for d in df["drug"]]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    y = np.arange(len(df))
    add_y_bands(ax, len(df))
    bars = ax.barh(y, df["false_susceptible_rate"] * 100, color=colors)
    ax.set_yticks(y, wrapped(labels, width=42))
    ax.invert_yaxis()
    ax.set_xlabel("False-susceptible rate among predicted susceptible calls (%)")
    ax.set_xlim(0, 105)
    title(ax, "Susceptible-call risk by confidence stratum", "risk among model-predicted susceptible calls")
    clean(ax, "x")
    for bar, n, f in zip(bars, df["n_predicted_susceptible"], df["false_susceptible_n"]):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{int(f)}/{int(n)}", ha="left", va="center", fontsize=8)
    footer(fig, "Fractions above bars show false-susceptible calls over all predicted susceptible calls.")
    save(fig, "S6C_susceptible_call_risk.pdf", pdf, paths)


def s7a(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S7", "S7A")
    scenarios = df["gate_scenario"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(scenarios))
    width = 0.18
    for di, (drug, color) in enumerate([("IPM", IPM), ("MEM", MEM)]):
        sub = df[df["drug"].eq(drug)].set_index("gate_scenario").reindex(scenarios)
        ax.bar(x + (di - 0.5) * width * 2, sub["coverage"] * 100, width, color=color, alpha=0.65, label=f"{drug} coverage")
        ax.scatter(x + (di - 0.5) * width * 2, sub["released_vme_rate"] * 100, color=color, edgecolor="white", s=55, zorder=3, label=f"{drug} released VME")
    scenario_labels = {
        "low_margin_only_gate": "Low-margin\nonly",
        "mechanism_or_low_margin_gate": "Mechanism warning\nor low-margin",
        "mechanism_warning_susceptible_gate": "Mechanism-warning\nsusceptible calls",
        "mem_warning_susceptible_gate": "MEM warning\nsusceptible calls",
        "all_susceptible_calls_withheld": "All susceptible\ncalls withheld",
        "mechanism_warning_or_low_margin_gate": "Mechanism warning\nor low-margin",
        "no_gate": "No gate",
    }
    ax.set_xticks(x, [scenario_labels.get(v, short(v).replace(" gate", "")) for v in scenarios], rotation=0, ha="center")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    title(ax, "Safety gate evaluation", "bars: prediction coverage; dots: standard VME among released true non-susceptible isolates")
    clean(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    footer(fig, "VME is calculated as false-susceptible calls divided by true non-susceptible isolates within released predictions. Mechanism-warning gates reduce false susceptibility at the cost of lower release coverage.")
    save(fig, "S7A_safety_gate_evaluation.pdf", pdf, paths)


def s7b(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S7", "S7B")
    keep = [
        "accurate_MIC_and_category",
        "MIC_error_only",
        "category_error_near_breakpoint",
        "dangerous_false_susceptible_far_from_breakpoint",
        "major_error",
    ]
    df = df[df["error_taxonomy"].isin(keep)].copy()
    groups = df.groupby(["drug", "error_taxonomy"], as_index=False)["n"].sum()
    pivot = groups.pivot(index="drug", columns="error_taxonomy", values="n").fillna(0).reindex(["IPM", "MEM"])
    colors = {
        "accurate_MIC_and_category": GREEN,
        "MIC_error_only": GREY,
        "category_error_near_breakpoint": YELLOW,
        "dangerous_false_susceptible_far_from_breakpoint": RED,
        "major_error": PURPLE,
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for cat in keep:
        vals = pivot[cat].to_numpy() if cat in pivot.columns else np.zeros(len(pivot))
        ax.bar(x, vals, bottom=bottom, color=colors[cat], label=short(cat))
        bottom += vals
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Prediction rows")
    title(ax, "Error taxonomy summary", "aggregate clinical error categories by drug")
    clean(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    footer(fig, "Dangerous false-susceptible far-from-breakpoint errors dominate MEM safety failures.")
    save(fig, "S7B_error_taxonomy_summary.pdf", pdf, paths)


def s7c(pdf: PdfPages | None, paths: list[Path]) -> None:
    df = read_panel("Supplementary_Figure_S7", "S7C")
    counts = df.groupby(["drug", "clinical_rule", "mechanism_subtype_high_confidence"]).size().reset_index(name="n")
    counts["row"] = counts["drug"] + " | " + counts["clinical_rule"].map(short)
    rows = counts["row"].drop_duplicates().tolist()
    cols = counts["mechanism_subtype_high_confidence"].drop_duplicates().tolist()
    mat = counts.pivot(index="row", columns="mechanism_subtype_high_confidence", values="n").fillna(0).reindex(index=rows, columns=cols).fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    maxv = max(mat.to_numpy().max(), 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("warn", ["#F7F7F7", "#F0B000", RED])
    for yy in range(mat.shape[0]):
        for xx in range(mat.shape[1]):
            value = mat.iloc[yy, xx]
            ax.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor=cmap(value / maxv), edgecolor="white", linewidth=0.8))
            if value > 0:
                ax.text(xx, yy, str(int(value)), ha="center", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(mat.shape[1]), [short(c) for c in mat.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]), [short(r) for r in mat.index])
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "False-susceptible cases with mechanism context", "cell values show VME case counts")
    footer(fig, "Rows combine drug and clinical rule; columns show high-confidence mechanism subtype.")
    save(fig, "S7C_false_susceptible_cases.pdf", pdf, paths)


PANEL_FUNCS = [
    s1a,
    s1b,
    s2a,
    s2b,
    s2c,
    s3a,
    s3b,
    s4a,
    s4b,
    s5a,
    s5b,
    s5c,
    s6a,
    s6b,
    s6c,
    s7a,
    s7b,
    s7c,
]


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    combined = OUT / f"IPM-GPT_supplementary_figures_combined_preview_{DATE}.pdf"
    with PdfPages(combined) as pdf:
        for fn in PANEL_FUNCS:
            fn(pdf, paths)
    manifest = pd.DataFrame(
        [
            {
                "panel": path.stem.split("_", 1)[0],
                "pdf_file": path.name,
                "absolute_path": str(path),
            }
            for path in paths
        ]
        + [
            {
                "panel": "combined",
                "pdf_file": combined.name,
                "absolute_path": str(combined),
            }
        ]
    )
    manifest.to_csv(OUT / f"supplementary_figure_pdf_manifest_{DATE}.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(paths)} supplementary panel PDFs to {OUT}")
    print(f"Combined preview: {combined}")


if __name__ == "__main__":
    main()
