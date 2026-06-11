#!/usr/bin/env python3
"""Statistical tests for high-confidence mechanism IPM/MEM predictability."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"

DEEP_ERRORS = (
    PROJECT
    / "results"
    / "08_deep_mechanism_annotation"
    / f"local_prediction_errors_with_deep_mechanism_{DATE}.tsv"
)
DEEP_PAIRED = (
    PROJECT
    / "results"
    / "08_deep_mechanism_annotation"
    / f"local_paired_ipm_mem_deep_mechanism_errors_{DATE}.csv"
)
OUT = PROJECT / "results" / "09_high_confidence_statistics"


def safe_p(value: float | None) -> float:
    if value is None or pd.isna(value):
        return math.nan
    return float(value)


def wilcoxon_paired(x: pd.Series, y: pd.Series) -> float:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(paired) < 3:
        return math.nan
    diff = paired["x"] - paired["y"]
    if np.allclose(diff, 0):
        return 1.0
    try:
        return safe_p(stats.wilcoxon(paired["x"], paired["y"], zero_method="wilcox").pvalue)
    except ValueError:
        return math.nan


def binom_p(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return math.nan
    if hasattr(stats, "binomtest"):
        return safe_p(stats.binomtest(k, n, p=p).pvalue)
    return safe_p(stats.binom_test(k, n, p=p))


def mannwhitney_p(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return math.nan
    try:
        return safe_p(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def kruskal_p(groups: list[pd.Series]) -> float:
    clean = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    clean = [g for g in clean if len(g) >= 3]
    if len(clean) < 2:
        return math.nan
    try:
        return safe_p(stats.kruskal(*clean).pvalue)
    except ValueError:
        return math.nan


def chi_square_p(table: pd.DataFrame) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return math.nan
    try:
        return safe_p(stats.chi2_contingency(table).pvalue)
    except ValueError:
        return math.nan


def fisher_p_from_bool(flag: pd.Series, outcome: pd.Series) -> float:
    tab = pd.crosstab(flag.astype(bool), outcome.astype(bool))
    if tab.shape != (2, 2):
        return math.nan
    try:
        return safe_p(stats.fisher_exact(tab.to_numpy()).pvalue)
    except ValueError:
        return math.nan


def summarize_high_confidence_performance(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (drug, subtype), sub in errors.groupby(
        ["drug", "mechanism_subtype_high_confidence"], dropna=False
    ):
        rows.append(
            {
                "drug": drug,
                "mechanism_subtype_high_confidence": subtype,
                "n": int(len(sub)),
                "n_s": int((sub["true_sns_int"] == 0).sum()),
                "n_ns": int((sub["true_sns_int"] == 1).sum()),
                "within_1_dilution": float(sub["mic_ea_pm1"].mean()),
                "mae_log2": float(sub["abs_log2_error"].mean()),
                "median_abs_error": float(sub["abs_log2_error"].median()),
                "bias_log2": float(sub["signed_log2_error"].mean()),
                "large_error_gt2": float((sub["abs_log2_error"] > 2).mean()),
                "categorical_agreement": float(sub["sns_correct"].mean()),
                "very_major_error_rate": float(sub["vme"].mean()),
                "major_error_rate": float(sub["me"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["drug", "mechanism_subtype_high_confidence"]
    ).reset_index(drop=True)


def paired_subtype_tests(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", paired)]
    groups.extend(
        (str(subtype), sub)
        for subtype, sub in paired.groupby("mechanism_subtype_high_confidence", dropna=False)
    )
    groups.extend(
        (f"pair_group::{group}", sub)
        for group, sub in paired.groupby("ipm_mem_pair_group", dropna=False)
    )
    for name, sub in groups:
        if sub.empty:
            continue
        mem_worse = int(sub["MEM_worse_than_IPM"].sum())
        rows.append(
            {
                "comparison_group": name,
                "n_paired": int(len(sub)),
                "mean_abs_error_IPM": float(sub["abs_log2_error_IPM"].mean()),
                "mean_abs_error_MEM": float(sub["abs_log2_error_MEM"].mean()),
                "median_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].median()),
                "mean_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "MEM_worse_n": mem_worse,
                "MEM_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
                "MEM_worse_binom_p": binom_p(mem_worse, len(sub)),
                "paired_wilcoxon_abs_error_p": wilcoxon_paired(
                    sub["abs_log2_error_MEM"],
                    sub["abs_log2_error_IPM"],
                ),
            }
        )
    return pd.DataFrame(rows)


def high_confidence_global_tests(errors: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, sub in errors.groupby("drug", dropna=False):
        groups = [
            g["abs_log2_error"]
            for _, g in sub.groupby("mechanism_subtype_high_confidence", dropna=False)
        ]
        rows.append(
            {
                "scope": drug,
                "test": "Kruskal-Wallis abs_log2_error across high-confidence subtypes",
                "n": int(len(sub)),
                "p_value": kruskal_p(groups),
            }
        )
        rows.append(
            {
                "scope": drug,
                "test": "Chi-square within_1_dilution across high-confidence subtypes",
                "n": int(len(sub)),
                "p_value": chi_square_p(
                    pd.crosstab(sub["mechanism_subtype_high_confidence"], sub["mic_ea_pm1"])
                ),
            }
        )

    baseline = paired["mechanism_subtype_high_confidence"].eq("No high-confidence driver")
    for flag_col, label in [
        ("oprd_deep_disruptive", "OprD deep disruptive"),
        ("oprd_severe_loss", "OprD severe loss first-pass"),
        ("ampc_core_driver_disruptive_any", "AmpC core disruptive"),
        ("efflux_strict_driver_disruptive_any", "strict efflux disruptive"),
    ]:
        if flag_col not in paired.columns:
            continue
        flag = paired[flag_col].astype(bool)
        rows.append(
            {
                "scope": "paired",
                "test": f"Mann-Whitney MEM_minus_IPM_abs_error by {label}",
                "n": int(paired[flag_col].notna().sum()),
                "p_value": mannwhitney_p(
                    paired.loc[flag, "MEM_minus_IPM_abs_error"],
                    paired.loc[~flag, "MEM_minus_IPM_abs_error"],
                ),
            }
        )
        rows.append(
            {
                "scope": "paired",
                "test": f"Fisher MEM_worse_than_IPM by {label}",
                "n": int(paired[flag_col].notna().sum()),
                "p_value": fisher_p_from_bool(flag, paired["MEM_worse_than_IPM"]),
            }
        )

    rows.append(
        {
            "scope": "paired",
            "test": "Mann-Whitney MEM_minus_IPM_abs_error high-confidence driver vs none",
            "n": int(len(paired)),
            "p_value": mannwhitney_p(
                paired.loc[~baseline, "MEM_minus_IPM_abs_error"],
                paired.loc[baseline, "MEM_minus_IPM_abs_error"],
            ),
        }
    )
    rows.append(
        {
            "scope": "paired",
            "test": "Fisher MEM_worse_than_IPM high-confidence driver vs none",
            "n": int(len(paired)),
            "p_value": fisher_p_from_bool(~baseline, paired["MEM_worse_than_IPM"]),
        }
    )
    return pd.DataFrame(rows)


def top_high_confidence_cases(errors: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "drug",
        "genome_id",
        "ipm_mem_pair_group",
        "mechanism_subtype_high_confidence",
        "mechanism_subtype_strict",
        "actual_mic_log2",
        "pred_mic_log2",
        "signed_log2_error",
        "abs_log2_error",
        "mic_ea_pm1",
        "sns_correct",
        "oprd_deep_severity",
        "ampR_severity",
        "ampC_severity",
        "ampD_severity",
        "dacB_severity",
        "mexR_severity",
        "nalD_severity",
        "mexS_severity",
        "mexZ_severity",
        "nfxB_severity",
    ]
    return (
        errors.sort_values(["drug", "abs_log2_error"], ascending=[True, False])
        .groupby("drug", group_keys=False)
        .head(25)[[c for c in cols if c in errors.columns]]
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    if df.empty:
        return "(no rows)"
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]):
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            safe[col] = safe[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(safe.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    perf: pd.DataFrame,
    paired_tests: pd.DataFrame,
    global_tests: pd.DataFrame,
    top_cases: pd.DataFrame,
) -> None:
    report = OUT / f"high_confidence_statistics_report_{DATE}.md"
    main_paired = paired_tests[
        paired_tests["comparison_group"].isin(
            [
                "overall",
                "OprD-loss",
                "High-confidence composite",
                "AmpC-axis disruptive",
                "No high-confidence driver",
            ]
        )
    ]
    lines = [
        "# High-Confidence Mechanism Statistics",
        "",
        f"Date: {DATE}",
        "",
        "## Performance Summary",
        "",
        markdown_table(perf),
        "",
        "## Paired IPM/MEM Tests",
        "",
        markdown_table(main_paired),
        "",
        "## Global And Flag Tests",
        "",
        markdown_table(global_tests),
        "",
        "## Top Error Cases",
        "",
        markdown_table(top_cases.head(20)),
        "",
        "## Interpretation",
        "",
        "- High-confidence OprD-loss and composite groups show consistently larger MEM than IPM errors.",
        "- AmpC-axis disruptive and strict efflux disruptive flags are small-n exploratory strata; use them as supportive signals unless externally validated.",
        "- The high-confidence layer is more manuscript-defensible than broad PAO1-relative missense grouping.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(DEEP_ERRORS, sep="\t", low_memory=False)
    paired = pd.read_csv(DEEP_PAIRED, low_memory=False)

    perf = summarize_high_confidence_performance(errors)
    paired_tests = paired_subtype_tests(paired)
    global_tests = high_confidence_global_tests(errors, paired)
    top_cases = top_high_confidence_cases(errors)

    perf.to_csv(OUT / f"high_confidence_predictability_summary_{DATE}.csv", index=False)
    paired_tests.to_csv(OUT / f"high_confidence_paired_ipm_mem_tests_{DATE}.csv", index=False)
    global_tests.to_csv(OUT / f"high_confidence_global_tests_{DATE}.csv", index=False)
    top_cases.to_csv(OUT / f"high_confidence_top_error_cases_{DATE}.csv", index=False)
    write_report(perf, paired_tests, global_tests, top_cases)


if __name__ == "__main__":
    main()
