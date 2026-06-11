#!/usr/bin/env python3
"""Statistical summaries for IPM/MEM mechanism-dependent predictability."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"
INPUT = (
    PROJECT
    / "results"
    / "03_mechanism_predictability"
    / f"ipm_mem_prediction_errors_with_mechanism_{DATE}.tsv"
)
OUT = PROJECT / "results" / "04_predictability_statistics"


def safe_p(value: float | None) -> float:
    if value is None or pd.isna(value):
        return math.nan
    return float(value)


def wilcoxon_paired(x: pd.Series, y: pd.Series) -> float:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(paired) < 6:
        return math.nan
    diff = paired["x"] - paired["y"]
    if np.allclose(diff, 0):
        return 1.0
    try:
        return safe_p(stats.wilcoxon(paired["x"], paired["y"], zero_method="wilcox").pvalue)
    except ValueError:
        return math.nan


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
    clean = [pd.to_numeric(group, errors="coerce").dropna() for group in groups]
    clean = [group for group in clean if len(group) >= 3]
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


def binom_p(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return math.nan
    if hasattr(stats, "binomtest"):
        return safe_p(stats.binomtest(k, n, p=p).pvalue)
    return safe_p(stats.binom_test(k, n, p=p))


def summarize_predictability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    local = df[df["cohort"].eq("locked_local_validation")].copy()
    for (drug, subtype), sub in local.groupby(["drug", "mechanism_subtype_strict"], dropna=False):
        n = len(sub)
        ea = float(sub["mic_ea_pm1"].mean())
        mae = float(sub["abs_log2_error"].mean())
        bias = float(sub["signed_log2_error"].mean())
        ca = float(sub["sns_correct"].mean())
        large = float((sub["abs_log2_error"] > 2).mean())
        if n >= 10 and ea >= 0.70 and mae <= 1.0 and abs(bias) <= 1.0:
            archetype = "predictable"
        elif n >= 10 and (ea < 0.30 or mae > 2.0 or abs(bias) > 2.0):
            archetype = "poorly_predictable"
        elif n >= 10 and ca >= 0.75:
            archetype = "directionally_predictable"
        else:
            archetype = "intermediate_or_small_n"
        rows.append(
            {
                "drug": drug,
                "mechanism_subtype_strict": subtype,
                "n": n,
                "mic_ea_pm1": ea,
                "mic_mae": mae,
                "mic_bias": bias,
                "large_error_gt2": large,
                "sns_ca": ca,
                "predictability_archetype": archetype,
            }
        )
    return pd.DataFrame(rows).sort_values(["drug", "mechanism_subtype_strict"]).reset_index(drop=True)


def paired_ipm_mem_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = df[df["cohort"].eq("locked_local_validation")].copy()
    base_cols = [
        "genome_id",
        "drug",
        "mechanism_subtype_strict",
        "ipm_mem_pair_group",
        "abs_log2_error",
        "signed_log2_error",
        "mic_ea_pm1",
        "sns_correct",
    ]
    wide = local[base_cols].pivot_table(
        index=["genome_id", "mechanism_subtype_strict", "ipm_mem_pair_group"],
        columns="drug",
        values=["abs_log2_error", "signed_log2_error", "mic_ea_pm1", "sns_correct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{drug}" for metric, drug in wide.columns]
    wide = wide.reset_index()
    paired = wide.dropna(subset=["abs_log2_error_IPM", "abs_log2_error_MEM"]).copy()
    paired["MEM_minus_IPM_abs_error"] = paired["abs_log2_error_MEM"] - paired["abs_log2_error_IPM"]
    paired["MEM_worse_than_IPM"] = paired["MEM_minus_IPM_abs_error"] > 0
    paired["MEM_much_worse_gt1"] = paired["MEM_minus_IPM_abs_error"] > 1

    rows = []
    groups = [("overall", paired)]
    groups.extend((str(subtype), sub) for subtype, sub in paired.groupby("mechanism_subtype_strict", dropna=False))
    groups.extend((f"pair_group::{name}", sub) for name, sub in paired.groupby("ipm_mem_pair_group", dropna=False))
    for name, sub in groups:
        if len(sub) == 0:
            continue
        rows.append(
            {
                "comparison_group": name,
                "n_paired": len(sub),
                "median_abs_error_IPM": float(sub["abs_log2_error_IPM"].median()),
                "median_abs_error_MEM": float(sub["abs_log2_error_MEM"].median()),
                "mean_abs_error_IPM": float(sub["abs_log2_error_IPM"].mean()),
                "mean_abs_error_MEM": float(sub["abs_log2_error_MEM"].mean()),
                "median_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].median()),
                "mean_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "MEM_worse_n": int(sub["MEM_worse_than_IPM"].sum()),
                "MEM_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
                "MEM_worse_binom_p": binom_p(int(sub["MEM_worse_than_IPM"].sum()), len(sub)),
                "MEM_much_worse_gt1_fraction": float(sub["MEM_much_worse_gt1"].mean()),
                "paired_wilcoxon_abs_error_p": wilcoxon_paired(
                    sub["abs_log2_error_MEM"],
                    sub["abs_log2_error_IPM"],
                ),
            }
        )
    return paired, pd.DataFrame(rows)


def subtype_tests(df: pd.DataFrame) -> pd.DataFrame:
    local = df[df["cohort"].eq("locked_local_validation")].copy()
    rows = []
    for drug, drug_df in local.groupby("drug", dropna=False):
        groups = [
            sub["abs_log2_error"]
            for _, sub in drug_df.groupby("mechanism_subtype_strict", dropna=False)
        ]
        subtype_x_ea = pd.crosstab(drug_df["mechanism_subtype_strict"], drug_df["mic_ea_pm1"])
        rows.append(
            {
                "drug": drug,
                "test": "Kruskal-Wallis abs_log2_error across strict subtypes",
                "p_value": kruskal_p(groups),
                "n": len(drug_df),
            }
        )
        rows.append(
            {
                "drug": drug,
                "test": "Chi-square within_1_dilution across strict subtypes",
                "p_value": chi_square_p(subtype_x_ea),
                "n": len(drug_df),
            }
        )

    for subtype, sub in local.groupby("mechanism_subtype_strict", dropna=False):
        wide = sub.pivot_table(index="genome_id", columns="drug", values="abs_log2_error", aggfunc="first")
        if {"IPM", "MEM"}.issubset(wide.columns):
            rows.append(
                {
                    "drug": "IPM_vs_MEM",
                    "test": f"Paired Wilcoxon abs_log2_error within {subtype}",
                    "p_value": wilcoxon_paired(wide["MEM"], wide["IPM"]),
                    "n": int(wide.dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def top_error_cases(df: pd.DataFrame) -> pd.DataFrame:
    local = df[df["cohort"].eq("locked_local_validation")].copy()
    cols = [
        "drug",
        "genome_id",
        "ipm_mem_pair_group",
        "mechanism_subtype_strict",
        "mechanism_multilabel",
        "actual_mic_log2",
        "pred_mic_log2",
        "signed_log2_error",
        "abs_log2_error",
        "mic_ea_pm1",
        "sns_correct",
        "mechanism_reason",
    ]
    return (
        local.sort_values(["drug", "abs_log2_error"], ascending=[True, False])
        .groupby("drug", group_keys=False)
        .head(20)[cols]
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
    archetypes: pd.DataFrame,
    paired_tests: pd.DataFrame,
    subtype_test_table: pd.DataFrame,
    top_errors: pd.DataFrame,
) -> None:
    report = OUT / f"predictability_statistics_report_{DATE}.md"
    lines = [
        "# IPM/MEM Predictability Statistics",
        "",
        f"Date: {DATE}",
        "",
        "## Purpose",
        "",
        "This report converts the first-pass mechanism-predictability table into statistical summaries for discussion.",
        "These are still first-pass results because ST/MLST adjustment has not yet been added.",
        "",
        "## Predictability Archetypes",
        "",
        markdown_table(archetypes),
        "",
        "## Paired IPM vs MEM Error Tests",
        "",
        markdown_table(paired_tests),
        "",
        "## Subtype Association Tests",
        "",
        markdown_table(subtype_test_table),
        "",
        "## Top Local Error Cases",
        "",
        markdown_table(top_errors),
        "",
        "## Interpretation Guardrail",
        "",
        "- Do not treat these p-values as final confirmatory statistics until ST/lineage adjustment is added.",
        "- Current results are useful for deciding whether the mechanism-dependent predictability story is worth pursuing.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, sep="\t", low_memory=False)
    archetypes = summarize_predictability(df)
    paired_rows, paired_tests = paired_ipm_mem_tests(df)
    subtype_test_table = subtype_tests(df)
    errors = top_error_cases(df)

    archetypes.to_csv(OUT / f"local_predictability_archetypes_{DATE}.csv", index=False)
    paired_rows.to_csv(OUT / f"local_paired_ipm_mem_error_table_{DATE}.csv", index=False)
    paired_tests.to_csv(OUT / f"local_paired_ipm_mem_error_tests_{DATE}.csv", index=False)
    subtype_test_table.to_csv(OUT / f"local_subtype_predictability_tests_{DATE}.csv", index=False)
    errors.to_csv(OUT / f"local_top_prediction_error_cases_{DATE}.csv", index=False)
    write_report(archetypes, paired_tests, subtype_test_table, errors)

    print(archetypes.to_string(index=False))
    print()
    print(paired_tests.to_string(index=False))
    print()
    print(subtype_test_table.to_string(index=False))
    print(f"Wrote predictability statistics to {OUT}")


if __name__ == "__main__":
    main()
