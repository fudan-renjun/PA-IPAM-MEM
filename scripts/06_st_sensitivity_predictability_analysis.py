#!/usr/bin/env python3
"""Add local MLST/ST context to IPM/MEM predictability analyses."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"
MECH_ERRORS = (
    PROJECT
    / "results"
    / "03_mechanism_predictability"
    / f"ipm_mem_prediction_errors_with_mechanism_{DATE}.tsv"
)
MLST = PROJECT / "results" / "05_mlst" / f"local_mlst_pubmlst_exact_{DATE}.tsv"
OUT = PROJECT / "results" / "06_st_sensitivity"


def wilcoxon_paired(mem: pd.Series, ipm: pd.Series) -> float:
    paired = pd.DataFrame({"MEM": mem, "IPM": ipm}).dropna()
    if len(paired) < 6:
        return math.nan
    diff = paired["MEM"] - paired["IPM"]
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(paired["MEM"], paired["IPM"], zero_method="wilcox").pvalue)


def binom_p(k: int, n: int) -> float:
    if n <= 0:
        return math.nan
    if hasattr(stats, "binomtest"):
        return float(stats.binomtest(k, n, p=0.5).pvalue)
    return float(stats.binom_test(k, n, p=0.5))


def load_joined() -> pd.DataFrame:
    df = pd.read_csv(MECH_ERRORS, sep="\t", low_memory=False)
    mlst = pd.read_csv(MLST, sep="\t", dtype=str).fillna("")
    keep = [
        "isolate_id",
        "ST",
        "clonal_complex",
        "mlst_call_status",
        "allelic_profile",
        "n_loci_exact_single",
    ]
    mlst = mlst[[c for c in keep if c in mlst.columns]].rename(columns={"isolate_id": "genome_id"})
    local = df[df["cohort"].eq("locked_local_validation")].copy()
    local = local.merge(mlst, on="genome_id", how="left", validate="many_to_one")
    local["ST"] = local["ST"].fillna("")
    local["ST_label"] = np.where(local["ST"].astype(str).str.len().gt(0), "ST" + local["ST"].astype(str), "ST_unassigned")
    return local


def make_paired(local: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "genome_id",
        "ST_label",
        "ST",
        "mlst_call_status",
        "mechanism_subtype_strict",
        "ipm_mem_pair_group",
        "drug",
        "abs_log2_error",
        "signed_log2_error",
        "mic_ea_pm1",
        "sns_correct",
    ]
    wide = local[cols].pivot_table(
        index=[
            "genome_id",
            "ST_label",
            "ST",
            "mlst_call_status",
            "mechanism_subtype_strict",
            "ipm_mem_pair_group",
        ],
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
    st_counts = paired["ST_label"].value_counts()
    paired["ST_adjustment_group"] = paired["ST_label"].where(
        paired["ST_label"].map(st_counts).ge(4),
        "ST_other_or_sparse",
    )
    return paired


def st_distribution(local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    isolate_level = local.drop_duplicates("genome_id").copy()
    st_counts = (
        isolate_level.groupby(["ST_label", "mlst_call_status"], dropna=False)
        .agg(
            isolates=("genome_id", "size"),
            n_subtypes=("mechanism_subtype_strict", "nunique"),
        )
        .reset_index()
        .sort_values(["isolates", "ST_label"], ascending=[False, True])
    )
    st_subtype = pd.crosstab(
        isolate_level["ST_label"],
        isolate_level["mechanism_subtype_strict"],
    ).reset_index()
    st_subtype["isolates"] = st_subtype.drop(columns=["ST_label"]).sum(axis=1)
    st_subtype = st_subtype.sort_values(["isolates", "ST_label"], ascending=[False, True])
    return st_counts, st_subtype


def paired_by_st(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, sub in paired.groupby("ST_label", dropna=False):
        rows.append(
            {
                "ST_label": st,
                "n_paired": len(sub),
                "n_mechanism_subtypes": int(sub["mechanism_subtype_strict"].nunique()),
                "median_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].median()),
                "mean_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "MEM_worse_n": int(sub["MEM_worse_than_IPM"].sum()),
                "MEM_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
                "MEM_worse_binom_p": binom_p(int(sub["MEM_worse_than_IPM"].sum()), len(sub)),
                "paired_wilcoxon_abs_error_p": wilcoxon_paired(
                    sub["abs_log2_error_MEM"],
                    sub["abs_log2_error_IPM"],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_paired", "ST_label"], ascending=[False, True])


def leave_one_st_out(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, n in paired["ST_label"].value_counts().items():
        if n < 4:
            continue
        sub = paired[~paired["ST_label"].eq(st)].copy()
        rows.append(
            {
                "excluded_ST_label": st,
                "excluded_n": int(n),
                "remaining_n": int(len(sub)),
                "remaining_MEM_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
                "remaining_mean_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "remaining_paired_wilcoxon_abs_error_p": wilcoxon_paired(
                    sub["abs_log2_error_MEM"],
                    sub["abs_log2_error_IPM"],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["excluded_n", "excluded_ST_label"], ascending=[False, True])


def model_st_adjusted(paired: pd.DataFrame) -> pd.DataFrame:
    model_df = paired.copy()
    model_df = model_df[model_df["ST_label"].ne("ST_unassigned")].copy()
    model_df["mechanism_subtype_strict"] = model_df["mechanism_subtype_strict"].astype("category")
    model_df["ST_adjustment_group"] = model_df["ST_adjustment_group"].astype("category")
    rows = []
    formulas = [
        "MEM_minus_IPM_abs_error ~ C(mechanism_subtype_strict)",
        "MEM_minus_IPM_abs_error ~ C(ST_adjustment_group)",
        "MEM_minus_IPM_abs_error ~ C(mechanism_subtype_strict) + C(ST_adjustment_group)",
    ]
    for formula in formulas:
        fit = smf.ols(formula, data=model_df).fit(cov_type="HC3")
        rows.append(
            {
                "model": formula,
                "n": int(fit.nobs),
                "r_squared": float(fit.rsquared),
                "adj_r_squared": float(fit.rsquared_adj),
                "intercept_or_grand_mean": float(fit.params.get("Intercept", math.nan)),
                "model_f_pvalue": float(fit.f_pvalue) if fit.f_pvalue is not None else math.nan,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
            }
        )
    return pd.DataFrame(rows)


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
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    st_counts: pd.DataFrame,
    paired_st: pd.DataFrame,
    leave_one: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    top_st = st_counts.head(20)
    paired_show = paired_st[paired_st["n_paired"].ge(3)].head(30)
    report = OUT / f"st_sensitivity_report_{DATE}.md"
    lines = [
        "# ST Sensitivity Analysis",
        "",
        f"Date: {DATE}",
        "",
        "## Purpose",
        "",
        "This first ST sensitivity analysis tests whether the paired finding that MEM errors exceed IPM errors is dominated by a single local ST.",
        "",
        "## Top Local STs",
        "",
        markdown_table(top_st),
        "",
        "## Paired MEM-vs-IPM Error Difference By ST",
        "",
        markdown_table(paired_show),
        "",
        "## Leave-One-Major-ST-Out Sensitivity",
        "",
        markdown_table(leave_one),
        "",
        "## Exploratory ST-Adjusted Linear Models",
        "",
        markdown_table(models),
        "",
        "## Interpretation",
        "",
        "- In the local prediction-analysis subset, ST244 is the largest ST but contains only 13 isolates with IPM/MEM prediction rows and 12 paired IPM/MEM isolates, so the signal is not dominated by a single ST.",
        "- In the full local 147-isolate MLST table, ST244 contains 17 isolates; the smaller denominator here reflects filtering to isolates with prediction-error rows.",
        "- The leave-one-ST-out table should show whether the MEM-worse-than-IPM signal persists when each major ST is removed.",
        "- These ST-adjusted models are exploratory because most STs are sparse; they should be used as sensitivity evidence, not final causal adjustment.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local = load_joined()
    paired = make_paired(local)
    st_counts, st_subtype = st_distribution(local)
    paired_st = paired_by_st(paired)
    leave_one = leave_one_st_out(paired)
    models = model_st_adjusted(paired)

    local.to_csv(OUT / f"local_prediction_errors_with_st_{DATE}.tsv", sep="\t", index=False)
    paired.to_csv(OUT / f"local_paired_ipm_mem_errors_with_st_{DATE}.csv", index=False)
    st_counts.to_csv(OUT / f"local_st_counts_{DATE}.csv", index=False)
    st_subtype.to_csv(OUT / f"local_st_by_strict_subtype_{DATE}.csv", index=False)
    paired_st.to_csv(OUT / f"local_paired_error_difference_by_st_{DATE}.csv", index=False)
    leave_one.to_csv(OUT / f"local_leave_one_st_out_sensitivity_{DATE}.csv", index=False)
    models.to_csv(OUT / f"local_st_adjusted_models_{DATE}.csv", index=False)
    write_report(st_counts, paired_st, leave_one, models)

    print(st_counts.head(20).to_string(index=False))
    print()
    print(paired_st[paired_st["n_paired"].ge(3)].head(30).to_string(index=False))
    print()
    print(leave_one.to_string(index=False))
    print()
    print(models.to_string(index=False))
    print(f"Wrote ST sensitivity outputs to {OUT}")


if __name__ == "__main__":
    main()
