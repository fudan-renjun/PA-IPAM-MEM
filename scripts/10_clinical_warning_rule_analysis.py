#!/usr/bin/env python3
"""Clinical warning-rule analysis for high-confidence IPM/MEM predictability."""

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
OUT = PROJECT / "results" / "10_clinical_warning_rules"


HIGH_CONF_ORDER = {
    "OprD-loss": 1,
    "High-confidence composite": 2,
    "AmpC-axis disruptive": 3,
    "carbapenemase": 4,
    "efflux disruptive": 5,
    "No high-confidence driver": 6,
}
RULE_ORDER = {
    "trustworthy": 1,
    "exploratory_trustworthy": 2,
    "caution": 3,
    "warning": 4,
    "exploratory_warning": 5,
    "mechanism_unresolved_warning": 6,
}


def safe_p(value: float | None) -> float:
    if value is None or pd.isna(value):
        return math.nan
    return float(value)


def fisher_p(flag: pd.Series, outcome: pd.Series) -> float:
    tab = pd.crosstab(flag.astype(bool), outcome.astype(bool))
    if tab.shape != (2, 2):
        return math.nan
    try:
        return safe_p(stats.fisher_exact(tab.to_numpy()).pvalue)
    except ValueError:
        return math.nan


def categorical_error_rates(sub: pd.DataFrame) -> dict[str, float | int]:
    n = int(len(sub))
    n_s = int((sub["true_sns_int"] == 0).sum())
    n_ns = int((sub["true_sns_int"] == 1).sum())
    false_susceptible_n = int(sub["vme"].astype(bool).sum())
    false_resistant_n = int(sub["me"].astype(bool).sum())
    return {
        "n": n,
        "n_s": n_s,
        "n_ns": n_ns,
        "false_susceptible_n": false_susceptible_n,
        "false_resistant_n": false_resistant_n,
        "very_major_error_rate": false_susceptible_n / n_ns if n_ns else math.nan,
        "major_error_rate": false_resistant_n / n_s if n_s else math.nan,
        "false_susceptible_all_row_rate": false_susceptible_n / n if n else math.nan,
        "false_resistant_all_row_rate": false_resistant_n / n if n else math.nan,
    }


def summarize_rule_candidates(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (drug, subtype), sub in errors.groupby(
        ["drug", "mechanism_subtype_high_confidence"], dropna=False
    ):
        rates = categorical_error_rates(sub)
        within_1 = float(sub["mic_ea_pm1"].mean())
        mae = float(sub["abs_log2_error"].mean())
        bias = float(sub["signed_log2_error"].mean())
        large = float((sub["abs_log2_error"] > 2).mean())
        ca = float(sub["sns_correct"].mean())
        vme = float(rates["very_major_error_rate"])
        me = float(rates["major_error_rate"])
        rule, reason = assign_rule(drug, subtype, int(rates["n"]), within_1, mae, bias, vme, me, ca)
        rows.append(
            {
                "drug": drug,
                "mechanism_subtype_high_confidence": subtype,
                "n": rates["n"],
                "n_s": rates["n_s"],
                "n_ns": rates["n_ns"],
                "within_1_dilution": within_1,
                "mae_log2": mae,
                "bias_log2": bias,
                "large_error_gt2": large,
                "categorical_agreement": ca,
                "very_major_error_rate": vme,
                "major_error_rate": me,
                "false_susceptible_n": rates["false_susceptible_n"],
                "false_resistant_n": rates["false_resistant_n"],
                "false_susceptible_all_row_rate": rates["false_susceptible_all_row_rate"],
                "false_resistant_all_row_rate": rates["false_resistant_all_row_rate"],
                "clinical_rule": rule,
                "rule_reason": reason,
                "drug_order": 1 if drug == "IPM" else 2,
                "high_conf_order": HIGH_CONF_ORDER.get(str(subtype), 99),
                "rule_order": RULE_ORDER.get(rule, 99),
            }
        )
    return pd.DataFrame(rows).sort_values(["drug_order", "high_conf_order"]).reset_index(drop=True)


def assign_rule(
    drug: str,
    subtype: str,
    n: int,
    within_1: float,
    mae: float,
    bias: float,
    vme: float,
    me: float,
    ca: float,
) -> tuple[str, str]:
    if subtype == "No high-confidence driver":
        return (
            "mechanism_unresolved_warning",
            "no high-confidence mechanism; performance requires phenotype confirmation",
        )

    vme_eval = 0.0 if pd.isna(vme) else float(vme)
    me_eval = 0.0 if pd.isna(me) else float(me)
    high_risk = vme_eval >= 0.10 or mae > 2.0 or bias <= -2.0 or within_1 < 0.30
    strong_performance = within_1 >= 0.80 and mae <= 1.0 and abs(bias) <= 1.0 and vme_eval == 0 and me_eval == 0 and ca >= 0.90

    if n < 10:
        if high_risk:
            return (
                "exploratory_warning",
                "small n but high VME/MAE/bias signal; do not use as stand-alone call",
            )
        if strong_performance:
            return (
                "exploratory_trustworthy",
                "small n with strong apparent performance; needs validation",
            )
        return ("caution", "small n or intermediate performance")

    if high_risk:
        return ("warning", "high VME/MAE/bias or poor essential agreement")
    if strong_performance:
        return ("trustworthy", "high essential agreement and no categorical safety errors")
    return ("caution", "intermediate performance")


def apply_rules(errors: pd.DataFrame, rule_table: pd.DataFrame) -> pd.DataFrame:
    cols = ["drug", "mechanism_subtype_high_confidence", "clinical_rule", "rule_reason"]
    out = errors.merge(rule_table[cols], on=["drug", "mechanism_subtype_high_confidence"], how="left")
    out["model_predicted_susceptible"] = out["pred_sns_int"].eq(0)
    out["true_non_susceptible"] = out["true_sns_int"].eq(1)
    out["false_susceptible_call"] = out["vme"].astype(bool)
    out["false_resistant_call"] = out["me"].astype(bool)
    out["large_underprediction_gt2"] = out["signed_log2_error"].lt(-2)
    out["large_overprediction_gt2"] = out["signed_log2_error"].gt(2)
    return out


def summarize_rule_performance(applied: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (drug, rule), sub in applied.groupby(["drug", "clinical_rule"], dropna=False):
        rates = categorical_error_rates(sub)
        rows.append(
            {
                "drug": drug,
                "clinical_rule": rule,
                "n": rates["n"],
                "n_s": rates["n_s"],
                "n_ns": rates["n_ns"],
                "within_1_dilution": float(sub["mic_ea_pm1"].mean()),
                "mae_log2": float(sub["abs_log2_error"].mean()),
                "bias_log2": float(sub["signed_log2_error"].mean()),
                "categorical_agreement": float(sub["sns_correct"].mean()),
                "very_major_error_rate": rates["very_major_error_rate"],
                "major_error_rate": rates["major_error_rate"],
                "false_susceptible_n": rates["false_susceptible_n"],
                "false_resistant_n": rates["false_resistant_n"],
                "false_susceptible_all_row_rate": rates["false_susceptible_all_row_rate"],
                "false_resistant_all_row_rate": rates["false_resistant_all_row_rate"],
                "large_underprediction_gt2_rate": float(sub["large_underprediction_gt2"].mean()),
                "large_overprediction_gt2_rate": float(sub["large_overprediction_gt2"].mean()),
                "rule_order": RULE_ORDER.get(str(rule), 99),
            }
        )
    return pd.DataFrame(rows).sort_values(["drug", "rule_order"]).reset_index(drop=True)


def mem_warning_enrichment(applied: pd.DataFrame) -> pd.DataFrame:
    mem = applied[applied["drug"].eq("MEM")].copy()
    rows = []
    for flag_col, label in [
        ("oprd_deep_disruptive", "OprD deep disruptive"),
        ("oprd_severe_loss", "OprD severe loss"),
        ("ampc_core_driver_disruptive_any", "AmpC core disruptive"),
        ("efflux_strict_driver_disruptive_any", "strict efflux disruptive"),
        ("mechanism_subtype_high_confidence", "any high-confidence driver"),
    ]:
        if flag_col == "mechanism_subtype_high_confidence":
            flag = ~mem[flag_col].eq("No high-confidence driver")
        else:
            flag = mem[flag_col].astype(bool)
        flagged = mem.loc[flag]
        unflagged = mem.loc[~flag]
        flagged_rates = categorical_error_rates(flagged)
        unflagged_rates = categorical_error_rates(unflagged)
        rows.append(
            {
                "flag": label,
                "flagged_n": int(flag.sum()),
                "unflagged_n": int((~flag).sum()),
                "flagged_vme_rate": flagged_rates["very_major_error_rate"],
                "unflagged_vme_rate": unflagged_rates["very_major_error_rate"],
                "flagged_false_susceptible_n": flagged_rates["false_susceptible_n"],
                "unflagged_false_susceptible_n": unflagged_rates["false_susceptible_n"],
                "flagged_n_ns": flagged_rates["n_ns"],
                "unflagged_n_ns": unflagged_rates["n_ns"],
                "flagged_large_underprediction_gt2_rate": float(mem.loc[flag, "large_underprediction_gt2"].mean())
                if flag.any()
                else math.nan,
                "unflagged_large_underprediction_gt2_rate": float(mem.loc[~flag, "large_underprediction_gt2"].mean())
                if (~flag).any()
                else math.nan,
                "fisher_vme_p": fisher_p(flag, mem["vme"]),
                "fisher_large_underprediction_p": fisher_p(flag, mem["large_underprediction_gt2"]),
            }
        )
    return pd.DataFrame(rows)


def paired_clinical_context(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subtype, sub in paired.groupby("mechanism_subtype_high_confidence", dropna=False):
        rows.append(
            {
                "mechanism_subtype_high_confidence": subtype,
                "n_paired": int(len(sub)),
                "ipm_r_mem_r_n": int(sub["ipm_mem_pair_group"].eq("IPM-R/MEM-R").sum()),
                "ipm_r_mem_s_or_i_n": int(sub["ipm_mem_pair_group"].eq("IPM-R/MEM-S_or_I").sum()),
                "ipm_s_mem_s_n": int(sub["ipm_mem_pair_group"].eq("IPM-S/MEM-S").sum()),
                "ipm_s_mem_r_n": int(sub["ipm_mem_pair_group"].eq("IPM-S/MEM-R").sum()),
                "mean_mem_minus_ipm_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "median_mem_minus_ipm_abs_error": float(sub["MEM_minus_IPM_abs_error"].median()),
                "mem_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
                "high_conf_order": HIGH_CONF_ORDER.get(str(subtype), 99),
            }
        )
    return pd.DataFrame(rows).sort_values(["high_conf_order", "mechanism_subtype_high_confidence"])


def false_susceptible_cases(applied: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "drug",
        "genome_id",
        "ipm_mem_pair_group",
        "mechanism_subtype_high_confidence",
        "clinical_rule",
        "actual_mic_log2",
        "pred_mic_log2",
        "signed_log2_error",
        "abs_log2_error",
        "actual_sns",
        "pred_sns",
        "prob_ns",
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
        applied[applied["vme"].astype(bool)]
        .sort_values(["drug", "mechanism_subtype_high_confidence", "abs_log2_error"], ascending=[True, True, False])
        [[c for c in cols if c in applied.columns]]
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
    rule_table: pd.DataFrame,
    rule_perf: pd.DataFrame,
    mem_enrichment: pd.DataFrame,
    paired_context: pd.DataFrame,
    vme_cases: pd.DataFrame,
) -> None:
    report = OUT / f"clinical_warning_rule_report_{DATE}.md"
    lines = [
        "# Clinical Warning Rule Analysis",
        "",
        f"Date: {DATE}",
        "",
        "## Rule Table",
        "",
        markdown_table(
            rule_table[
                [
                    "drug",
                    "mechanism_subtype_high_confidence",
                    "n",
                    "within_1_dilution",
                    "mae_log2",
                    "bias_log2",
                    "very_major_error_rate",
                    "major_error_rate",
                    "clinical_rule",
                    "rule_reason",
                ]
            ]
        ),
        "",
        "## Rule-Level Performance",
        "",
        markdown_table(rule_perf),
        "",
        "## MEM Warning Enrichment",
        "",
        markdown_table(mem_enrichment),
        "",
        "## Paired Clinical Context",
        "",
        markdown_table(paired_context),
        "",
        "## False-Susceptible Cases",
        "",
        markdown_table(vme_cases.head(25)),
        "",
        "## Interpretation",
        "",
        "- IPM high-confidence OprD-loss is the clearest trustworthy stratum in this dataset.",
        "- MEM high-confidence OprD-loss is a warning stratum because it combines high MAE, negative bias, and high VME.",
        "- Composite/AmpC-axis strata point in the same direction but should be treated as exploratory because sample sizes are small.",
        "- No high-confidence driver should not be interpreted as no mechanism; it is best presented as mechanism-unresolved and requiring phenotype confirmation.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(DEEP_ERRORS, sep="\t", low_memory=False)
    paired = pd.read_csv(DEEP_PAIRED, low_memory=False)

    rule_table = summarize_rule_candidates(errors)
    applied = apply_rules(errors, rule_table)
    rule_perf = summarize_rule_performance(applied)
    mem_enrichment = mem_warning_enrichment(applied)
    paired_context = paired_clinical_context(paired)
    vme_cases = false_susceptible_cases(applied)

    rule_table.to_csv(OUT / f"clinical_warning_rule_table_{DATE}.csv", index=False)
    applied.to_csv(OUT / f"prediction_rows_with_clinical_warning_rule_{DATE}.tsv", sep="\t", index=False)
    rule_perf.to_csv(OUT / f"clinical_rule_level_performance_{DATE}.csv", index=False)
    mem_enrichment.to_csv(OUT / f"mem_warning_enrichment_tests_{DATE}.csv", index=False)
    paired_context.to_csv(OUT / f"paired_clinical_context_by_high_confidence_subtype_{DATE}.csv", index=False)
    vme_cases.to_csv(OUT / f"false_susceptible_cases_by_warning_rule_{DATE}.csv", index=False)
    write_report(rule_table, rule_perf, mem_enrichment, paired_context, vme_cases)


if __name__ == "__main__":
    main()
