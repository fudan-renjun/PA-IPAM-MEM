#!/usr/bin/env python3
"""Additional clinical/error analyses for the IPM/MEM predictability study."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"

PRED_ROWS = (
    PROJECT
    / "results"
    / "10_clinical_warning_rules"
    / f"prediction_rows_with_clinical_warning_rule_{DATE}.tsv"
)
DEEP_PAIRED = (
    PROJECT
    / "results"
    / "08_deep_mechanism_annotation"
    / f"local_paired_ipm_mem_deep_mechanism_errors_{DATE}.csv"
)
OUT = PROJECT / "results" / "11_enriched_clinical_error_analyses"


HIGH_CONF_ORDER = {
    "OprD-loss": 1,
    "High-confidence composite": 2,
    "AmpC-axis disruptive": 3,
    "carbapenemase": 4,
    "efflux disruptive": 5,
    "No high-confidence driver": 6,
}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def categorical_error_rates(sub: pd.DataFrame) -> dict[str, float | int]:
    n = int(len(sub))
    true_s = pd.to_numeric(sub["true_sns_int"], errors="coerce").eq(0) if "true_sns_int" in sub else pd.Series(False, index=sub.index)
    true_ns = pd.to_numeric(sub["true_sns_int"], errors="coerce").eq(1) if "true_sns_int" in sub else pd.Series(False, index=sub.index)
    vme = bool_series(sub["vme"]) if "vme" in sub else pd.Series(False, index=sub.index)
    me = bool_series(sub["me"]) if "me" in sub else pd.Series(False, index=sub.index)
    n_s = int(true_s.sum())
    n_ns = int(true_ns.sum())
    vme_n = int(vme.sum())
    me_n = int(me.sum())
    return {
        "n": n,
        "n_s": n_s,
        "n_ns": n_ns,
        "vme_n": vme_n,
        "me_n": me_n,
        "vme_rate": vme_n / n_ns if n_ns else math.nan,
        "me_rate": me_n / n_s if n_s else math.nan,
        "vme_all_row_rate": vme_n / n if n else math.nan,
        "me_all_row_rate": me_n / n if n else math.nan,
    }


def add_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prob_ns"] = pd.to_numeric(out["prob_ns"], errors="coerce")
    out["true_sns_int"] = pd.to_numeric(out["true_sns_int"], errors="coerce")
    out["pred_sns_int"] = pd.to_numeric(out["pred_sns_int"], errors="coerce")
    out["actual_mic_log2"] = pd.to_numeric(out["actual_mic_log2"], errors="coerce")
    out["pred_mic_log2"] = pd.to_numeric(out["pred_mic_log2"], errors="coerce")
    out["breakpoint_log2"] = pd.to_numeric(out["breakpoint_log2"], errors="coerce")
    out["signed_log2_error"] = pd.to_numeric(out["signed_log2_error"], errors="coerce")
    out["abs_log2_error"] = pd.to_numeric(out["abs_log2_error"], errors="coerce")
    out["actual_distance_to_breakpoint"] = out["actual_mic_log2"] - out["breakpoint_log2"]
    out["pred_distance_to_breakpoint"] = out["pred_mic_log2"] - out["breakpoint_log2"]
    out["breakpoint_zone"] = out["actual_distance_to_breakpoint"].map(classify_breakpoint_zone)
    out["censored_high_mic"] = out["actual_op"].astype(str).eq(">=") | out["interval_upper_log2"].astype(str).str.lower().eq("inf")
    out["prob_margin_from_gate"] = (out["prob_ns"] - 0.45).abs()
    out["prob_bin"] = pd.cut(
        out["prob_ns"],
        bins=[0, 0.10, 0.25, 0.45, 0.65, 0.85, 1.0],
        include_lowest=True,
        labels=["0-0.10", "0.10-0.25", "0.25-0.45", "0.45-0.65", "0.65-0.85", "0.85-1.00"],
    )
    out["susceptible_call_confidence"] = pd.cut(
        out["prob_ns"],
        bins=[0, 0.10, 0.25, 0.45],
        include_lowest=True,
        labels=["very_confident_S", "confident_S", "low_margin_S"],
    )
    out.loc[out["pred_sns_int"].ne(0), "susceptible_call_confidence"] = np.nan
    out["error_taxonomy"] = out.apply(classify_error_taxonomy, axis=1)
    return out


def classify_breakpoint_zone(distance: float) -> str:
    if pd.isna(distance):
        return "missing"
    if distance >= 2:
        return "far_NS_gt1_dilution_above_breakpoint"
    if distance > 0:
        return "near_NS_within_1_dilution"
    if distance == 0:
        return "at_breakpoint"
    if distance >= -1:
        return "near_S_within_1_dilution"
    return "far_S_gt1_dilution_below_breakpoint"


def classify_error_taxonomy(row: pd.Series) -> str:
    if bool(row.get("mic_ea_pm1", False)) and bool(row.get("sns_correct", False)):
        return "accurate_MIC_and_category"
    if bool(row.get("vme", False)):
        if row.get("actual_distance_to_breakpoint", 0) >= 2:
            return "dangerous_false_susceptible_far_from_breakpoint"
        return "dangerous_false_susceptible_near_breakpoint"
    if bool(row.get("me", False)):
        if row.get("actual_distance_to_breakpoint", 0) <= -2:
            return "false_resistant_far_from_breakpoint"
        return "false_resistant_near_breakpoint"
    if row.get("signed_log2_error", 0) < -2:
        return "large_MIC_underprediction_without_category_error"
    if row.get("signed_log2_error", 0) > 2:
        return "large_MIC_overprediction_without_category_error"
    if row.get("breakpoint_zone") in ["near_NS_within_1_dilution", "near_S_within_1_dilution", "at_breakpoint"]:
        return "near_breakpoint_MIC_drift"
    if row.get("clinical_rule") == "mechanism_unresolved_warning":
        return "mechanism_unresolved_instability"
    return "intermediate_MIC_error"


def calibration_by_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows = []
    for keys, sub in df.groupby(["drug", "mechanism_subtype_high_confidence", "prob_bin"], observed=False, dropna=False):
        drug, subtype, prob_bin = keys
        if len(sub) == 0:
            continue
        rates = categorical_error_rates(sub)
        bin_rows.append(
            {
                "drug": drug,
                "mechanism_subtype_high_confidence": subtype,
                "prob_bin": str(prob_bin),
                "n": int(len(sub)),
                "mean_prob_ns": float(sub["prob_ns"].mean()),
                "observed_ns_rate": float(sub["true_sns_int"].mean()),
                "brier_component": float(((sub["prob_ns"] - sub["true_sns_int"]) ** 2).mean()),
                "vme_rate": rates["vme_rate"],
                "me_rate": rates["me_rate"],
                "vme_all_row_rate": rates["vme_all_row_rate"],
                "me_all_row_rate": rates["me_all_row_rate"],
            }
        )
    bins = pd.DataFrame(bin_rows)

    summary_rows = []
    for keys, sub in df.groupby(["drug", "mechanism_subtype_high_confidence"], dropna=False):
        drug, subtype = keys
        local_bins = bins[
            bins["drug"].eq(drug)
            & bins["mechanism_subtype_high_confidence"].eq(subtype)
        ].copy()
        if local_bins.empty:
            ece = math.nan
        else:
            weights = local_bins["n"] / local_bins["n"].sum()
            ece = float((weights * (local_bins["mean_prob_ns"] - local_bins["observed_ns_rate"]).abs()).sum())
        rates = categorical_error_rates(sub)
        summary_rows.append(
            {
                "drug": drug,
                "mechanism_subtype_high_confidence": subtype,
                "n": int(len(sub)),
                "mean_prob_ns": float(sub["prob_ns"].mean()),
                "observed_ns_rate": float(sub["true_sns_int"].mean()),
                "brier_score": float(((sub["prob_ns"] - sub["true_sns_int"]) ** 2).mean()),
                "ece_by_bins": ece,
                "vme_rate": rates["vme_rate"],
                "me_rate": rates["me_rate"],
                "vme_all_row_rate": rates["vme_all_row_rate"],
                "me_all_row_rate": rates["me_all_row_rate"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["drug", "mechanism_subtype_high_confidence"])
    return bins, summary


def susceptible_call_risk(df: pd.DataFrame) -> pd.DataFrame:
    sus = df[df["pred_sns_int"].eq(0)].copy()
    rows = []
    for keys, sub in sus.groupby(
        ["drug", "mechanism_subtype_high_confidence", "susceptible_call_confidence"],
        observed=False,
        dropna=False,
    ):
        drug, subtype, conf = keys
        if len(sub) == 0:
            continue
        rows.append(
            {
                "drug": drug,
                "mechanism_subtype_high_confidence": subtype,
                "susceptible_call_confidence": str(conf),
                "n_predicted_susceptible": int(len(sub)),
                "false_susceptible_n": int(bool_series(sub["vme"]).sum()),
                "false_susceptible_rate": float(bool_series(sub["vme"]).mean()),
                "mean_prob_ns": float(sub["prob_ns"].mean()),
                "median_actual_distance_to_breakpoint": float(sub["actual_distance_to_breakpoint"].median()),
                "far_from_breakpoint_vme_n": int(
                    (bool_series(sub["vme"]) & sub["actual_distance_to_breakpoint"].ge(2)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["drug", "mechanism_subtype_high_confidence", "susceptible_call_confidence"])


def breakpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["drug", "breakpoint_zone", "mechanism_subtype_high_confidence"], dropna=False):
        drug, zone, subtype = keys
        rates = categorical_error_rates(sub)
        rows.append(
            {
                "drug": drug,
                "breakpoint_zone": zone,
                "mechanism_subtype_high_confidence": subtype,
                "n": int(len(sub)),
                "within_1_dilution": float(bool_series(sub["mic_ea_pm1"]).mean()),
                "mae_log2": float(sub["abs_log2_error"].mean()),
                "bias_log2": float(sub["signed_log2_error"].mean()),
                "vme_rate": rates["vme_rate"],
                "me_rate": rates["me_rate"],
                "vme_all_row_rate": rates["vme_all_row_rate"],
                "me_all_row_rate": rates["me_all_row_rate"],
                "large_underprediction_gt2_rate": float(sub["signed_log2_error"].lt(-2).mean()),
                "censored_high_mic_rate": float(sub["censored_high_mic"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["drug", "breakpoint_zone", "mechanism_subtype_high_confidence"])


def phenotype_combo_summary(df: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    isolate = df.drop_duplicates("genome_id").copy()
    subtype_counts = (
        pd.crosstab(
            isolate["ipm_mem_pair_group"],
            isolate["mechanism_subtype_high_confidence"],
        )
        .reset_index()
        .sort_values("ipm_mem_pair_group")
    )

    rows = []
    for keys, sub in paired.groupby(["ipm_mem_pair_group", "mechanism_subtype_high_confidence"], dropna=False):
        pair_group, subtype = keys
        rows.append(
            {
                "ipm_mem_pair_group": pair_group,
                "mechanism_subtype_high_confidence": subtype,
                "n_paired": int(len(sub)),
                "mean_abs_error_IPM": float(sub["abs_log2_error_IPM"].mean()),
                "mean_abs_error_MEM": float(sub["abs_log2_error_MEM"].mean()),
                "mean_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "median_MEM_minus_IPM_abs_error": float(sub["MEM_minus_IPM_abs_error"].median()),
                "MEM_worse_fraction": float(bool_series(sub["MEM_worse_than_IPM"]).mean()),
            }
        )
    combo_errors = pd.DataFrame(rows).sort_values(["ipm_mem_pair_group", "mechanism_subtype_high_confidence"])
    return subtype_counts, combo_errors


def taxonomy_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, sub in df.groupby(["drug", "error_taxonomy", "mechanism_subtype_high_confidence"], dropna=False):
        drug, tax, subtype = keys
        rows.append(
            {
                "drug": drug,
                "error_taxonomy": tax,
                "mechanism_subtype_high_confidence": subtype,
                "n": int(len(sub)),
                "mean_abs_error": float(sub["abs_log2_error"].mean()),
                "mean_signed_error": float(sub["signed_log2_error"].mean()),
                "vme_n": int(bool_series(sub["vme"]).sum()),
                "me_n": int(bool_series(sub["me"]).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["drug", "error_taxonomy", "mechanism_subtype_high_confidence"])
    detail_cols = [
        "drug",
        "genome_id",
        "ipm_mem_pair_group",
        "mechanism_subtype_high_confidence",
        "clinical_rule",
        "error_taxonomy",
        "breakpoint_zone",
        "actual_mic_log2",
        "breakpoint_log2",
        "pred_mic_log2",
        "prob_ns",
        "signed_log2_error",
        "abs_log2_error",
        "vme",
        "me",
        "oprd_deep_severity",
        "ampC_severity",
        "ampD_severity",
        "dacB_severity",
        "mexR_severity",
        "nalD_severity",
        "mexS_severity",
        "mexZ_severity",
        "nfxB_severity",
    ]
    details = df[[c for c in detail_cols if c in df.columns]].copy()
    details = details.sort_values(["drug", "error_taxonomy", "abs_log2_error"], ascending=[True, True, False])
    return summary, details


def safety_gate_rows(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    for scenario in [
        "no_gate",
        "low_margin_only_gate",
        "mechanism_warning_susceptible_gate",
        "mechanism_or_low_margin_gate",
        "mem_warning_susceptible_gate",
    ]:
        tmp = df.copy()
        tmp["gate_scenario"] = scenario
        tmp["withheld"] = False
        pred_s = tmp["pred_sns_int"].eq(0)
        warning_rule = tmp["clinical_rule"].isin(
            ["warning", "exploratory_warning", "mechanism_unresolved_warning"]
        )
        low_margin = pred_s & tmp["prob_ns"].ge(0.25)
        if scenario == "low_margin_only_gate":
            tmp["withheld"] = low_margin
        elif scenario == "mechanism_warning_susceptible_gate":
            tmp["withheld"] = pred_s & warning_rule
        elif scenario == "mechanism_or_low_margin_gate":
            tmp["withheld"] = (pred_s & warning_rule) | low_margin
        elif scenario == "mem_warning_susceptible_gate":
            tmp["withheld"] = pred_s & tmp["drug"].eq("MEM") & warning_rule
        tmp["released"] = ~tmp["withheld"]
        scenarios.append(tmp)
    return pd.concat(scenarios, ignore_index=True, sort=False)


def safety_gate_summary(gated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", "all", gated)]
    groups.extend(
        (scenario, drug, sub)
        for (scenario, drug), sub in gated.groupby(["gate_scenario", "drug"], dropna=False)
    )
    for scenario, drug, sub in groups:
        if scenario == "overall":
            for scen, scen_sub in gated.groupby("gate_scenario"):
                rows.append(summarize_gate_group(scen, "all", scen_sub))
            continue
        rows.append(summarize_gate_group(scenario, drug, sub))
    out = pd.DataFrame(rows)
    return out.sort_values(["gate_scenario", "drug"]).reset_index(drop=True)


def summarize_gate_group(scenario: str, drug: str, sub: pd.DataFrame) -> dict[str, float | int | str]:
    released = sub[sub["released"]]
    pred_s_released = released[released["pred_sns_int"].eq(0)]
    original_vme = int(bool_series(sub["vme"]).sum())
    released_vme = int(bool_series(released["vme"]).sum()) if len(released) else 0
    released_rates = categorical_error_rates(released)
    sub_rates = categorical_error_rates(sub)
    return {
        "gate_scenario": scenario,
        "drug": drug,
        "n_total": int(len(sub)),
        "n_released": int(len(released)),
        "coverage": float(len(released) / len(sub)) if len(sub) else math.nan,
        "n_withheld": int((~sub["released"]).sum()),
        "original_vme_n": original_vme,
        "released_vme_n": released_vme,
        "avoided_vme_n": original_vme - released_vme,
        "original_vme_rate": sub_rates["vme_rate"],
        "released_vme_rate": released_rates["vme_rate"],
        "released_vme_all_row_rate": released_rates["vme_all_row_rate"],
        "released_susceptible_calls": int(len(pred_s_released)),
        "released_susceptible_false_rate": float(bool_series(pred_s_released["vme"]).mean())
        if len(pred_s_released)
        else math.nan,
        "released_major_error_rate": released_rates["me_rate"],
        "released_major_error_all_row_rate": released_rates["me_all_row_rate"],
    }


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
    calibration_summary: pd.DataFrame,
    susceptible_risk: pd.DataFrame,
    breakpoint: pd.DataFrame,
    combo_errors: pd.DataFrame,
    taxonomy: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> None:
    report = OUT / f"enriched_clinical_error_analysis_report_{DATE}.md"
    mem_cal = calibration_summary[calibration_summary["drug"].eq("MEM")]
    mem_s_risk = susceptible_risk[susceptible_risk["drug"].eq("MEM")]
    far_vme = breakpoint[
        breakpoint["breakpoint_zone"].eq("far_NS_gt1_dilution_above_breakpoint")
    ]
    top_tax = taxonomy.sort_values(["drug", "n"], ascending=[True, False]).groupby("drug").head(8)
    gate_focus = gate_summary[
        gate_summary["gate_scenario"].isin(
            ["no_gate", "mechanism_warning_susceptible_gate", "mechanism_or_low_margin_gate"]
        )
        & gate_summary["drug"].isin(["all", "MEM"])
    ]
    lines = [
        "# Enriched Clinical Error Analyses",
        "",
        f"Date: {DATE}",
        "",
        "## MEM Calibration Summary",
        "",
        markdown_table(mem_cal),
        "",
        "## MEM Susceptible-Call Risk",
        "",
        markdown_table(mem_s_risk),
        "",
        "## Far-From-Breakpoint Error Summary",
        "",
        markdown_table(far_vme),
        "",
        "## Phenotype Combination Error Context",
        "",
        markdown_table(combo_errors),
        "",
        "## Error Taxonomy Snapshot",
        "",
        markdown_table(top_tax),
        "",
        "## Safety Gate Evaluation",
        "",
        markdown_table(gate_focus),
        "",
        "## Interpretation",
        "",
        "- Calibration and susceptible-call risk determine whether the model recognizes uncertainty before a false-susceptible call.",
        "- Breakpoint-zone analyses distinguish near-breakpoint drift from far-from-breakpoint dangerous underprediction.",
        "- Phenotype-combination analyses connect IPM/MEM discordance to mechanism-dependent error profiles.",
        "- Safety-gate analyses estimate how many VMEs can be avoided by withholding susceptible calls in warning strata.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = add_common_columns(pd.read_csv(PRED_ROWS, sep="\t", low_memory=False))
    paired = pd.read_csv(DEEP_PAIRED, low_memory=False)

    cal_bins, cal_summary = calibration_by_group(rows)
    sus_risk = susceptible_call_risk(rows)
    bp_summary = breakpoint_summary(rows)
    combo_counts, combo_errors = phenotype_combo_summary(rows, paired)
    tax_summary, tax_rows = taxonomy_summary(rows)
    gated = safety_gate_rows(rows)
    gate_summary = safety_gate_summary(gated)

    rows.to_csv(OUT / f"prediction_rows_with_error_taxonomy_{DATE}.tsv", sep="\t", index=False)
    cal_bins.to_csv(OUT / f"calibration_by_probability_bin_{DATE}.csv", index=False)
    cal_summary.to_csv(OUT / f"calibration_summary_by_high_confidence_subtype_{DATE}.csv", index=False)
    sus_risk.to_csv(OUT / f"susceptible_call_risk_by_confidence_{DATE}.csv", index=False)
    bp_summary.to_csv(OUT / f"breakpoint_zone_error_summary_{DATE}.csv", index=False)
    combo_counts.to_csv(OUT / f"phenotype_combo_high_confidence_counts_{DATE}.csv", index=False)
    combo_errors.to_csv(OUT / f"phenotype_combo_paired_error_summary_{DATE}.csv", index=False)
    tax_summary.to_csv(OUT / f"error_taxonomy_summary_{DATE}.csv", index=False)
    tax_rows.to_csv(OUT / f"error_taxonomy_rows_{DATE}.tsv", sep="\t", index=False)
    gated.to_csv(OUT / f"safety_gate_prediction_rows_{DATE}.tsv", sep="\t", index=False)
    gate_summary.to_csv(OUT / f"safety_gate_evaluation_summary_{DATE}.csv", index=False)
    write_report(cal_summary, sus_risk, bp_summary, combo_errors, tax_summary, gate_summary)


if __name__ == "__main__":
    main()
