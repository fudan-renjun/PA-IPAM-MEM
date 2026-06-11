#!/usr/bin/env python3
"""Export manuscript figure-panel data for IPM/MEM predictability analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"

METADATA = PROJECT / "data" / "metadata" / "clean_metadata.tsv"
PREDICTION_ERRORS = (
    PROJECT
    / "results"
    / "02_unified_ipm_mem_prediction"
    / f"ipm_mem_unified_prediction_errors_{DATE}.tsv"
)
MECH_ERRORS = (
    PROJECT
    / "results"
    / "03_mechanism_predictability"
    / f"ipm_mem_prediction_errors_with_mechanism_{DATE}.tsv"
)
MECH_EVIDENCE = (
    PROJECT
    / "results"
    / "03_mechanism_predictability"
    / f"mechanism_evidence_strict_first_pass_{DATE}.tsv"
)
ARCHETYPES = (
    PROJECT
    / "results"
    / "04_predictability_statistics"
    / f"local_predictability_archetypes_{DATE}.csv"
)
PAIRED = (
    PROJECT
    / "results"
    / "06_st_sensitivity"
    / f"local_paired_ipm_mem_errors_with_st_{DATE}.csv"
)
ST_COUNTS = PROJECT / "results" / "06_st_sensitivity" / f"local_st_counts_{DATE}.csv"
ST_LEAVE_ONE = (
    PROJECT
    / "results"
    / "06_st_sensitivity"
    / f"local_leave_one_st_out_sensitivity_{DATE}.csv"
)
ST_MODELS = PROJECT / "results" / "06_st_sensitivity" / f"local_st_adjusted_models_{DATE}.csv"
ST_PAIRED_ERROR_BY_ST = (
    PROJECT
    / "results"
    / "06_st_sensitivity"
    / f"local_paired_error_difference_by_st_{DATE}.csv"
)
DEEP_EVIDENCE = (
    PROJECT
    / "results"
    / "08_deep_mechanism_annotation"
    / f"local_deep_mechanism_evidence_{DATE}.tsv"
)
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
CLINICAL_RULE_TABLE = (
    PROJECT
    / "results"
    / "10_clinical_warning_rules"
    / f"clinical_warning_rule_table_{DATE}.csv"
)
CLINICAL_RULE_PERF = (
    PROJECT
    / "results"
    / "10_clinical_warning_rules"
    / f"clinical_rule_level_performance_{DATE}.csv"
)
MEM_WARNING_ENRICHMENT = (
    PROJECT
    / "results"
    / "10_clinical_warning_rules"
    / f"mem_warning_enrichment_tests_{DATE}.csv"
)
FALSE_SUSCEPTIBLE_CASES = (
    PROJECT
    / "results"
    / "10_clinical_warning_rules"
    / f"false_susceptible_cases_by_warning_rule_{DATE}.csv"
)
CALIBRATION_SUMMARY = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"calibration_summary_by_high_confidence_subtype_{DATE}.csv"
)
SUSCEPTIBLE_CALL_RISK = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"susceptible_call_risk_by_confidence_{DATE}.csv"
)
BREAKPOINT_ZONE_SUMMARY = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"breakpoint_zone_error_summary_{DATE}.csv"
)
PHENOTYPE_COMBO_ERRORS = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"phenotype_combo_paired_error_summary_{DATE}.csv"
)
ERROR_TAXONOMY_SUMMARY = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"error_taxonomy_summary_{DATE}.csv"
)
SAFETY_GATE_SUMMARY = (
    PROJECT
    / "results"
    / "11_enriched_clinical_error_analyses"
    / f"safety_gate_evaluation_summary_{DATE}.csv"
)
TRAINING_APPARENT_PREDICTIONS = (
    PROJECT
    / "models"
    / "ipm_mem_unified_public_only"
    / f"ipm_mem_unified_public_only_training_apparent_predictions_{DATE}.csv"
)

OUT = PROJECT / "results" / "07_figure_data"


DRUG_ORDER = {"IPM": 1, "MEM": 2}
SUBTYPE_ORDER = {
    "OprD-loss": 1,
    "Composite": 2,
    "Carbapenemase-mediated": 3,
    "Efflux-associated genotype": 4,
    "AmpC-associated genotype": 5,
    "No strict mechanism": 6,
    "Mechanism missing": 7,
}
PAIR_GROUP_ORDER = {
    "IPM-S/MEM-S": 1,
    "IPM-R/MEM-S_or_I": 2,
    "IPM-R/MEM-R": 3,
    "IPM-S/MEM-R": 4,
    "IPM/MEM incomplete": 5,
}
HIGH_CONF_ORDER = {
    "OprD-loss": 1,
    "High-confidence composite": 2,
    "AmpC-axis disruptive": 3,
    "carbapenemase": 4,
    "efflux disruptive": 5,
    "No high-confidence driver": 6,
}


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = OUT / filename
    frame.to_csv(path, index=False)
    return path


def pct(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(value) * 100.0


def categorical_error_rates(sub: pd.DataFrame) -> pd.Series:
    n = int(len(sub))
    n_s = int((sub["true_sns_int"] == 0).sum())
    n_ns = int((sub["true_sns_int"] == 1).sum())
    false_susceptible_n = int(sub["vme"].astype(bool).sum())
    false_resistant_n = int(sub["me"].astype(bool).sum())
    return pd.Series(
        {
            "n": n,
            "n_s": n_s,
            "n_ns": n_ns,
            "false_susceptible_n": false_susceptible_n,
            "false_resistant_n": false_resistant_n,
            "very_major_error_rate": false_susceptible_n / n_ns if n_ns else np.nan,
            "major_error_rate": false_resistant_n / n_s if n_s else np.nan,
            "false_susceptible_all_row_rate": false_susceptible_n / n if n else np.nan,
            "false_resistant_all_row_rate": false_resistant_n / n if n else np.nan,
        }
    )


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "metadata": pd.read_csv(METADATA, sep="\t", dtype=str).fillna(""),
        "training_apparent_predictions": pd.read_csv(TRAINING_APPARENT_PREDICTIONS, low_memory=False),
        "pred": pd.read_csv(PREDICTION_ERRORS, sep="\t", low_memory=False),
        "mech": pd.read_csv(MECH_ERRORS, sep="\t", low_memory=False),
        "evidence": pd.read_csv(MECH_EVIDENCE, sep="\t", low_memory=False),
        "archetypes": pd.read_csv(ARCHETYPES),
        "paired": pd.read_csv(PAIRED),
        "st_counts": pd.read_csv(ST_COUNTS),
        "st_leave_one": pd.read_csv(ST_LEAVE_ONE),
        "st_models": pd.read_csv(ST_MODELS),
        "st_paired_error_by_st": pd.read_csv(ST_PAIRED_ERROR_BY_ST),
        "deep_evidence": pd.read_csv(DEEP_EVIDENCE, sep="\t", low_memory=False),
        "deep_errors": pd.read_csv(DEEP_ERRORS, sep="\t", low_memory=False),
        "deep_paired": pd.read_csv(DEEP_PAIRED, low_memory=False),
        "clinical_rule_table": pd.read_csv(CLINICAL_RULE_TABLE),
        "clinical_rule_perf": pd.read_csv(CLINICAL_RULE_PERF),
        "mem_warning_enrichment": pd.read_csv(MEM_WARNING_ENRICHMENT),
        "false_susceptible_cases": pd.read_csv(FALSE_SUSCEPTIBLE_CASES),
        "calibration_summary": pd.read_csv(CALIBRATION_SUMMARY),
        "susceptible_call_risk": pd.read_csv(SUSCEPTIBLE_CALL_RISK),
        "breakpoint_zone_summary": pd.read_csv(BREAKPOINT_ZONE_SUMMARY),
        "phenotype_combo_errors": pd.read_csv(PHENOTYPE_COMBO_ERRORS),
        "error_taxonomy_summary": pd.read_csv(ERROR_TAXONOMY_SUMMARY),
        "safety_gate_summary": pd.read_csv(SAFETY_GATE_SUMMARY),
    }


def fig1_dataset_and_problem(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    metadata = data["metadata"]
    training = data["training_apparent_predictions"]
    pred = data["pred"]

    local_meta = metadata[metadata["data_origin"].eq("local_clinical")].copy()
    local_readiness = (
        local_meta.groupby("data_origin", dropna=False)
        .agg(
            isolates=("isolate_id", "size"),
            assemblies_present=("assembly_status", lambda s: int((s == "present").sum())),
            ipm_mic_and_sir=("IPM_phenotype_status", lambda s: int((s == "mic_and_sir").sum())),
            mem_mic_and_sir=("MEM_phenotype_status", lambda s: int((s == "mic_and_sir").sum())),
            ipm_sir_only=("IPM_phenotype_status", lambda s: int((s == "sir_only").sum())),
            mem_sir_only=("MEM_phenotype_status", lambda s: int((s == "sir_only").sum())),
        )
        .reset_index()
    )
    local_readiness["cohort_role"] = "locked_local_validation"
    local_readiness["display_label"] = "Local clinical validation"

    train_by_drug = training.groupby("drug")["genome_id"].nunique()
    train_readiness = pd.DataFrame(
        [
            {
                "data_origin": "public_training_model_development",
                "isolates": int(training["genome_id"].nunique()),
                "assemblies_present": int(training["genome_id"].nunique()),
                "ipm_mic_and_sir": int(train_by_drug.get("IPM", 0)),
                "mem_mic_and_sir": int(train_by_drug.get("MEM", 0)),
                "ipm_sir_only": 0,
                "mem_sir_only": 0,
                "cohort_role": "model_development_train_only",
                "display_label": "Public training",
            }
        ]
    )

    public_external = pred[pred["cohort"].eq("locked_public_external")].copy()
    external_by_drug = public_external.groupby("drug")["genome_id"].nunique()
    public_external_readiness = pd.DataFrame(
        [
            {
                "data_origin": "locked_public_external",
                "isolates": int(public_external["genome_id"].nunique()),
                "assemblies_present": int(public_external["genome_id"].nunique()),
                "ipm_mic_and_sir": int(external_by_drug.get("IPM", 0)),
                "mem_mic_and_sir": int(external_by_drug.get("MEM", 0)),
                "ipm_sir_only": 0,
                "mem_sir_only": 0,
                "cohort_role": "locked_public_external_validation",
                "display_label": "Locked public external",
            }
        ]
    )

    readiness = pd.concat(
        [train_readiness, public_external_readiness, local_readiness],
        ignore_index=True,
    )
    readiness["cohort_order"] = readiness["data_origin"].map(
        {
            "public_training_model_development": 1,
            "locked_public_external": 2,
            "local_clinical": 3,
        }
    )
    readiness = readiness.sort_values("cohort_order")

    perf_base = (
        pred.groupby(["cohort", "drug"], dropna=False)
        .agg(
            within_1_dilution=("mic_ea_pm1", "mean"),
            mae_log2=("abs_log2_error", "mean"),
            bias_log2=("signed_log2_error", "mean"),
            categorical_agreement=("sns_correct", "mean"),
        )
        .reset_index()
    )
    perf_rates = pred.groupby(["cohort", "drug"], dropna=False).apply(
        categorical_error_rates, include_groups=False
    ).reset_index()
    perf = perf_base.merge(perf_rates, on=["cohort", "drug"], how="left")
    for col in [
        "within_1_dilution",
        "categorical_agreement",
        "very_major_error_rate",
        "major_error_rate",
    ]:
        perf[f"{col}_pct"] = perf[col].map(pct)
    perf["drug_order"] = perf["drug"].map(DRUG_ORDER)
    perf = perf.sort_values(["cohort", "drug_order"])

    local = pred[pred["cohort"].eq("locked_local_validation")].copy()
    local_isolate = local.drop_duplicates("genome_id")
    pair_groups = (
        local_isolate["ipm_mem_pair_group"]
        .value_counts(dropna=False)
        .rename_axis("ipm_mem_pair_group")
        .reset_index(name="isolates")
    )
    pair_groups["pair_group_order"] = pair_groups["ipm_mem_pair_group"].map(PAIR_GROUP_ORDER)
    pair_groups = pair_groups.sort_values(["pair_group_order", "ipm_mem_pair_group"])

    return {
        "fig1a_dataset_readiness": write_csv(
            readiness, f"fig1a_dataset_readiness_{DATE}.csv"
        ),
        "fig1b_unified_ipm_mem_performance": write_csv(
            perf, f"fig1b_unified_ipm_mem_performance_{DATE}.csv"
        ),
        "fig1c_local_ipm_mem_pair_groups": write_csv(
            pair_groups, f"fig1c_local_ipm_mem_pair_groups_{DATE}.csv"
        ),
    }


def fig2_mechanism_landscape(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    mech = data["mech"]
    evidence = data["evidence"]
    deep_evidence = data["deep_evidence"]
    local_evidence = evidence[evidence["cohort"].eq("locked_local_validation")].copy()

    subtype_counts = (
        local_evidence.groupby("mechanism_subtype_strict", dropna=False)
        .agg(
            isolates=("genome_id", "size"),
            oprd_severe_loss=("oprd_severe_loss", "sum"),
            acquired_carbapenemase_strict=("acquired_carbapenemase_strict", "sum"),
            ampc_associated_strict=("ampc_associated_strict", "sum"),
            efflux_regulator_strict=("efflux_regulator_strict", "sum"),
        )
        .reset_index()
    )
    subtype_counts["subtype_order"] = subtype_counts["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    subtype_counts = subtype_counts.sort_values(["subtype_order", "mechanism_subtype_strict"])

    local_isolate = (
        mech[mech["cohort"].eq("locked_local_validation")]
        .drop_duplicates("genome_id")
        .copy()
    )
    subtype_by_pair = pd.crosstab(
        local_isolate["ipm_mem_pair_group"],
        local_isolate["mechanism_subtype_strict"],
    ).reset_index()
    subtype_by_pair["pair_group_order"] = subtype_by_pair["ipm_mem_pair_group"].map(PAIR_GROUP_ORDER)
    subtype_by_pair = subtype_by_pair.sort_values(["pair_group_order", "ipm_mem_pair_group"])

    evidence_cols = [
        "cohort",
        "genome_id",
        "mechanism_subtype_strict",
        "mechanism_multilabel",
        "oprd_severe_loss",
        "oprd_mutated_broad",
        "acquired_carbapenemase_strict",
        "acquired_carbapenemase_genes",
        "ampc_associated_strict",
        "pdc_present_broad",
        "efflux_regulator_strict",
        "efflux_regulator_strict_count",
        "efflux_regulator_broad",
        "mechanism_reason",
    ]
    local_matrix = local_evidence[[c for c in evidence_cols if c in local_evidence.columns]].copy()
    local_matrix["subtype_order"] = local_matrix["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    local_matrix = local_matrix.sort_values(["subtype_order", "genome_id"])

    high_conf_counts = (
        deep_evidence.groupby("mechanism_subtype_high_confidence", dropna=False)
        .agg(
            isolates=("genome_id", "size"),
            oprd_severe_loss=("oprd_severe_loss", "sum"),
            oprd_deep_disruptive=("oprd_deep_disruptive", "sum"),
            acquired_carbapenemase_strict=("acquired_carbapenemase_strict", "sum"),
            ampc_core_driver_disruptive=("ampc_core_driver_disruptive_any", "sum"),
            efflux_strict_driver_disruptive=("efflux_strict_driver_disruptive_any", "sum"),
        )
        .reset_index()
    )
    high_conf_counts["high_conf_order"] = high_conf_counts[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    high_conf_counts = high_conf_counts.sort_values(
        ["high_conf_order", "mechanism_subtype_high_confidence"]
    )

    burden_cols = [
        "oprd_deep_disruptive",
        "efflux_strict_driver_disruptive_any",
        "ampc_core_driver_disruptive_any",
        "acquired_carbapenemase_strict",
        "oprd_deep_non_syn_or_disruptive",
        "efflux_strict_driver_non_syn_any",
        "ampc_core_driver_non_syn_any",
        "ampc_coding_non_syn_or_regulatory",
    ]
    high_conf_matrix_cols = [
        "genome_id",
        "mechanism_subtype_strict",
        "mechanism_subtype_high_confidence",
        "mechanism_subtype_refined",
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
    ] + burden_cols
    high_conf_matrix = deep_evidence[
        [c for c in high_conf_matrix_cols if c in deep_evidence.columns]
    ].copy()
    high_conf_matrix["high_conf_order"] = high_conf_matrix[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    high_conf_matrix = high_conf_matrix.sort_values(["high_conf_order", "genome_id"])

    return {
        "fig2a_local_strict_subtype_counts": write_csv(
            subtype_counts, f"fig2a_local_strict_subtype_counts_{DATE}.csv"
        ),
        "fig2b_local_pair_group_by_subtype": write_csv(
            subtype_by_pair, f"fig2b_local_pair_group_by_subtype_{DATE}.csv"
        ),
        "fig2c_local_mechanism_evidence_matrix": write_csv(
            local_matrix, f"fig2c_local_mechanism_evidence_matrix_{DATE}.csv"
        ),
        "fig2d_local_high_confidence_subtype_counts": write_csv(
            high_conf_counts,
            f"fig2d_local_high_confidence_subtype_counts_{DATE}.csv",
        ),
        "fig2e_local_high_confidence_evidence_matrix": write_csv(
            high_conf_matrix,
            f"fig2e_local_high_confidence_evidence_matrix_{DATE}.csv",
        ),
    }


def fig3_subtype_predictability(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    mech = data["mech"]
    archetypes = data["archetypes"].copy()
    deep_errors = data["deep_errors"]
    local = mech[mech["cohort"].eq("locked_local_validation")].copy()

    subtype_perf = (
        local.groupby(["drug", "mechanism_subtype_strict"], dropna=False)
        .agg(
            n=("genome_id", "size"),
            n_s=("true_sns_int", lambda s: int((s == 0).sum())),
            n_ns=("true_sns_int", lambda s: int((s == 1).sum())),
            within_1_dilution=("mic_ea_pm1", "mean"),
            mae_log2=("abs_log2_error", "mean"),
            median_abs_error=("abs_log2_error", "median"),
            bias_log2=("signed_log2_error", "mean"),
            large_error_gt2=("abs_log2_error", lambda s: float((s > 2).mean())),
            categorical_agreement=("sns_correct", "mean"),
        )
        .reset_index()
    )
    subtype_rates = local.groupby(["drug", "mechanism_subtype_strict"], dropna=False).apply(
        categorical_error_rates, include_groups=False
    ).reset_index()
    subtype_rates = subtype_rates.drop(columns=["n", "n_s", "n_ns"], errors="ignore")
    subtype_perf = subtype_perf.merge(
        subtype_rates,
        on=["drug", "mechanism_subtype_strict"],
        how="left",
    )
    subtype_perf = subtype_perf.merge(
        archetypes[["drug", "mechanism_subtype_strict", "predictability_archetype"]],
        on=["drug", "mechanism_subtype_strict"],
        how="left",
    )
    subtype_perf["drug_order"] = subtype_perf["drug"].map(DRUG_ORDER)
    subtype_perf["subtype_order"] = subtype_perf["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    subtype_perf = subtype_perf.sort_values(["drug_order", "subtype_order"])

    error_points = local[
        [
            "drug",
            "genome_id",
            "mechanism_subtype_strict",
            "ipm_mem_pair_group",
            "actual_mic_log2",
            "pred_mic_log2",
            "signed_log2_error",
            "abs_log2_error",
            "mic_ea_pm1",
            "sns_correct",
        ]
    ].copy()
    error_points["drug_order"] = error_points["drug"].map(DRUG_ORDER)
    error_points["subtype_order"] = error_points["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    error_points["pair_group_order"] = error_points["ipm_mem_pair_group"].map(PAIR_GROUP_ORDER)
    error_points = error_points.sort_values(["drug_order", "subtype_order", "genome_id"])

    high_conf_perf = (
        deep_errors.groupby(["drug", "mechanism_subtype_high_confidence"], dropna=False)
        .agg(
            n=("genome_id", "size"),
            n_s=("true_sns_int", lambda s: int((s == 0).sum())),
            n_ns=("true_sns_int", lambda s: int((s == 1).sum())),
            within_1_dilution=("mic_ea_pm1", "mean"),
            mae_log2=("abs_log2_error", "mean"),
            median_abs_error=("abs_log2_error", "median"),
            bias_log2=("signed_log2_error", "mean"),
            large_error_gt2=("abs_log2_error", lambda s: float((s > 2).mean())),
            categorical_agreement=("sns_correct", "mean"),
        )
        .reset_index()
    )
    high_conf_rates = deep_errors.groupby(
        ["drug", "mechanism_subtype_high_confidence"], dropna=False
    ).apply(
        categorical_error_rates, include_groups=False
    ).reset_index()
    high_conf_rates = high_conf_rates.drop(columns=["n", "n_s", "n_ns"], errors="ignore")
    high_conf_perf = high_conf_perf.merge(
        high_conf_rates,
        on=["drug", "mechanism_subtype_high_confidence"],
        how="left",
    )
    high_conf_perf["drug_order"] = high_conf_perf["drug"].map(DRUG_ORDER)
    high_conf_perf["high_conf_order"] = high_conf_perf[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    high_conf_perf = high_conf_perf.sort_values(["drug_order", "high_conf_order"])

    high_conf_points = deep_errors[
        [
            "drug",
            "genome_id",
            "mechanism_subtype_high_confidence",
            "mechanism_subtype_strict",
            "ipm_mem_pair_group",
            "actual_mic_log2",
            "pred_mic_log2",
            "signed_log2_error",
            "abs_log2_error",
            "mic_ea_pm1",
            "sns_correct",
            "oprd_deep_severity",
            "ampC_severity",
            "ampD_severity",
            "dacB_severity",
            "efflux_strict_driver_disruptive_any",
            "ampc_core_driver_disruptive_any",
        ]
    ].copy()
    high_conf_points["drug_order"] = high_conf_points["drug"].map(DRUG_ORDER)
    high_conf_points["high_conf_order"] = high_conf_points[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    high_conf_points["pair_group_order"] = high_conf_points["ipm_mem_pair_group"].map(
        PAIR_GROUP_ORDER
    )
    high_conf_points = high_conf_points.sort_values(
        ["drug_order", "high_conf_order", "genome_id"]
    )

    mic_shift_summary = (
        high_conf_points.groupby(["drug", "mechanism_subtype_high_confidence"], dropna=False)
        .agg(
            n=("genome_id", "size"),
            median_actual_mic_log2=("actual_mic_log2", "median"),
            median_pred_mic_log2=("pred_mic_log2", "median"),
            mean_actual_mic_log2=("actual_mic_log2", "mean"),
            mean_pred_mic_log2=("pred_mic_log2", "mean"),
            median_signed_log2_error=("signed_log2_error", "median"),
            mean_signed_log2_error=("signed_log2_error", "mean"),
            mean_abs_log2_error=("abs_log2_error", "mean"),
        )
        .reset_index()
    )
    mic_shift_summary["median_actual_mic_mg_l"] = 2 ** mic_shift_summary["median_actual_mic_log2"]
    mic_shift_summary["median_pred_mic_mg_l"] = 2 ** mic_shift_summary["median_pred_mic_log2"]
    mic_shift_summary["mean_actual_mic_mg_l"] = 2 ** mic_shift_summary["mean_actual_mic_log2"]
    mic_shift_summary["mean_pred_mic_mg_l"] = 2 ** mic_shift_summary["mean_pred_mic_log2"]
    mic_shift_summary["drug_order"] = mic_shift_summary["drug"].map(DRUG_ORDER)
    mic_shift_summary["high_conf_order"] = mic_shift_summary[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    mic_shift_summary = mic_shift_summary.sort_values(["drug_order", "high_conf_order"])

    def direction_summary(sub: pd.DataFrame) -> pd.Series:
        n = len(sub)
        large_under = sub["signed_log2_error"].lt(-2)
        large_over = sub["signed_log2_error"].gt(2)
        within_two = ~(large_under | large_over)
        return pd.Series(
            {
                "n": int(n),
                "large_underprediction_n": int(large_under.sum()),
                "large_overprediction_n": int(large_over.sum()),
                "within_2_log2_error_n": int(within_two.sum()),
                "large_underprediction_rate": float(large_under.mean()) if n else np.nan,
                "large_overprediction_rate": float(large_over.mean()) if n else np.nan,
                "within_2_log2_error_rate": float(within_two.mean()) if n else np.nan,
            }
        )

    error_direction_summary = (
        high_conf_points.groupby(["drug", "mechanism_subtype_high_confidence"], dropna=False)
        .apply(direction_summary, include_groups=False)
        .reset_index()
    )
    for col in [
        "large_underprediction_rate",
        "large_overprediction_rate",
        "within_2_log2_error_rate",
    ]:
        error_direction_summary[f"{col}_pct"] = error_direction_summary[col] * 100.0
    error_direction_summary["drug_order"] = error_direction_summary["drug"].map(DRUG_ORDER)
    error_direction_summary["high_conf_order"] = error_direction_summary[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    error_direction_summary = error_direction_summary.sort_values(["drug_order", "high_conf_order"])

    return {
        "fig3a_subtype_predictability_metrics": write_csv(
            subtype_perf, f"fig3a_subtype_predictability_metrics_{DATE}.csv"
        ),
        "fig3b_local_error_points_by_subtype": write_csv(
            error_points, f"fig3b_local_error_points_by_subtype_{DATE}.csv"
        ),
        "fig3c_high_confidence_predictability_metrics": write_csv(
            high_conf_perf,
            f"fig3c_high_confidence_predictability_metrics_{DATE}.csv",
        ),
        "fig3d_high_confidence_error_points": write_csv(
            high_conf_points,
            f"fig3d_high_confidence_error_points_{DATE}.csv",
        ),
        "fig3e_high_confidence_observed_predicted_mic_shift": write_csv(
            mic_shift_summary,
            f"fig3e_high_confidence_observed_predicted_mic_shift_{DATE}.csv",
        ),
        "fig3f_high_confidence_large_error_direction": write_csv(
            error_direction_summary,
            f"fig3f_high_confidence_large_error_direction_{DATE}.csv",
        ),
    }


def fig4_paired_and_st(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paired = data["paired"].copy()
    deep_paired = data["deep_paired"].copy()
    st_counts = data["st_counts"].copy()
    leave_one = data["st_leave_one"].copy()
    models = data["st_models"].copy()
    st_by_error = data["st_paired_error_by_st"].copy()

    paired["subtype_order"] = paired["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    paired["pair_group_order"] = paired["ipm_mem_pair_group"].map(PAIR_GROUP_ORDER)
    paired = paired.sort_values(["subtype_order", "pair_group_order", "genome_id"])

    paired_summary = (
        paired.groupby(["mechanism_subtype_strict"], dropna=False)
        .agg(
            n_paired=("genome_id", "size"),
            median_MEM_minus_IPM_abs_error=("MEM_minus_IPM_abs_error", "median"),
            mean_MEM_minus_IPM_abs_error=("MEM_minus_IPM_abs_error", "mean"),
            MEM_worse_fraction=("MEM_worse_than_IPM", "mean"),
            MEM_much_worse_gt1_fraction=("MEM_much_worse_gt1", "mean"),
        )
        .reset_index()
    )
    paired_summary["subtype_order"] = paired_summary["mechanism_subtype_strict"].map(SUBTYPE_ORDER)
    paired_summary = paired_summary.sort_values(["subtype_order", "mechanism_subtype_strict"])

    deep_paired["high_conf_order"] = deep_paired["mechanism_subtype_high_confidence"].map(
        HIGH_CONF_ORDER
    )
    deep_paired["pair_group_order"] = deep_paired["ipm_mem_pair_group"].map(PAIR_GROUP_ORDER)
    deep_paired = deep_paired.sort_values(["high_conf_order", "pair_group_order", "genome_id"])

    high_conf_paired_summary = (
        deep_paired.groupby(["mechanism_subtype_high_confidence"], dropna=False)
        .agg(
            n_paired=("genome_id", "size"),
            median_MEM_minus_IPM_abs_error=("MEM_minus_IPM_abs_error", "median"),
            mean_MEM_minus_IPM_abs_error=("MEM_minus_IPM_abs_error", "mean"),
            MEM_worse_fraction=("MEM_worse_than_IPM", "mean"),
            mean_abs_error_IPM=("abs_log2_error_IPM", "mean"),
            mean_abs_error_MEM=("abs_log2_error_MEM", "mean"),
        )
        .reset_index()
    )
    high_conf_paired_summary["high_conf_order"] = high_conf_paired_summary[
        "mechanism_subtype_high_confidence"
    ].map(HIGH_CONF_ORDER)
    high_conf_paired_summary = high_conf_paired_summary.sort_values(
        ["high_conf_order", "mechanism_subtype_high_confidence"]
    )

    st_consistency = st_by_error[st_by_error["n_paired"].ge(3)].copy()
    st_consistency["ST_plot_label"] = st_consistency["ST_label"].astype(str)
    st_consistency["MEM_worse_percent"] = st_consistency["MEM_worse_fraction"] * 100.0
    st_consistency["mean_MEM_minus_IPM_abs_error_positive_supports_MEM_worse"] = (
        st_consistency["mean_MEM_minus_IPM_abs_error"]
    )
    st_consistency["median_MEM_minus_IPM_abs_error_positive_supports_MEM_worse"] = (
        st_consistency["median_MEM_minus_IPM_abs_error"]
    )
    st_consistency["point_size_suggested"] = (
        35 + 10 * np.sqrt(st_consistency["n_paired"].astype(float))
    )
    st_consistency["plot_order"] = (
        st_consistency["mean_MEM_minus_IPM_abs_error"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    st_consistency = st_consistency.sort_values(
        ["mean_MEM_minus_IPM_abs_error", "n_paired"],
        ascending=[False, False],
    )

    return {
        "fig4a_paired_ipm_mem_error_points": write_csv(
            paired, f"fig4a_paired_ipm_mem_error_points_{DATE}.csv"
        ),
        "fig4b_paired_error_difference_by_subtype": write_csv(
            paired_summary, f"fig4b_paired_error_difference_by_subtype_{DATE}.csv"
        ),
        "fig4c_local_st_counts": write_csv(
            st_counts, f"fig4c_local_st_counts_{DATE}.csv"
        ),
        "fig4d_leave_one_st_out_sensitivity": write_csv(
            leave_one, f"fig4d_leave_one_st_out_sensitivity_{DATE}.csv"
        ),
        "fig4e_st_adjusted_models": write_csv(
            models, f"fig4e_st_adjusted_models_{DATE}.csv"
        ),
        "fig4h_st_level_paired_error_consistency": write_csv(
            st_consistency,
            f"fig4h_st_level_paired_error_consistency_{DATE}.csv",
        ),
        "fig4f_high_confidence_paired_error_points": write_csv(
            deep_paired,
            f"fig4f_high_confidence_paired_error_points_{DATE}.csv",
        ),
        "fig4g_high_confidence_paired_error_summary": write_csv(
            high_conf_paired_summary,
            f"fig4g_high_confidence_paired_error_summary_{DATE}.csv",
        ),
    }


def fig5_clinical_warning_rules(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    rule_table = data["clinical_rule_table"].copy()
    rule_perf = data["clinical_rule_perf"].copy()
    mem_enrichment = data["mem_warning_enrichment"].copy()
    vme_cases = data["false_susceptible_cases"].copy()

    rule_table = rule_table.sort_values(["drug_order", "high_conf_order"])
    rule_perf = rule_perf.sort_values(["drug", "rule_order"])
    if {
        "flagged_large_underprediction_gt2_rate",
        "unflagged_large_underprediction_gt2_rate",
    }.issubset(mem_enrichment.columns):
        mem_enrichment["large_underprediction_enrichment"] = (
            mem_enrichment["flagged_large_underprediction_gt2_rate"]
            - mem_enrichment["unflagged_large_underprediction_gt2_rate"]
        )
    if {"flagged_vme_rate", "unflagged_vme_rate"}.issubset(mem_enrichment.columns):
        mem_enrichment["standard_vme_difference"] = (
            mem_enrichment["flagged_vme_rate"] - mem_enrichment["unflagged_vme_rate"]
        )
    mem_enrichment = mem_enrichment.sort_values(
        ["fisher_large_underprediction_p", "flag"],
        na_position="last",
    )
    vme_cases["drug_order"] = vme_cases["drug"].map(DRUG_ORDER)
    vme_cases["high_conf_order"] = vme_cases["mechanism_subtype_high_confidence"].map(
        HIGH_CONF_ORDER
    )
    vme_cases = vme_cases.sort_values(
        ["drug_order", "high_conf_order", "abs_log2_error"],
        ascending=[True, True, False],
    )

    return {
        "fig5a_clinical_warning_rule_table": write_csv(
            rule_table, f"fig5a_clinical_warning_rule_table_{DATE}.csv"
        ),
        "fig5b_clinical_rule_level_performance": write_csv(
            rule_perf, f"fig5b_clinical_rule_level_performance_{DATE}.csv"
        ),
        "fig5c_mem_warning_enrichment": write_csv(
            mem_enrichment, f"fig5c_mem_warning_enrichment_{DATE}.csv"
        ),
        "fig5d_false_susceptible_cases": write_csv(
            vme_cases, f"fig5d_false_susceptible_cases_{DATE}.csv"
        ),
    }


def fig6_enriched_error_analyses(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    calibration = data["calibration_summary"].copy()
    susceptible_risk = data["susceptible_call_risk"].copy()
    breakpoint = data["breakpoint_zone_summary"].copy()
    combo_errors = data["phenotype_combo_errors"].copy()
    taxonomy = data["error_taxonomy_summary"].copy()
    safety_gate = data["safety_gate_summary"].copy()

    for frame in [calibration, susceptible_risk, breakpoint, combo_errors, taxonomy]:
        if "mechanism_subtype_high_confidence" in frame.columns:
            frame["high_conf_order"] = frame["mechanism_subtype_high_confidence"].map(
                HIGH_CONF_ORDER
            )
    calibration["drug_order"] = calibration["drug"].map(DRUG_ORDER)
    susceptible_risk["drug_order"] = susceptible_risk["drug"].map(DRUG_ORDER)
    breakpoint["drug_order"] = breakpoint["drug"].map(DRUG_ORDER)
    taxonomy["drug_order"] = taxonomy["drug"].map(DRUG_ORDER)
    calibration = calibration.sort_values(["drug_order", "high_conf_order"])
    susceptible_risk = susceptible_risk.sort_values(
        ["drug_order", "high_conf_order", "susceptible_call_confidence"]
    )
    breakpoint = breakpoint.sort_values(["drug_order", "breakpoint_zone", "high_conf_order"])
    combo_errors = combo_errors.sort_values(["ipm_mem_pair_group", "high_conf_order"])
    taxonomy = taxonomy.sort_values(["drug_order", "error_taxonomy", "high_conf_order"])
    safety_gate = safety_gate.sort_values(["gate_scenario", "drug"])

    return {
        "fig6a_calibration_summary": write_csv(
            calibration, f"fig6a_calibration_summary_{DATE}.csv"
        ),
        "fig6b_susceptible_call_risk": write_csv(
            susceptible_risk, f"fig6b_susceptible_call_risk_{DATE}.csv"
        ),
        "fig6c_breakpoint_zone_error_summary": write_csv(
            breakpoint, f"fig6c_breakpoint_zone_error_summary_{DATE}.csv"
        ),
        "fig6d_phenotype_combo_paired_errors": write_csv(
            combo_errors, f"fig6d_phenotype_combo_paired_errors_{DATE}.csv"
        ),
        "fig6e_error_taxonomy_summary": write_csv(
            taxonomy, f"fig6e_error_taxonomy_summary_{DATE}.csv"
        ),
        "fig6f_safety_gate_evaluation": write_csv(
            safety_gate, f"fig6f_safety_gate_evaluation_{DATE}.csv"
        ),
    }


def write_readme(outputs: dict[str, Path]) -> None:
    lines = [
        "# IPM-GPT Predictability Figure Data",
        "",
        f"Date: {DATE}",
        "",
        "These tables are raw panel data for the mechanism-dependent IPM/MEM MIC predictability article.",
        "No figure images are generated here.",
        "",
        "## Suggested Figure Map",
        "",
        "- Figure 1: dataset readiness, unified IPM/MEM model performance, and local IPM/MEM phenotype groups.",
        "- Figure 2: mechanism landscape, including first-pass strict and high-confidence disruptive layers.",
        "- Figure 3: subtype-specific IPM/MEM predictability, including high-confidence mechanism strata.",
        "- Figure 4: paired IPM-vs-MEM error differences, high-confidence strata, and ST sensitivity.",
        "- Figure 5: clinical warning-rule strata, VME/ME safety performance, and false-susceptible cases.",
        "- Figure 6: calibration, breakpoint distance, phenotype discordance, error taxonomy, and safety-gate analyses.",
        "",
        "## Exported Files",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- `{key}`: `{path.name}`")
    (OUT / f"README_predictability_figure_data_{DATE}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    outputs: dict[str, Path] = {}
    outputs.update(fig1_dataset_and_problem(data))
    outputs.update(fig2_mechanism_landscape(data))
    outputs.update(fig3_subtype_predictability(data))
    outputs.update(fig4_paired_and_st(data))
    outputs.update(fig5_clinical_warning_rules(data))
    outputs.update(fig6_enriched_error_analyses(data))
    write_readme(outputs)
    for key, path in outputs.items():
        frame = pd.read_csv(path)
        print(f"{key}: {frame.shape[0]} rows x {frame.shape[1]} cols -> {path}")
    print(f"Wrote figure data to {OUT}")


if __name__ == "__main__":
    main()
