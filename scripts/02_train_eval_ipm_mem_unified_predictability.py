#!/usr/bin/env python3
"""Train and evaluate a unified IPM/MEM prediction-error table for IPM-GPT.

This script creates the study-specific prediction layer requested for the
IPM-GPT mechanism-dependent predictability article. IPM and MEM are trained
with the same public-only model recipe so downstream subtype/error analysis is
not confounded by endpoint-specific model choices.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TRAIN_HELPER = WORKSPACE / "scripts" / "66_train_pamic_eight_v3_public_transportable_models.py"
PREDICT_HELPER = WORKSPACE / "scripts" / "67_predict_with_pamic_eight_v3_public_model.py"
EVAL_HELPER = WORKSPACE / "scripts" / "68_evaluate_pamic_eight_v3_locked_once.py"

PUBLIC_EXTERNAL_PATH = (
    WORKSPACE / "data" / "processed" / "features" / "feature_matrix_pamic_locked_public_external_2026-05-19.csv"
)
LOCAL_FEATURE_PATH = (
    WORKSPACE / "data" / "processed" / "features" / "feature_matrix_pamic_locked_local_validation_2026-05-19.csv"
)
LOCAL_ACTUAL_PATH = WORKSPACE / "results" / "external_prediction" / "external_actual_MIC_only_parsed_long.csv"

OUT = PROJECT / "results" / "02_unified_ipm_mem_prediction"
MODEL_DIR = PROJECT / "models" / "ipm_mem_unified_public_only"

DRUGS = ["IPM", "MEM"]
BREAKPOINTS = {"IPM": 2.0, "MEM": 1.0}
DATE = "2026-06-05"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
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


def build_policies(train_module) -> dict[str, object]:
    EndpointPolicy = train_module.EndpointPolicy
    policies = {}
    for drug in DRUGS:
        policies[drug] = EndpointPolicy(
            drug=drug,
            status="ipm_gpt_unified_public_only",
            breakpoint_log2=BREAKPOINTS[drug],
            gate_model_name="hgb",
            gate_feature_set="all_numeric",
            stage2_feature_set="all_numeric",
            threshold=0.45,
            approach="hard_gate",
            snapped_to_train_levels=True,
            cap_policy="no_cap",
            selection_basis="Pre-specified shared IPM/MEM recipe for mechanism-dependent predictability analysis.",
            rationale=(
                "Public-training-only HGB gate plus MIC regressor using all numeric features; "
                "same model class, feature policy, threshold, gating rule, and snapping rule for IPM and MEM."
            ),
        )
    return policies


def train_unified_artifact(train_module):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df = train_module.load_training_table()
    policies = build_policies(train_module)
    artifact = {
        "artifact_name": "ipm_mem_unified_public_only",
        "created_date": DATE,
        "training_feature_path": str(train_module.FEATURE_PATH),
        "scope": "IPM-GPT study-specific unified IPM/MEM prediction-error layer",
        "independence_rule": (
            "Trained only on public_training feature table. Locked public external and local validation "
            "feature tables are not read during training or policy selection."
        ),
        "shared_recipe": {
            "gate_model_name": "hgb",
            "stage2_model_name": "hgb_regressor",
            "gate_feature_set": "all_numeric",
            "stage2_feature_set": "all_numeric",
            "threshold": 0.45,
            "approach": "hard_gate",
            "snapped_to_train_levels": True,
            "cap_policy": "no_cap",
        },
        "endpoints": {},
    }

    apparent_frames = []
    for drug in DRUGS:
        policy = policies[drug]
        endpoint_artifact, apparent = train_module.fit_endpoint(train_df, policy)
        artifact["endpoints"][drug] = endpoint_artifact
        apparent_frames.append(apparent)

    apparent_predictions = pd.concat(apparent_frames, ignore_index=True)
    apparent_summary = train_module.summarize_training_apparent(apparent_predictions)
    policy_df = pd.DataFrame([asdict(policies[drug]) for drug in DRUGS])

    model_path = MODEL_DIR / f"ipm_mem_unified_public_only_model_{DATE}.joblib"
    joblib.dump(artifact, model_path)
    policy_df.to_csv(MODEL_DIR / f"ipm_mem_unified_public_only_policy_{DATE}.csv", index=False)
    apparent_predictions.to_csv(
        MODEL_DIR / f"ipm_mem_unified_public_only_training_apparent_predictions_{DATE}.csv",
        index=False,
    )
    apparent_summary.to_csv(
        MODEL_DIR / f"ipm_mem_unified_public_only_training_apparent_summary_{DATE}.csv",
        index=False,
    )
    (MODEL_DIR / f"ipm_mem_unified_public_only_policy_{DATE}.json").write_text(
        json.dumps({drug: asdict(policies[drug]) for drug in DRUGS}, indent=2),
        encoding="utf-8",
    )
    return artifact, model_path, policy_df, apparent_summary


def predict_long(features: pd.DataFrame, artifact: dict[str, object], cohort: str, predict_module) -> pd.DataFrame:
    work = features.copy()
    if "genome_id" not in work.columns:
        raise ValueError("Feature table lacks genome_id column.")
    work["genome_id"] = work["genome_id"].astype(str).str.strip()
    predict_module.add_derived_features(work)

    frames = []
    for drug in DRUGS:
        endpoint = artifact["endpoints"][drug]
        policy = endpoint["policy"]
        base = predict_module.endpoint_base_predictions(work, endpoint)
        pred = predict_module.apply_policy(base, endpoint)
        breakpoint = float(policy["breakpoint_log2"])
        frames.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "drug": drug,
                    "prediction_role": "ipm_mem_unified",
                    "genome_id": work["genome_id"].to_numpy(),
                    "model_status": policy["status"],
                    "policy": (
                        f"{policy['approach']}@{policy['threshold']}|"
                        f"{policy['stage2_feature_set']}|{policy['cap_policy']}|unified_ipm_mem"
                    ),
                    "breakpoint_log2": breakpoint,
                    "prob_ns": base["prob_ns"].to_numpy(dtype=float),
                    "pred_mic_log2": pred,
                    "pred_sns": np.where(pred > breakpoint, "NS", "S"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def public_external_actuals(features: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for drug in DRUGS:
        sub = features[pd.to_numeric(features[drug], errors="coerce").notna()].copy()
        if sub.empty:
            continue
        actual = pd.to_numeric(sub[drug], errors="coerce").astype(float)
        breakpoint = BREAKPOINTS[drug]
        frames.append(
            pd.DataFrame(
                {
                    "cohort": "locked_public_external",
                    "drug": drug,
                    "genome_id": sub["genome_id"].astype(str).str.strip().to_numpy(),
                    "actual_mic_log2": actual.to_numpy(dtype=float),
                    "interval_lower_log2": actual.to_numpy(dtype=float),
                    "interval_upper_log2": actual.to_numpy(dtype=float),
                    "actual_sns": np.where(actual.to_numpy(dtype=float) > breakpoint, "NS", "S"),
                    "actual_op": "=",
                    "actual_source": "feature_table_numeric_breakpoint",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def local_actuals() -> pd.DataFrame:
    local = pd.read_csv(LOCAL_ACTUAL_PATH, low_memory=False)
    local["genome_id"] = local["genome_id"].astype(str).str.strip()
    local = local[local["antibiotic"].isin(DRUGS)].copy()
    local["actual_MIC_log2"] = pd.to_numeric(local["actual_MIC_log2"], errors="coerce")
    local = local[local["actual_MIC_log2"].notna()].copy()

    lower = local["actual_MIC_log2"].astype(float).copy()
    upper = local["actual_MIC_log2"].astype(float).copy()
    right_censored = local["actual_op"].isin([">", ">="])
    left_censored = local["actual_op"].isin(["<", "<="])
    upper.loc[right_censored] = np.inf
    lower.loc[left_censored] = -np.inf

    breakpoint = local["antibiotic"].map(BREAKPOINTS).astype(float)
    numeric_sns = np.where(local["actual_MIC_log2"].to_numpy(dtype=float) > breakpoint.to_numpy(dtype=float), "NS", "S")
    source_sns = local["actual_SNS_from_source"].fillna("").astype(str).str.upper()
    actual_sns = np.where(source_sns.isin(["S", "NS"]), source_sns, numeric_sns)

    return pd.DataFrame(
        {
            "cohort": "locked_local_validation",
            "drug": local["antibiotic"].astype(str).to_numpy(),
            "genome_id": local["genome_id"].astype(str).to_numpy(),
            "sample_id": local["sample_id"].astype(str).to_numpy(),
            "actual_mic_raw": local["actual_MIC_raw"].astype(str).to_numpy(),
            "actual_mic_log2": local["actual_MIC_log2"].to_numpy(dtype=float),
            "interval_lower_log2": lower.to_numpy(dtype=float),
            "interval_upper_log2": upper.to_numpy(dtype=float),
            "actual_sns": actual_sns,
            "actual_op": local["actual_op"].astype(str).to_numpy(),
            "actual_source": "local_actual_mic_long_source_sir_when_available",
        }
    )


def enrich_for_ipm_gpt(predictions: pd.DataFrame) -> pd.DataFrame:
    clean = pd.read_csv(PROJECT / "data" / "metadata" / "clean_metadata.tsv", sep="\t", dtype=str).fillna("")
    local_meta = clean[clean["data_origin"].eq("local_clinical")].copy()
    keep_cols = [
        "isolate_id",
        "m0_analysis_tier",
        "specimen_type",
        "department",
        "diagnosis",
        "year",
        "IPM_MIC",
        "IPM_SIR",
        "MEM_MIC",
        "MEM_SIR",
    ]
    local_meta = local_meta[[c for c in keep_cols if c in local_meta.columns]].rename(columns={"isolate_id": "genome_id"})
    out = predictions.merge(local_meta, on="genome_id", how="left")
    out["ipm_mem_pair_group"] = ""
    local_rows = out["cohort"].eq("locked_local_validation")
    ipm_sir = out["IPM_SIR"].fillna("").astype(str)
    mem_sir = out["MEM_SIR"].fillna("").astype(str)
    out.loc[local_rows & ipm_sir.eq("S") & mem_sir.eq("S"), "ipm_mem_pair_group"] = "IPM-S/MEM-S"
    out.loc[local_rows & ipm_sir.eq("R") & mem_sir.isin(["S", "I"]), "ipm_mem_pair_group"] = "IPM-R/MEM-S_or_I"
    out.loc[local_rows & ipm_sir.eq("R") & mem_sir.eq("R"), "ipm_mem_pair_group"] = "IPM-R/MEM-R"
    out.loc[local_rows & ipm_sir.eq("S") & mem_sir.eq("R"), "ipm_mem_pair_group"] = "IPM-S/MEM-R"
    out.loc[local_rows & out["ipm_mem_pair_group"].eq(""), "ipm_mem_pair_group"] = "IPM/MEM incomplete"
    return out


def summarize_by_drug(predictions: pd.DataFrame, eval_module) -> pd.DataFrame:
    rows = []
    for (cohort, role, drug), sub in predictions.groupby(["cohort", "prediction_role", "drug"], dropna=False):
        row = {
            "cohort": cohort,
            "prediction_role": role,
            "drug": drug,
            "model_status": sub["model_status"].iloc[0],
            "policy": sub["policy"].iloc[0],
        }
        row.update(eval_module.summarize_group(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cohort", "drug"]).reset_index(drop=True)


def summarize_overall(predictions: pd.DataFrame, eval_module) -> pd.DataFrame:
    rows = []
    scopes = [
        ("IPM_only", ["IPM"]),
        ("MEM_only", ["MEM"]),
        ("IPM_MEM_combined", DRUGS),
    ]
    for cohort, cohort_df in predictions.groupby("cohort", dropna=False):
        for scope, drugs in scopes:
            sub = cohort_df[cohort_df["drug"].isin(drugs)].copy()
            if sub.empty:
                continue
            row = {
                "cohort": cohort,
                "scope": scope,
                "n_drugs_with_rows": int(sub["drug"].nunique()),
            }
            row.update(eval_module.summarize_group(sub))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["cohort", "scope"]).reset_index(drop=True)


def summarize_local_pair_groups(predictions: pd.DataFrame, eval_module) -> pd.DataFrame:
    rows = []
    local = predictions[predictions["cohort"].eq("locked_local_validation")].copy()
    for (pair_group, drug), sub in local.groupby(["ipm_mem_pair_group", "drug"], dropna=False):
        row = {"ipm_mem_pair_group": pair_group, "drug": drug}
        row.update(eval_module.summarize_group(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ipm_mem_pair_group", "drug"]).reset_index(drop=True)


def write_report(
    model_path: Path,
    policy_df: pd.DataFrame,
    apparent_summary: pd.DataFrame,
    by_drug: pd.DataFrame,
    overall: pd.DataFrame,
    local_pair_summary: pd.DataFrame,
) -> None:
    report = OUT / f"ipm_mem_unified_prediction_report_{DATE}.md"
    by_cols = [
        "cohort",
        "prediction_role",
        "drug",
        "model_status",
        "n",
        "n_s",
        "n_ns",
        "mic_ea_pm1",
        "mic_mae",
        "mic_bias",
        "sns_ca",
        "sns_vme",
        "sns_me",
        "pred_ns_pct",
        "gate_auc",
        "policy",
    ]
    lines = [
        "# IPM/MEM Unified Public-Only Prediction Layer",
        "",
        f"Date: {DATE}",
        "",
        "## Purpose",
        "",
        "This output is the study-specific prediction-error layer for the IPM-GPT mechanism-dependent predictability article.",
        "IPM and MEM are trained and evaluated with the same public-only recipe so downstream subtype-specific error analysis compares drugs under a shared modeling policy.",
        "",
        "## Independence Rule",
        "",
        "- Training uses only the public-training feature table from the previous PaMIC no-leak split.",
        "- Locked public external and locked local validation cohorts are not used for training or policy selection.",
        "- The shared policy is pre-specified here rather than optimized separately per drug.",
        "",
        "## Model Artifact",
        "",
        f"`{model_path}`",
        "",
        "## Shared Policy",
        "",
        markdown_table(policy_df),
        "",
        "## Apparent Public-Training Smoke Metrics",
        "",
        markdown_table(apparent_summary),
        "",
        "## Locked Evaluation By Drug",
        "",
        markdown_table(by_drug[[c for c in by_cols if c in by_drug.columns]]),
        "",
        "## Locked Evaluation Overall",
        "",
        markdown_table(overall),
        "",
        "## Local Paired IPM/MEM Phenotype Group Metrics",
        "",
        markdown_table(local_pair_summary),
        "",
        "## Interpretation",
        "",
        "- These outputs should be used as the first prediction-error table for mechanism-subtype analysis.",
        "- Poor accuracy is expected to be interpreted after joining WGS mechanism subtypes, not as a final endpoint claim by itself.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_module = load_module("ipm_gpt_train_helper", TRAIN_HELPER)
    predict_module = load_module("ipm_gpt_predict_helper", PREDICT_HELPER)
    eval_module = load_module("ipm_gpt_eval_helper", EVAL_HELPER)

    artifact, model_path, policy_df, apparent_summary = train_unified_artifact(train_module)
    public_features = pd.read_csv(PUBLIC_EXTERNAL_PATH, low_memory=False)
    local_features = pd.read_csv(LOCAL_FEATURE_PATH, low_memory=False)

    public_pred = predict_long(public_features, artifact, "locked_public_external", predict_module)
    local_pred = predict_long(local_features, artifact, "locked_local_validation", predict_module)
    public_actual = public_external_actuals(public_features)
    local_actual = local_actuals()

    public_eval = public_actual.merge(public_pred, on=["cohort", "drug", "genome_id"], how="inner")
    local_eval = local_actual.merge(local_pred, on=["cohort", "drug", "genome_id"], how="inner")
    predictions = eval_module.attach_metrics_columns(pd.concat([public_eval, local_eval], ignore_index=True))
    predictions = enrich_for_ipm_gpt(predictions)
    by_drug = summarize_by_drug(predictions, eval_module)
    overall = summarize_overall(predictions, eval_module)
    local_pair_summary = summarize_local_pair_groups(predictions, eval_module)

    predictions.to_csv(OUT / f"ipm_mem_unified_prediction_errors_{DATE}.tsv", sep="\t", index=False)
    by_drug.to_csv(OUT / f"ipm_mem_unified_summary_by_drug_{DATE}.csv", index=False)
    overall.to_csv(OUT / f"ipm_mem_unified_summary_overall_{DATE}.csv", index=False)
    local_pair_summary.to_csv(OUT / f"ipm_mem_unified_local_pair_group_summary_{DATE}.csv", index=False)
    write_report(model_path, policy_df, apparent_summary, by_drug, overall, local_pair_summary)

    print(by_drug[["cohort", "drug", "n", "mic_ea_pm1", "mic_mae", "mic_bias", "sns_ca", "sns_vme", "sns_me"]].to_string(index=False))
    print(f"Wrote unified IPM/MEM prediction layer to {OUT}")


if __name__ == "__main__":
    main()
