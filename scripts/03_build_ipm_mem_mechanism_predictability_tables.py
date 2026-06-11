#!/usr/bin/env python3
"""Join unified IPM/MEM prediction errors with strict mechanism subtypes."""

from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATE = "2026-06-05"

PREDICTION_ERRORS = (
    PROJECT
    / "results"
    / "02_unified_ipm_mem_prediction"
    / f"ipm_mem_unified_prediction_errors_{DATE}.tsv"
)
LOCAL_FEATURES = (
    WORKSPACE / "data" / "processed" / "features" / "feature_matrix_pamic_locked_local_validation_2026-05-19.csv"
)
PUBLIC_EXTERNAL_FEATURES = (
    WORKSPACE / "data" / "processed" / "features" / "feature_matrix_pamic_locked_public_external_2026-05-19.csv"
)
OUT = PROJECT / "results" / "03_mechanism_predictability"


def num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def any_positive(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=frame.index)
    present = pd.DataFrame({col: num(frame, col) for col in columns})
    return present.gt(0).any(axis=1)


def positive_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0, index=frame.index, dtype=int)
    present = pd.DataFrame({col: num(frame, col) for col in columns})
    return present.gt(0).sum(axis=1).astype(int)


def regex_cols(frame: pd.DataFrame, pattern: str) -> list[str]:
    return [column for column in frame.columns if re.search(pattern, column, flags=re.IGNORECASE)]


def load_feature_tables() -> pd.DataFrame:
    frames = []
    for cohort, path in [
        ("locked_local_validation", LOCAL_FEATURES),
        ("locked_public_external", PUBLIC_EXTERNAL_FEATURES),
    ]:
        frame = pd.read_csv(path, low_memory=False)
        frame["cohort"] = cohort
        frame["genome_id"] = frame["genome_id"].astype(str).str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_mechanism_evidence(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["cohort", "genome_id"]].copy()

    oprd_len = num(features, "oprd_len", default=np.nan)
    out["oprd_len"] = oprd_len
    out["oprd_disrupted"] = num(features, "oprD_disrupted").gt(0)
    out["oprd_mutated_broad"] = num(features, "oprD_mutated").gt(0)
    out["oprd_truncated"] = num(features, "oprd_truncated").gt(0)
    out["oprd_high_conf_disruptive"] = num(features, "has_high_conf_disruptive_effect").gt(0)
    out["oprd_large_indel_10bp"] = num(features, "has_large_indel_10bp").gt(0)
    out["oprd_short_len_lt_430"] = oprd_len.lt(430).fillna(False)
    out["oprd_severe_loss"] = out[
        [
            "oprd_disrupted",
            "oprd_truncated",
            "oprd_high_conf_disruptive",
            "oprd_large_indel_10bp",
            "oprd_short_len_lt_430",
        ]
    ].any(axis=1)

    carbapenemase_strict_cols = []
    for pattern in [
        r"^(VIM|IMP|NDM|KPC|SPM|GIM)-.+__present$",
        r"^GES-5__present$",
        r"^GES-6__present$",
        r"^GES-14__present$",
        r"^GES-20__present$",
    ]:
        carbapenemase_strict_cols.extend(regex_cols(features, pattern))
    carbapenemase_strict_cols = sorted(set(carbapenemase_strict_cols))
    out["acquired_carbapenemase_strict"] = any_positive(features, carbapenemase_strict_cols)
    out["acquired_carbapenemase_gene_count"] = positive_count(features, carbapenemase_strict_cols)
    out["carbapenemase_any_broad"] = num(features, "carbapenemase_any").gt(0)

    carb_genes = []
    for _, rec in features.iterrows():
        genes = []
        for col in carbapenemase_strict_cols:
            if pd.to_numeric(rec.get(col, 0), errors="coerce") > 0:
                genes.append(col.replace("__present", ""))
        carb_genes.append(";".join(genes))
    out["acquired_carbapenemase_genes"] = carb_genes

    amp_cols = regex_cols(features, r"(ampR|ampD|ampC|dacB).+(__present|_mut$)")
    pdc_cols = regex_cols(features, r"^PDC-.+__present$")
    out["ampc_ampR_strict"] = any_positive(features, [c for c in amp_cols if "ampr" in c.lower()])
    out["pdc_present_broad"] = any_positive(features, pdc_cols) | num(features, "blaPDC_any").gt(0)
    out["ampc_associated_strict"] = out["ampc_ampR_strict"]

    efflux_strict_cols = [
        col
        for col in ["mexR_mut", "nalD_mut", "nfxB_mut", "mexS_mut", "mexZ_mut"]
        if col in features.columns
    ]
    efflux_broad_cols = [
        col
        for col in ["mexR_mut", "nalC_mut", "nalD_mut", "mexT_mut", "mexS_mut", "mexZ_mut", "nfxB_mut"]
        if col in features.columns
    ]
    out["efflux_regulator_strict"] = any_positive(features, efflux_strict_cols)
    out["efflux_regulator_strict_count"] = positive_count(features, efflux_strict_cols)
    out["efflux_regulator_broad"] = any_positive(features, efflux_broad_cols) | num(features, "efflux_regulator_any").gt(0)
    out["efflux_regulator_broad_count"] = positive_count(features, efflux_broad_cols)

    strong_cols = [
        "oprd_severe_loss",
        "acquired_carbapenemase_strict",
        "ampc_associated_strict",
        "efflux_regulator_strict",
    ]
    out["strong_mechanism_count"] = out[strong_cols].sum(axis=1).astype(int)
    out["mechanism_multilabel"] = [
        ";".join(
            label
            for label, flag in [
                ("OprD-loss", rec.oprd_severe_loss),
                ("carbapenemase", rec.acquired_carbapenemase_strict),
                ("AmpC-associated", rec.ampc_associated_strict),
                ("efflux-associated", rec.efflux_regulator_strict),
            ]
            if bool(flag)
        )
        or "no_strict_mechanism"
        for rec in out.itertuples(index=False)
    ]

    out["mechanism_subtype_strict"] = "No strict mechanism"
    out.loc[out["efflux_regulator_strict"], "mechanism_subtype_strict"] = "Efflux-associated genotype"
    out.loc[out["ampc_associated_strict"], "mechanism_subtype_strict"] = "AmpC-associated genotype"
    out.loc[out["oprd_severe_loss"], "mechanism_subtype_strict"] = "OprD-loss"
    out.loc[out["acquired_carbapenemase_strict"], "mechanism_subtype_strict"] = "Carbapenemase-mediated"
    out.loc[out["strong_mechanism_count"].ge(2), "mechanism_subtype_strict"] = "Composite"

    out["mechanism_reason"] = out.apply(mechanism_reason, axis=1)
    return out


def mechanism_reason(row: pd.Series) -> str:
    parts = []
    if row["oprd_severe_loss"]:
        parts.append("severe_oprD_loss")
    if row["acquired_carbapenemase_strict"]:
        genes = row.get("acquired_carbapenemase_genes", "")
        parts.append(f"acquired_carbapenemase={genes or 'present'}")
    if row["ampc_associated_strict"]:
        parts.append("ampC_ampR_strict")
    if row["efflux_regulator_strict"]:
        parts.append(f"strict_efflux_regulator_count={row['efflux_regulator_strict_count']}")
    if not parts:
        broad = []
        if row["oprd_mutated_broad"]:
            broad.append("broad_oprD_mutation")
        if row["pdc_present_broad"]:
            broad.append("broad_PDC_present")
        if row["efflux_regulator_broad"]:
            broad.append("broad_efflux_signal")
        return "no_strict_mechanism" + (f"; broad_context={';'.join(broad)}" if broad else "")
    return "; ".join(parts)


def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    true = df["true_sns_int"].astype(int).to_numpy()
    pred = df["pred_sns_int"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "n": int(len(df)),
        "n_s": int((true == 0).sum()),
        "n_ns": int((true == 1).sum()),
        "mic_ea_pm1": float(df["mic_ea_pm1"].mean()),
        "mic_mae": float(df["abs_log2_error"].mean()),
        "mic_bias": float(df["signed_log2_error"].mean()),
        "large_error_gt2": float((df["abs_log2_error"] > 2).mean()),
        "sns_ca": float((true == pred).mean()),
        "sns_balanced_accuracy": float(balanced_accuracy_score(true, pred)) if len(set(true)) > 1 else math.nan,
        "sns_s_recall": float(recall_score(true, pred, pos_label=0, zero_division=0)),
        "sns_ns_recall": float(recall_score(true, pred, pos_label=1, zero_division=0)),
        "sns_me": float(fp / (fp + tn)) if (fp + tn) else math.nan,
        "sns_vme": float(fn / (fn + tp)) if (fn + tp) else math.nan,
    }
    try:
        out["gate_auc"] = float(roc_auc_score(true, df["prob_ns"].astype(float).to_numpy()))
    except ValueError:
        out["gate_auc"] = math.nan
    return out


def summarize_by(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(summarize_group(sub))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


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


def write_report(
    evidence_summary: pd.DataFrame,
    subtype_summary: pd.DataFrame,
    pair_subtype_summary: pd.DataFrame,
) -> None:
    report = OUT / f"mechanism_predictability_first_pass_report_{DATE}.md"
    lines = [
        "# IPM/MEM Mechanism-Predictability First Pass",
        "",
        f"Date: {DATE}",
        "",
        "## Purpose",
        "",
        "This first-pass table joins unified IPM/MEM prediction errors with strict genome-defined mechanism evidence.",
        "The strict subtype is intended for exploratory error stratification and auditability, not as the final manuscript-grade mechanism call.",
        "",
        "## Strict Mechanism Rules",
        "",
        "- OprD-loss: `oprD_disrupted`, `oprd_truncated`, high-confidence disruptive effect, >=10 bp indel, or OprD length <430 aa.",
        "- Carbapenemase-mediated: strict acquired carbapenemase genes (`VIM`, `IMP`, `NDM`, `KPC`, `SPM`, `GIM`, or carbapenemase-associated `GES` variants). Broad OXA/PDC signals are not counted as strict carbapenemase.",
        "- AmpC-associated genotype: strict ampC/ampR/dacB/ampD mutation evidence only; broad PDC presence is retained as context but not used as a strict subtype driver.",
        "- Efflux-associated genotype: strict regulator mutations in `mexR`, `nalD`, `nfxB`, `mexS`, or `mexZ`; broad `nalC`, `mexT`, and all-positive efflux signals are retained as context but not used alone.",
        "- Composite: at least two strict mechanism classes.",
        "",
        "## Isolate-Level Mechanism Counts",
        "",
        markdown_table(evidence_summary),
        "",
        "## Prediction Performance By Strict Subtype",
        "",
        markdown_table(subtype_summary),
        "",
        "## Local Phenotype Group x Strict Subtype",
        "",
        markdown_table(pair_subtype_summary),
        "",
        "## Interpretation Guardrail",
        "",
        "AmpC and efflux calls remain genotype-associated. Expression or functional validation would be needed to claim overexpression.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTION_ERRORS, sep="\t", low_memory=False)
    predictions["genome_id"] = predictions["genome_id"].astype(str).str.strip()

    features = load_feature_tables()
    mechanism = build_mechanism_evidence(features)

    joined = predictions.merge(mechanism, on=["cohort", "genome_id"], how="left", validate="many_to_one")
    joined["mechanism_subtype_strict"] = joined["mechanism_subtype_strict"].fillna("Mechanism missing")
    joined["mechanism_multilabel"] = joined["mechanism_multilabel"].fillna("mechanism_missing")

    evidence_summary = (
        mechanism.groupby(["cohort", "mechanism_subtype_strict"], dropna=False)
        .agg(
            isolates=("genome_id", "size"),
            oprd_severe_loss=("oprd_severe_loss", "sum"),
            acquired_carbapenemase_strict=("acquired_carbapenemase_strict", "sum"),
            ampc_associated_strict=("ampc_associated_strict", "sum"),
            efflux_regulator_strict=("efflux_regulator_strict", "sum"),
        )
        .reset_index()
        .sort_values(["cohort", "mechanism_subtype_strict"])
    )
    subtype_summary = summarize_by(
        joined,
        ["cohort", "drug", "mechanism_subtype_strict"],
    )
    local = joined[joined["cohort"].eq("locked_local_validation")].copy()
    pair_subtype_summary = summarize_by(
        local,
        ["drug", "ipm_mem_pair_group", "mechanism_subtype_strict"],
    )

    mechanism.to_csv(OUT / f"mechanism_evidence_strict_first_pass_{DATE}.tsv", sep="\t", index=False)
    joined.to_csv(OUT / f"ipm_mem_prediction_errors_with_mechanism_{DATE}.tsv", sep="\t", index=False)
    evidence_summary.to_csv(OUT / f"mechanism_evidence_strict_counts_{DATE}.csv", index=False)
    subtype_summary.to_csv(OUT / f"subtype_predictability_summary_by_drug_{DATE}.csv", index=False)
    pair_subtype_summary.to_csv(OUT / f"local_pair_group_subtype_predictability_{DATE}.csv", index=False)
    write_report(evidence_summary, subtype_summary, pair_subtype_summary)

    print(evidence_summary.to_string(index=False))
    print()
    print(
        subtype_summary[
            [
                "cohort",
                "drug",
                "mechanism_subtype_strict",
                "n",
                "mic_ea_pm1",
                "mic_mae",
                "mic_bias",
                "sns_ca",
                "sns_vme",
                "sns_me",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote mechanism predictability outputs to {OUT}")


if __name__ == "__main__":
    main()
