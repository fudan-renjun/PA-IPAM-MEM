#!/usr/bin/env python3
"""Build the first analysis-ready metadata audit for the IPM-GPT study."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SEED = DATA / "metadata"
LOCAL_SOURCE = DATA / "raw_local" / "clinical_source" / "combined_local_clinical_source.xlsx"
PUBLIC_RAW_AUDIT = (
    DATA
    / "public_background"
    / "bvbrc_ipm_candidates"
    / "metadata"
    / "mic_ast_standard_public_raw_audit_2026-05-19.csv"
)
RESULTS = PROJECT / "results" / "00_metadata"
LOGS = PROJECT / "logs"

CONTRACT_COLUMNS = [
    "isolate_id",
    "data_origin",
    "year",
    "country",
    "specimen_type",
    "IPM_MIC",
    "IPM_SIR",
    "MEM_MIC",
    "MEM_SIR",
    "assembly_path",
]


def setup_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "m0_metadata_cleaning.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def normalize_isolate_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, int):
        return f"TL{value}"
    if isinstance(value, float) and value.is_integer():
        return f"TL{int(value)}"
    text = clean_text(value)
    if re.fullmatch(r"\d+(\.0)?", text):
        return f"TL{int(float(text))}"
    return text


def infer_local_year(isolate_id: str, sample_id: str) -> tuple[str, str]:
    for value, basis in ((isolate_id, "isolate_id_prefix"), (sample_id, "sample_id_prefix")):
        match = re.match(r"^(\d{2})", clean_text(value))
        if match:
            year = 2000 + int(match.group(1))
            if 2010 <= year <= 2035:
                return str(year), basis
    return "", "missing"


def status_from_mic_sir(mic: Any, sir: Any, sir_only_label: bool = False) -> str:
    has_mic = bool(clean_text(mic))
    has_sir = bool(clean_text(sir))
    if has_mic and has_sir:
        return "mic_and_sir"
    if has_mic:
        return "mic_only"
    if has_sir:
        return "sir_only" if sir_only_label else "sir_only_no_mic"
    return "missing"


def assembly_status(path: Any) -> str:
    text = clean_text(path)
    return "present" if text and Path(text).exists() else "missing"


def load_local_clinical() -> pd.DataFrame:
    sheets = pd.read_excel(LOCAL_SOURCE, sheet_name=None)
    frames: list[pd.DataFrame] = []
    for sheet_name, frame in sheets.items():
        subset = frame.iloc[:, :7].copy()
        subset.columns = [
            "specimen_type",
            "age",
            "sex",
            "department",
            "diagnosis",
            "isolate_id_raw",
            "sample_id",
        ]
        subset["source_sheet"] = sheet_name
        frames.append(subset)
    clinical = pd.concat(frames, ignore_index=True)
    clinical["isolate_id"] = clinical["isolate_id_raw"].map(normalize_isolate_id)
    clinical = clinical[clinical["isolate_id"].astype(str).str.len() > 0].copy()
    if clinical["isolate_id"].duplicated().any():
        duplicate_ids = clinical.loc[clinical["isolate_id"].duplicated(), "isolate_id"].tolist()
        raise ValueError(f"Duplicate local isolate IDs in clinical workbook: {duplicate_ids[:10]}")
    logging.info("Loaded local clinical workbook rows=%s", len(clinical))
    return clinical


def load_public_measurement_map() -> dict[tuple[str, str], dict[str, str]]:
    raw = pd.read_csv(PUBLIC_RAW_AUDIT, dtype=str).fillna("")
    raw = raw[
        raw["antibiotic"].isin(["imipenem", "meropenem"])
        & raw["include_standard"].str.lower().eq("true")
        & raw["mic"].astype(str).str.len().gt(0)
    ].copy()

    measurement_map: dict[tuple[str, str], dict[str, str]] = {}
    for (genome_id, antibiotic), group in raw.groupby(["genome_id", "antibiotic"], dropna=False):
        evidence = (
            group[["mic", "operator", "testing_standard", "is_censored"]]
            .drop_duplicates()
            .sort_values(["mic", "operator", "testing_standard"])
        )
        drug = "IPM" if antibiotic == "imipenem" else "MEM"
        if len(evidence) == 1:
            row = evidence.iloc[0]
            measurement_map[(str(genome_id), drug)] = {
                "mic": clean_text(row["mic"]),
                "operator": clean_text(row["operator"]) or "=",
                "standard": clean_text(row["testing_standard"]),
                "is_censored": clean_text(row["is_censored"]),
                "status": "standardized_raw_mic",
                "record_count": str(len(group)),
            }
        else:
            measurement_map[(str(genome_id), drug)] = {
                "mic": "",
                "operator": "",
                "standard": ";".join(sorted(set(evidence["testing_standard"].map(clean_text)))),
                "is_censored": "",
                "status": "conflicting_standardized_raw_mic",
                "record_count": str(len(group)),
            }
    logging.info("Loaded standardized public measurements=%s", len(measurement_map))
    return measurement_map


def build_local_rows() -> list[dict[str, str]]:
    clinical = load_local_clinical()
    seed = pd.read_csv(SEED / "local_ipm_mem_seed_metadata.csv", dtype=str).fillna("")
    phenotype = seed.set_index("isolate_id").to_dict(orient="index")
    rows: list[dict[str, str]] = []

    for _, rec in clinical.sort_values("isolate_id").iterrows():
        isolate_id = clean_text(rec["isolate_id"])
        phen = phenotype.get(isolate_id, {})
        sample_id = clean_text(rec["sample_id"])
        year, year_basis = infer_local_year(isolate_id, sample_id)
        assembly_path_value = str(DATA / "raw_local" / "assemblies" / f"{isolate_id}.fasta")
        ipm_mic = clean_text(phen.get("IPM_MIC_mg_L", ""))
        mem_mic = clean_text(phen.get("MEM_MIC_mg_L", ""))
        ipm_sir = clean_text(phen.get("IPM_SIR", ""))
        mem_sir = clean_text(phen.get("MEM_SIR", ""))
        ipm_status = status_from_mic_sir(ipm_mic, ipm_sir)
        mem_status = status_from_mic_sir(mem_mic, mem_sir)

        if ipm_status == "mic_and_sir" and assembly_status(assembly_path_value) == "present":
            tier = "local_primary_ipm"
        else:
            tier = "local_context_pending_ipm"

        rows.append(
            {
                "isolate_id": isolate_id,
                "data_origin": "local_clinical",
                "cohort_role": "local_core",
                "year": year,
                "year_basis": year_basis,
                "country": "",
                "specimen_type": clean_text(rec["specimen_type"]),
                "sample_id": sample_id,
                "age": clean_text(rec["age"]),
                "sex": clean_text(rec["sex"]),
                "department": clean_text(rec["department"]),
                "diagnosis": clean_text(rec["diagnosis"]),
                "disease": "",
                "IPM_MIC": ipm_mic,
                "IPM_MIC_operator": clean_text(phen.get("IPM_operator", "")),
                "IPM_MIC_basis": "local_parsed_mg_L" if ipm_mic else "missing",
                "IPM_SIR": ipm_sir,
                "MEM_MIC": mem_mic,
                "MEM_MIC_operator": clean_text(phen.get("MEM_operator", "")),
                "MEM_MIC_basis": "local_parsed_mg_L" if mem_mic else "missing",
                "MEM_SIR": mem_sir,
                "IPM_candidate_log2_MIC": "",
                "MEM_candidate_log2_MIC": "",
                "IPM_testing_standard": "",
                "MEM_testing_standard": "",
                "assembly_path": assembly_path_value,
                "assembly_status": assembly_status(assembly_path_value),
                "IPM_phenotype_status": ipm_status,
                "MEM_phenotype_status": mem_status,
                "m0_analysis_tier": tier,
                "source_metadata": "raw_local/clinical_source/combined_local_clinical_source.xlsx",
                "notes": f"source_sheet={clean_text(rec['source_sheet'])}",
            }
        )
    return rows


def build_ardap_rows() -> list[dict[str, str]]:
    seed = pd.read_csv(SEED / "ardap2024_ipm_mem_seed_metadata.csv", dtype=str).fillna("")
    rows: list[dict[str, str]] = []
    for _, rec in seed.sort_values("isolate_id").iterrows():
        path_value = clean_text(rec["assembly_path"])
        path_status = assembly_status(path_value)
        if path_status == "present":
            tier = "published_sir_validation"
        else:
            tier = "published_label_only_missing_assembly"
        rows.append(
            {
                "isolate_id": clean_text(rec["isolate_id"]),
                "data_origin": "published_ARDaP2024",
                "cohort_role": "published_validation",
                "year": clean_text(rec["year"]),
                "year_basis": "source_table",
                "country": clean_text(rec.get("country", "")),
                "specimen_type": clean_text(rec.get("specimen_type", "")),
                "sample_id": "",
                "age": "",
                "sex": "",
                "department": "",
                "diagnosis": "",
                "disease": clean_text(rec.get("disease", "")),
                "IPM_MIC": "",
                "IPM_MIC_operator": "",
                "IPM_MIC_basis": "not_available_sir_only",
                "IPM_SIR": clean_text(rec["IPM_SIR"]),
                "MEM_MIC": "",
                "MEM_MIC_operator": "",
                "MEM_MIC_basis": "not_available_sir_only",
                "MEM_SIR": clean_text(rec["MEM_SIR"]),
                "IPM_candidate_log2_MIC": "",
                "MEM_candidate_log2_MIC": "",
                "IPM_testing_standard": "",
                "MEM_testing_standard": "",
                "assembly_path": path_value,
                "assembly_status": path_status,
                "IPM_phenotype_status": status_from_mic_sir("", rec["IPM_SIR"], sir_only_label=True),
                "MEM_phenotype_status": status_from_mic_sir("", rec["MEM_SIR"], sir_only_label=True),
                "m0_analysis_tier": tier,
                "source_metadata": clean_text(rec["source_metadata"]),
                "notes": "ARDaP 2024 IPM/MEM labels staged as SIR-only",
            }
        )
    return rows


def public_status(
    measurement: dict[str, str] | None, candidate_log2_mic: str
) -> tuple[str, str, str, str, str]:
    if measurement and measurement["status"] == "standardized_raw_mic":
        return (
            measurement["mic"],
            measurement["operator"],
            "standardized_public_raw_audit_mg_L",
            measurement["standard"],
            "standardized_raw_mic",
        )
    if measurement and measurement["status"] == "conflicting_standardized_raw_mic":
        return "", "", "conflicting_standardized_public_raw_audit", measurement["standard"], "conflict"
    if candidate_log2_mic:
        return "", "", "candidate_log2_only_recheck_raw", "", "candidate_log2_only"
    return "", "", "missing", "", "missing"


def build_public_rows() -> list[dict[str, str]]:
    seed = pd.read_csv(SEED / "public_bvbrc_ipm_candidate_metadata.csv", dtype=str).fillna("")
    measurement_map = load_public_measurement_map()
    rows: list[dict[str, str]] = []

    for _, rec in seed.sort_values("isolate_id").iterrows():
        isolate_id = clean_text(rec["isolate_id"])
        ipm_measurement = measurement_map.get((isolate_id, "IPM"))
        mem_measurement = measurement_map.get((isolate_id, "MEM"))
        ipm_mic, ipm_op, ipm_basis, ipm_standard, ipm_status = public_status(
            ipm_measurement, clean_text(rec["IPM_existing_log2_MIC"])
        )
        mem_mic, mem_op, mem_basis, mem_standard, mem_status = public_status(
            mem_measurement, clean_text(rec["MEM_existing_log2_MIC"])
        )
        path_value = clean_text(rec["assembly_path"])
        if ipm_status == "standardized_raw_mic" and assembly_status(path_value) == "present":
            tier = "public_standardized_ipm_mic_background"
        else:
            tier = "public_candidate_recheck_raw_ipm"
        rows.append(
            {
                "isolate_id": isolate_id,
                "data_origin": "public_BV-BRC_IPM_candidate",
                "cohort_role": "public_background_candidate",
                "year": "",
                "year_basis": "missing",
                "country": "",
                "specimen_type": "",
                "sample_id": "",
                "age": "",
                "sex": "",
                "department": "",
                "diagnosis": "",
                "disease": "",
                "IPM_MIC": ipm_mic,
                "IPM_MIC_operator": ipm_op,
                "IPM_MIC_basis": ipm_basis,
                "IPM_SIR": "",
                "MEM_MIC": mem_mic,
                "MEM_MIC_operator": mem_op,
                "MEM_MIC_basis": mem_basis,
                "MEM_SIR": "",
                "IPM_candidate_log2_MIC": clean_text(rec["IPM_existing_log2_MIC"]),
                "MEM_candidate_log2_MIC": clean_text(rec["MEM_existing_log2_MIC"]),
                "IPM_testing_standard": ipm_standard,
                "MEM_testing_standard": mem_standard,
                "assembly_path": path_value,
                "assembly_status": assembly_status(path_value),
                "IPM_phenotype_status": ipm_status,
                "MEM_phenotype_status": mem_status,
                "m0_analysis_tier": tier,
                "source_metadata": clean_text(rec["source_metadata"]),
                "notes": (
                    "Public candidate selected by existing combined IPM field and QC-pass assembly; "
                    "final phenotype statistics need raw-record policy."
                ),
            }
        )
    return rows


def bool_count(frame: pd.DataFrame, column: str, value: str) -> int:
    return int(frame[column].astype(str).eq(value).sum())


def write_missingness(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, group in clean.groupby("data_origin", dropna=False):
        for column in CONTRACT_COLUMNS:
            missing = int(group[column].map(clean_text).eq("").sum())
            rows.append(
                {
                    "data_origin": cohort,
                    "field": column,
                    "rows": len(group),
                    "missing_rows": missing,
                    "missing_fraction": round(missing / len(group), 4) if len(group) else math.nan,
                }
            )
    missingness = pd.DataFrame(rows)
    missingness.to_csv(RESULTS / "m0_missingness_by_cohort.csv", index=False)
    return missingness


def write_summary(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, group in clean.groupby("data_origin", dropna=False):
        rows.append(
            {
                "data_origin": cohort,
                "rows": len(group),
                "assembly_present": bool_count(group, "assembly_status", "present"),
                "ipm_mic_and_sir": bool_count(group, "IPM_phenotype_status", "mic_and_sir"),
                "ipm_sir_only": bool_count(group, "IPM_phenotype_status", "sir_only"),
                "ipm_standardized_raw_mic": bool_count(
                    group, "IPM_phenotype_status", "standardized_raw_mic"
                ),
                "ipm_candidate_log2_only": bool_count(
                    group, "IPM_phenotype_status", "candidate_log2_only"
                ),
                "ipm_missing": bool_count(group, "IPM_phenotype_status", "missing"),
                "mem_mic_and_sir": bool_count(group, "MEM_phenotype_status", "mic_and_sir"),
                "mem_sir_only": bool_count(group, "MEM_phenotype_status", "sir_only"),
                "tiers": "; ".join(
                    f"{tier}={count}"
                    for tier, count in group["m0_analysis_tier"].value_counts().sort_index().items()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "m0_metadata_readiness_summary.csv", index=False)
    return summary


def write_tier_counts(clean: pd.DataFrame) -> pd.DataFrame:
    tiers = (
        clean.groupby(["data_origin", "m0_analysis_tier"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["data_origin", "m0_analysis_tier"])
    )
    tiers.to_csv(RESULTS / "m0_analysis_tier_counts.csv", index=False)
    return tiers


def frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [clean_text(row[column]).replace("|", "/") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(clean: pd.DataFrame, summary: pd.DataFrame, missingness: pd.DataFrame) -> None:
    summary_markdown = frame_to_markdown(summary)
    country_missing = missingness[
        (missingness["field"] == "country") & (missingness["missing_rows"] > 0)
    ]
    specimen_missing = missingness[
        (missingness["field"] == "specimen_type") & (missingness["missing_rows"] > 0)
    ]
    report = f"""# M0 Metadata Cleaning Audit

Date: 2026-05-22

## Outputs

- Clean metadata: `data/metadata/clean_metadata.tsv`
- Readiness summary: `results/00_metadata/m0_metadata_readiness_summary.csv`
- Tier counts: `results/00_metadata/m0_analysis_tier_counts.csv`
- Contract missingness: `results/00_metadata/m0_missingness_by_cohort.csv`
- Log: `logs/m0_metadata_cleaning.log`

## Cohort Readiness

{summary_markdown}

## Interpretation

- The local workbook contributes all 147 clinical isolates and keeps local specimen, sample, department, diagnosis, age, and sex fields.
- Local rows with parsed IPM MIC and source SIR labels are the first primary IPM phenotype layer.
- ARDaP 2024 stays a published SIR-only IPM/MEM validation layer at this stage.
- BV-BRC IPM candidates are separated into rows with standardized raw-audit MIC evidence and rows that still only carry an existing candidate `log2 MIC` selector.
- `country` remains blank where the staged source tables do not provide explicit evidence.

## Main Missingness Signals

- Country missingness rows by cohort: {country_missing[["data_origin", "missing_rows"]].to_dict(orient="records")}
- Specimen-type missingness rows by cohort: {specimen_missing[["data_origin", "missing_rows"]].to_dict(orient="records")}

## Next Move

Use `clean_metadata.tsv` to drive project-local MLST, AMR, and fine `oprD` annotation. The first high-value analysis set is `m0_analysis_tier == "local_primary_ipm"`, with ARDaP SIR rows as the first published validation layer and public standardized MIC rows as a background layer after breakpoint policy is fixed.
"""
    (RESULTS / "m0_metadata_audit_2026-05-22.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = build_local_rows() + build_ardap_rows() + build_public_rows()
    clean = pd.DataFrame(rows)
    clean = clean.sort_values(["data_origin", "isolate_id"]).reset_index(drop=True)
    if clean.duplicated(["data_origin", "isolate_id"]).any():
        duplicated = clean.loc[
            clean.duplicated(["data_origin", "isolate_id"], keep=False),
            ["data_origin", "isolate_id"],
        ]
        raise ValueError(f"Duplicate cohort-isolate rows in clean metadata: {duplicated.head(10)}")

    clean.to_csv(SEED / "clean_metadata.tsv", sep="\t", index=False)
    missingness = write_missingness(clean)
    summary = write_summary(clean)
    write_tier_counts(clean)
    write_report(clean, summary, missingness)
    logging.info("Wrote clean metadata rows=%s columns=%s", len(clean), len(clean.columns))


if __name__ == "__main__":
    main()
