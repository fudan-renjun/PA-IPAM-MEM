#!/usr/bin/env python3
"""Audit reusable RGI and Snippy outputs for staged IPM-GPT metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
INPUT = PROJECT / "data" / "metadata" / "clean_metadata.tsv"
RESULTS = PROJECT / "results" / "01_annotation_reuse"
LOGS = PROJECT / "logs"


def setup_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "m1_annotation_reuse_audit.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def exists(path: Path) -> str:
    return "present" if path.exists() else "missing"


def annotation_paths(data_origin: str, isolate_id: str) -> dict[str, Path]:
    if data_origin in {"local_clinical", "published_ARDaP2024"}:
        rgi_root = WORKSPACE / "data" / "processed" / "rgi" / "external"
        snippy_root = WORKSPACE / "data" / "processed" / "snippy" / "external"
    else:
        rgi_root = WORKSPACE / "data" / "processed" / "rgi" / "per_genome"
        snippy_root = WORKSPACE / "data" / "processed" / "snippy" / "per_genome"
    return {
        "existing_rgi_txt_path": rgi_root / f"{isolate_id}.txt",
        "existing_rgi_json_path": rgi_root / f"{isolate_id}.json",
        "existing_snippy_dir_path": snippy_root / isolate_id,
        "existing_snippy_tab_path": snippy_root / isolate_id / "snps.tab",
    }


def reuse_hint(row: pd.Series) -> str:
    rgi_ok = row["existing_rgi_txt_status"] == "present"
    snippy_ok = row["existing_snippy_tab_status"] == "present"
    if rgi_ok and snippy_ok:
        return "raw_rgi_and_snippy_available_for_recheck"
    if snippy_ok:
        return "snippy_available_rgi_needed_for_amr_mechanisms"
    if rgi_ok:
        return "rgi_available_snippy_needed_for_variant_mechanisms"
    return "annotation_outputs_needed"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "/") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_manifest(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, rec in clean.iterrows():
        isolate_id = str(rec["isolate_id"])
        data_origin = str(rec["data_origin"])
        paths = annotation_paths(data_origin, isolate_id)
        row = {
            "isolate_id": isolate_id,
            "data_origin": data_origin,
            "cohort_role": str(rec["cohort_role"]),
            "m0_analysis_tier": str(rec["m0_analysis_tier"]),
            "assembly_status": str(rec["assembly_status"]),
        }
        for column, path in paths.items():
            row[column] = str(path)
            row[column.replace("_path", "_status")] = exists(path)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest["reuse_hint"] = manifest.apply(reuse_hint, axis=1)
    return manifest


def summarize(manifest: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        manifest.groupby(group_columns, dropna=False)
        .agg(
            rows=("isolate_id", "size"),
            assembly_present=("assembly_status", lambda s: int((s == "present").sum())),
            rgi_txt_present=("existing_rgi_txt_status", lambda s: int((s == "present").sum())),
            rgi_json_present=("existing_rgi_json_status", lambda s: int((s == "present").sum())),
            snippy_tab_present=("existing_snippy_tab_status", lambda s: int((s == "present").sum())),
        )
        .reset_index()
    )
    return summary


def write_report(origin_summary: pd.DataFrame, tier_summary: pd.DataFrame) -> None:
    report = f"""# Existing Annotation Coverage Audit

Date: 2026-05-22

## Outputs

- Per-isolate manifest: `results/01_annotation_reuse/existing_annotation_manifest.tsv`
- Origin summary: `results/01_annotation_reuse/existing_annotation_coverage_by_origin.csv`
- Tier summary: `results/01_annotation_reuse/existing_annotation_coverage_by_tier.csv`
- Log: `logs/m1_annotation_reuse_audit.log`

## Coverage By Data Origin

{frame_to_markdown(origin_summary)}

## Coverage By M0 Tier

{frame_to_markdown(tier_summary)}

## Interpretation

- Existing outputs are treated as reusable raw evidence candidates, not final IPM-GPT mechanism labels.
- Local rows and downloaded ARDaP rows already have broad RGI and Snippy coverage from the prior external pipeline.
- The staged BV-BRC IPM candidate set has broad Snippy coverage but incomplete RGI coverage, so public carbapenemase and beta-lactamase mechanism annotation still needs a project-local AMR pass or a documented raw-AMR fallback.
- Fine `oprD` work should re-parse severity-aware variant evidence instead of reusing binary `oprD_mutated` style features.
"""
    (RESULTS / "existing_annotation_coverage_audit_2026-05-22.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")
    manifest = build_manifest(clean)
    origin_summary = summarize(manifest, ["data_origin"])
    tier_summary = summarize(manifest, ["data_origin", "m0_analysis_tier"])

    manifest.to_csv(RESULTS / "existing_annotation_manifest.tsv", sep="\t", index=False)
    origin_summary.to_csv(RESULTS / "existing_annotation_coverage_by_origin.csv", index=False)
    tier_summary.to_csv(RESULTS / "existing_annotation_coverage_by_tier.csv", index=False)
    write_report(origin_summary, tier_summary)
    logging.info("Wrote annotation reuse manifest rows=%s", len(manifest))


if __name__ == "__main__":
    main()
