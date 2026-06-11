#!/usr/bin/env python3
"""Deepen local OprD, efflux-regulator, and AmpC-axis variant annotation.

This script parses local Snippy variant tables directly, assigns coding and
proximal upstream variants to selected PAO1 genes, and joins the resulting
severity calls to the unified IPM/MEM prediction-error table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATE = "2026-06-05"

SNIPPY_DIR = WORKSPACE / "data" / "processed" / "snippy" / "external"
PREDICTION_ERRORS = (
    PROJECT
    / "results"
    / "02_unified_ipm_mem_prediction"
    / f"ipm_mem_unified_prediction_errors_{DATE}.tsv"
)
FIRST_PASS_MECHANISMS = (
    PROJECT
    / "results"
    / "03_mechanism_predictability"
    / f"mechanism_evidence_strict_first_pass_{DATE}.tsv"
)
OUT = PROJECT / "results" / "08_deep_mechanism_annotation"


@dataclass(frozen=True)
class TargetGene:
    gene: str
    locus_tag: str
    strand: str
    start: int
    end: int
    axis: str
    strict_driver: bool
    note: str

    @property
    def upstream_start(self) -> int:
        if self.strand == "+":
            return max(1, self.start - 200)
        return self.end + 1

    @property
    def upstream_end(self) -> int:
        if self.strand == "+":
            return self.start - 1
        return self.end + 200


TARGETS = [
    TargetGene("oprD", "PA0958", "-", 1060548, 1062245, "oprd", True, "carbapenem porin"),
    TargetGene("mexR", "PA0424", "-", 470060, 470596, "efflux", True, "MexAB-OprM repressor"),
    TargetGene("nalC", "PA3720", "+", 4185165, 4185905, "efflux", False, "MexAB-OprM associated regulator"),
    TargetGene("nalD", "PA3574", "-", 4018226, 4018960, "efflux", True, "MexAB-OprM repressor"),
    TargetGene("mexT", "PA2492", "-", 2788905, 2790266, "efflux", False, "MexEF-OprN activator/context"),
    TargetGene("mexS", "PA2491", "-", 2788163, 2788900, "efflux", True, "MexEF-OprN associated regulator"),
    TargetGene("mexZ", "PA2020", "+", 2251397, 2252014, "efflux", True, "MexXY-OprM repressor"),
    TargetGene("nfxB", "PA4596", "+", 5165027, 5165758, "efflux", True, "MexCD-OprJ repressor"),
    TargetGene("ampR", "PA4109", "-", 4592990, 4593880, "ampc_axis", True, "AmpC transcriptional regulator"),
    TargetGene("ampC", "PA4110", "+", 4594029, 4595222, "ampc_axis", False, "PDC/AmpC coding allele context"),
    TargetGene("ampD", "PA4522", "-", 5064774, 5065340, "ampc_axis", True, "AmpC expression regulator"),
    TargetGene("dacB", "PA3047", "+", 3410264, 3411694, "ampc_axis", True, "PBP4/dacB AmpC-axis regulator"),
    TargetGene("ampDh2", "PA5485", "-", 6176516, 6177295, "ampc_axis_context", False, "AmpD homolog context"),
    TargetGene("ampDh3", "PA0807", "+", 884799, 885566, "ampc_axis_context", False, "AmpD homolog context"),
]

BY_LOCUS = {target.locus_tag: target for target in TARGETS}
BY_GENE = {target.gene.lower(): target for target in TARGETS}


def safe_read_snps(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "POS" in frame.columns:
        frame["POS"] = pd.to_numeric(frame["POS"], errors="coerce")
    return frame


def assign_target(row: pd.Series | dict[str, object]) -> tuple[TargetGene | None, str]:
    locus = str(row.get("LOCUS_TAG", "")).strip()
    gene = str(row.get("GENE", "")).strip().lower()
    if locus in BY_LOCUS:
        return BY_LOCUS[locus], "coding_or_annotated"
    if gene in BY_GENE:
        return BY_GENE[gene], "coding_or_annotated"

    pos = row.get("POS")
    if pd.isna(pos):
        return None, ""
    pos_int = int(pos)
    for target in TARGETS:
        if target.upstream_start <= pos_int <= target.upstream_end:
            return target, "proximal_upstream_200bp"
    return None, ""


def classify_effect(effect: str, variant_type: str, location_class: str) -> str:
    effect_l = effect.lower()
    type_l = variant_type.lower()
    if location_class == "proximal_upstream_200bp":
        return "promoter_or_upstream"
    if any(term in effect_l for term in ["frameshift_variant", "stop_gained", "start_lost", "stop_lost"]):
        return "disruptive"
    if any(term in effect_l for term in ["inframe_insertion", "inframe_deletion"]):
        return "inframe_indel"
    if "missense_variant" in effect_l:
        return "missense"
    if "synonymous_variant" in effect_l:
        return "synonymous"
    if type_l in {"del", "ins"} and effect_l:
        return "coding_indel_uncertain"
    if type_l in {"del", "ins"}:
        return "indel_unannotated"
    if effect_l:
        return "other_annotated"
    return "unannotated"


def severity_rank(effect_class: str) -> int:
    return {
        "none": 0,
        "synonymous": 1,
        "other_annotated": 2,
        "promoter_or_upstream": 3,
        "missense": 4,
        "inframe_indel": 5,
        "coding_indel_uncertain": 6,
        "indel_unannotated": 6,
        "disruptive": 7,
    }.get(effect_class, 2)


def severity_label(classes: list[str]) -> str:
    if not classes:
        return "none"
    max_rank = max(severity_rank(c) for c in classes)
    if max_rank >= 7:
        return "disruptive"
    if max_rank >= 6:
        return "coding_indel_uncertain"
    if max_rank >= 5:
        return "inframe_indel"
    if max_rank >= 4:
        return "missense"
    if max_rank >= 3:
        return "promoter_or_upstream"
    if max_rank >= 2:
        return "other_annotated"
    return "synonymous_only"


def parse_local_target_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_rows = []
    gene_rows = []
    sample_dirs = sorted([p for p in SNIPPY_DIR.iterdir() if p.is_dir()])

    for sample_dir in sample_dirs:
        genome_id = sample_dir.name
        snps = safe_read_snps(sample_dir / "snps.tab")
        sample_variant_rows = []
        if not snps.empty:
            locus_mask = snps["LOCUS_TAG"].isin(BY_LOCUS) if "LOCUS_TAG" in snps.columns else False
            gene_mask = (
                snps["GENE"].str.lower().isin(BY_GENE)
                if "GENE" in snps.columns
                else False
            )
            upstream_mask = pd.Series(False, index=snps.index)
            if "POS" in snps.columns:
                for target in TARGETS:
                    upstream_mask |= snps["POS"].between(target.upstream_start, target.upstream_end)
            target_snps = snps[locus_mask | gene_mask | upstream_mask].copy()

            for row in target_snps.itertuples(index=False):
                rec = row._asdict()
                target, location_class = assign_target(rec)
                if target is None:
                    continue
                effect_class = classify_effect(
                    str(rec.get("EFFECT", "")),
                    str(rec.get("TYPE", "")),
                    location_class,
                )
                sample_variant_rows.append(
                    {
                        "genome_id": genome_id,
                        "gene": target.gene,
                        "locus_tag": target.locus_tag,
                        "axis": target.axis,
                        "strict_driver": target.strict_driver,
                        "location_class": location_class,
                        "pos": rec.get("POS"),
                        "variant_type": rec.get("TYPE", ""),
                        "ref": rec.get("REF", ""),
                        "alt": rec.get("ALT", ""),
                        "effect": rec.get("EFFECT", ""),
                        "effect_class": effect_class,
                    }
                )
        variant_rows.extend(sample_variant_rows)

        by_gene: dict[str, list[dict[str, object]]] = {}
        for rec in sample_variant_rows:
            by_gene.setdefault(str(rec["gene"]), []).append(rec)

        for target in TARGETS:
            records = by_gene.get(target.gene, [])
            classes = [str(r["effect_class"]) for r in records]
            nonsyn_classes = [
                c
                for c in classes
                if c
                not in {
                    "synonymous",
                    "other_annotated",
                    "unannotated",
                }
            ]
            gene_rows.append(
                {
                    "genome_id": genome_id,
                    "gene": target.gene,
                    "locus_tag": target.locus_tag,
                    "axis": target.axis,
                    "strict_driver": target.strict_driver,
                    "variant_count": len(records),
                    "n_synonymous": classes.count("synonymous"),
                    "n_missense": classes.count("missense"),
                    "n_inframe_indel": classes.count("inframe_indel"),
                    "n_disruptive": classes.count("disruptive"),
                    "n_promoter_upstream": classes.count("promoter_or_upstream"),
                    "n_coding_indel_uncertain": classes.count("coding_indel_uncertain")
                    + classes.count("indel_unannotated"),
                    "n_non_syn_or_regulatory": len(nonsyn_classes),
                    "severity": severity_label(classes),
                    "variant_signature": "; ".join(
                        compact_variant_signature(r) for r in records[:12]
                    ),
                }
            )

    return pd.DataFrame(variant_rows), pd.DataFrame(gene_rows)


def compact_variant_signature(rec: dict[str, object]) -> str:
    effect = str(rec.get("effect", "")).strip()
    aa_match = re.search(r"p\.[A-Za-z*?0-9_]+", effect)
    effect_short = aa_match.group(0) if aa_match else effect.split(" ")[0] if effect else str(rec["effect_class"])
    return f"{rec['gene']}:{rec['effect_class']}:{effect_short}@{rec['pos']}"


def summarize_isolate_mechanisms(gene_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genome_id, sub in gene_summary.groupby("genome_id", sort=True):
        def gene_severity(gene: str) -> str:
            vals = sub.loc[sub["gene"].eq(gene), "severity"]
            return vals.iloc[0] if len(vals) else "none"

        def axis_flag(axis: str, strict_only: bool, min_rank: int) -> bool:
            ss = sub[sub["axis"].eq(axis)]
            if strict_only:
                ss = ss[ss["strict_driver"].astype(bool)]
            return any(severity_rank(str(v)) >= min_rank for v in ss["severity"])

        def axis_count(axis: str, strict_only: bool, min_rank: int) -> int:
            ss = sub[sub["axis"].eq(axis)]
            if strict_only:
                ss = ss[ss["strict_driver"].astype(bool)]
            return int(sum(severity_rank(str(v)) >= min_rank for v in ss["severity"]))

        rows.append(
            {
                "cohort": "locked_local_validation",
                "genome_id": genome_id,
                "oprd_deep_severity": gene_severity("oprD"),
                "oprd_deep_non_syn_or_disruptive": severity_rank(gene_severity("oprD")) >= 4,
                "oprd_deep_disruptive": severity_rank(gene_severity("oprD")) >= 6,
                "efflux_strict_driver_non_syn_any": axis_flag("efflux", True, 4),
                "efflux_strict_driver_disruptive_any": axis_flag("efflux", True, 6),
                "efflux_strict_driver_non_syn_gene_count": axis_count("efflux", True, 4),
                "efflux_broad_non_syn_any": axis_flag("efflux", False, 4),
                "efflux_broad_non_syn_gene_count": axis_count("efflux", False, 4),
                "ampc_core_driver_non_syn_any": axis_flag("ampc_axis", True, 4),
                "ampc_core_driver_disruptive_any": axis_flag("ampc_axis", True, 6),
                "ampc_core_driver_non_syn_gene_count": axis_count("ampc_axis", True, 4),
                "ampc_coding_non_syn_or_regulatory": severity_rank(gene_severity("ampC")) >= 3,
                "ampc_context_homolog_non_syn_any": axis_flag("ampc_axis_context", False, 4),
                "ampR_severity": gene_severity("ampR"),
                "ampC_severity": gene_severity("ampC"),
                "ampD_severity": gene_severity("ampD"),
                "dacB_severity": gene_severity("dacB"),
                "mexR_severity": gene_severity("mexR"),
                "nalC_severity": gene_severity("nalC"),
                "nalD_severity": gene_severity("nalD"),
                "mexT_severity": gene_severity("mexT"),
                "mexS_severity": gene_severity("mexS"),
                "mexZ_severity": gene_severity("mexZ"),
                "nfxB_severity": gene_severity("nfxB"),
            }
        )
    return pd.DataFrame(rows)


def derive_refined_subtype(df: pd.DataFrame) -> pd.Series:
    labels = []
    for row in df.itertuples(index=False):
        strong = []
        if bool(getattr(row, "oprd_severe_loss", False)) or bool(getattr(row, "oprd_deep_disruptive", False)):
            strong.append("OprD-loss")
        if bool(getattr(row, "acquired_carbapenemase_strict", False)):
            strong.append("carbapenemase")
        if bool(getattr(row, "ampc_core_driver_non_syn_any", False)):
            strong.append("AmpC-axis")
        if bool(getattr(row, "efflux_strict_driver_non_syn_any", False)):
            strong.append("efflux")
        if len(strong) >= 2:
            labels.append("Composite refined")
        elif strong:
            labels.append(f"{strong[0]} refined")
        elif bool(getattr(row, "ampc_coding_non_syn_or_regulatory", False)):
            labels.append("AmpC coding-context only")
        elif bool(getattr(row, "efflux_broad_non_syn_any", False)):
            labels.append("Efflux broad-context only")
        else:
            labels.append("No refined mechanism")
    return pd.Series(labels, index=df.index)


def derive_high_confidence_subtype(df: pd.DataFrame) -> pd.Series:
    labels = []
    for row in df.itertuples(index=False):
        strong = []
        if bool(getattr(row, "oprd_severe_loss", False)) or bool(getattr(row, "oprd_deep_disruptive", False)):
            strong.append("OprD-loss")
        if bool(getattr(row, "acquired_carbapenemase_strict", False)):
            strong.append("carbapenemase")
        if bool(getattr(row, "ampc_core_driver_disruptive_any", False)):
            strong.append("AmpC-axis disruptive")
        if bool(getattr(row, "efflux_strict_driver_disruptive_any", False)):
            strong.append("efflux disruptive")
        if len(strong) >= 2:
            labels.append("High-confidence composite")
        elif strong:
            labels.append(f"{strong[0]}")
        else:
            labels.append("No high-confidence driver")
    return pd.Series(labels, index=df.index)


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


def summarize_by(df: pd.DataFrame, group_cols: list[str], min_n: int = 1) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if len(sub) < min_n:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(summarize_group(sub))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_paired_error_table(local_predictions: pd.DataFrame, mechanisms: pd.DataFrame) -> pd.DataFrame:
    piv = local_predictions.pivot_table(
        index="genome_id",
        columns="drug",
        values=["abs_log2_error", "signed_log2_error", "actual_mic_log2", "pred_mic_log2"],
        aggfunc="first",
    )
    piv.columns = [f"{metric}_{drug}" for metric, drug in piv.columns]
    piv = piv.reset_index()
    needed = ["abs_log2_error_IPM", "abs_log2_error_MEM"]
    piv = piv.dropna(subset=needed)
    piv["MEM_minus_IPM_abs_error"] = piv["abs_log2_error_MEM"] - piv["abs_log2_error_IPM"]
    piv["MEM_worse_than_IPM"] = piv["MEM_minus_IPM_abs_error"] > 0
    pair_groups = local_predictions[
        ["genome_id", "ipm_mem_pair_group"]
    ].drop_duplicates("genome_id")
    piv = piv.merge(pair_groups, on="genome_id", how="left")
    return piv.merge(mechanisms, on="genome_id", how="left")


def summarize_paired_by(paired: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, sub in paired.groupby(group_col, dropna=False):
        rows.append(
            {
                group_col: value,
                "n_paired": int(len(sub)),
                "mean_ipm_abs_error": float(sub["abs_log2_error_IPM"].mean()),
                "mean_mem_abs_error": float(sub["abs_log2_error_MEM"].mean()),
                "mean_mem_minus_ipm_abs_error": float(sub["MEM_minus_IPM_abs_error"].mean()),
                "mem_worse_fraction": float(sub["MEM_worse_than_IPM"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_paired", ascending=False).reset_index(drop=True)


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
    variant_rows: pd.DataFrame,
    gene_summary: pd.DataFrame,
    mechanisms: pd.DataFrame,
    local_summary: pd.DataFrame,
    local_high_conf_summary: pd.DataFrame,
    pair_refined: pd.DataFrame,
    pair_high_conf: pd.DataFrame,
    pair_ampc: pd.DataFrame,
    pair_efflux: pd.DataFrame,
    pair_ampc_disruptive: pd.DataFrame,
    pair_efflux_disruptive: pd.DataFrame,
) -> None:
    target_table = pd.DataFrame(
        [
            {
                "gene": t.gene,
                "locus_tag": t.locus_tag,
                "axis": t.axis,
                "strict_driver": t.strict_driver,
                "coords": f"{t.start}-{t.end}({t.strand})",
                "upstream_200bp": f"{t.upstream_start}-{t.upstream_end}",
            }
            for t in TARGETS
        ]
    )
    gene_counts = (
        gene_summary.assign(has_non_syn_or_reg=lambda x: x["n_non_syn_or_regulatory"].gt(0))
        .groupby(["axis", "gene"], as_index=False)
        .agg(
            isolates_with_any_variant=("variant_count", lambda s: int((s > 0).sum())),
            isolates_with_non_syn_or_reg=("has_non_syn_or_reg", "sum"),
            disruptive_or_uncertain=(
                "severity",
                lambda s: int(s.isin(["disruptive", "coding_indel_uncertain"]).sum()),
            ),
        )
        .sort_values(["axis", "gene"])
    )
    refined_counts = (
        mechanisms.groupby("mechanism_subtype_refined", as_index=False)
        .agg(
            isolates=("genome_id", "nunique"),
            oprd_deep_disruptive=("oprd_deep_disruptive", "sum"),
            efflux_strict_non_syn=("efflux_strict_driver_non_syn_any", "sum"),
            ampc_core_non_syn=("ampc_core_driver_non_syn_any", "sum"),
            ampc_core_disruptive=("ampc_core_driver_disruptive_any", "sum"),
        )
        .sort_values("isolates", ascending=False)
    )
    high_conf_counts = (
        mechanisms.groupby("mechanism_subtype_high_confidence", as_index=False)
        .agg(
            isolates=("genome_id", "nunique"),
            oprd_severe_loss=("oprd_severe_loss", "sum"),
            oprd_deep_disruptive=("oprd_deep_disruptive", "sum"),
            efflux_strict_disruptive=("efflux_strict_driver_disruptive_any", "sum"),
            ampc_core_disruptive=("ampc_core_driver_disruptive_any", "sum"),
            carbapenemase=("acquired_carbapenemase_strict", "sum"),
        )
        .sort_values("isolates", ascending=False)
    )

    report = OUT / f"deep_mechanism_annotation_report_{DATE}.md"
    lines = [
        "# Deep Local Mechanism Annotation for IPM/MEM Predictability",
        "",
        f"Date: {DATE}",
        "",
        "## Why This Adds Depth",
        "",
        "The first-pass subtype table was intentionally strict and sparse, especially for AmpC-associated resistance.",
        "This analysis re-parses local Snippy variant tables to capture coding and proximal upstream variants in OprD, efflux regulators, and the AmpC regulatory axis.",
        "It separates strict driver genes from broad context genes so the manuscript can distinguish mechanism-informed predictability from phylogenetic background.",
        "",
        "## Target Genes",
        "",
        markdown_table(target_table),
        "",
        "## Local Variant Burden By Gene",
        "",
        markdown_table(gene_counts),
        "",
        "## Refined Local Mechanism Counts",
        "",
        markdown_table(refined_counts),
        "",
        "## High-Confidence Local Mechanism Counts",
        "",
        "Because PAO1-relative missense variation is very common in several loci, this stricter layer keeps only severe OprD loss, acquired carbapenemase, and disruptive/uncertain-indel signals in strict efflux or AmpC-core driver genes.",
        "",
        markdown_table(high_conf_counts),
        "",
        "## Local Prediction Performance By Refined Subtype",
        "",
        markdown_table(
            local_summary[
                [
                    "drug",
                    "mechanism_subtype_refined",
                    "n",
                    "mic_ea_pm1",
                    "mic_mae",
                    "mic_bias",
                    "large_error_gt2",
                    "sns_ca",
                    "sns_vme",
                    "sns_me",
                ]
            ]
        ),
        "",
        "## Local Prediction Performance By High-Confidence Subtype",
        "",
        markdown_table(
            local_high_conf_summary[
                [
                    "drug",
                    "mechanism_subtype_high_confidence",
                    "n",
                    "mic_ea_pm1",
                    "mic_mae",
                    "mic_bias",
                    "large_error_gt2",
                    "sns_ca",
                    "sns_vme",
                    "sns_me",
                ]
            ]
        ),
        "",
        "## Paired IPM/MEM Error Difference",
        "",
        "### By Refined Subtype",
        "",
        markdown_table(pair_refined),
        "",
        "### By High-Confidence Subtype",
        "",
        markdown_table(pair_high_conf),
        "",
        "### By AmpC Core Driver Flag",
        "",
        markdown_table(pair_ampc),
        "",
        "### By AmpC Core Disruptive Flag",
        "",
        markdown_table(pair_ampc_disruptive),
        "",
        "### By Strict Efflux Driver Flag",
        "",
        markdown_table(pair_efflux),
        "",
        "### By Strict Efflux Disruptive Flag",
        "",
        markdown_table(pair_efflux_disruptive),
        "",
        "## Interpretation",
        "",
        "- This table is still genomic evidence, not expression evidence. AmpC or efflux overexpression should be described as genotype-associated unless RNA/protein/functional assays are added.",
        "- If AmpC-core or efflux-driver subgroups retain large MEM underprediction while IPM remains relatively accurate, the article gains a stronger mechanism-specific failure model.",
        "- PAO1-relative non-synonymous calls are too broad for final mechanism labeling in this local set; the high-confidence disruptive layer is more defensible for manuscript subtyping.",
        "",
        "## Output Files",
        "",
        f"- `target_variant_rows_{DATE}.tsv`: {len(variant_rows)} target-region variants.",
        f"- `target_gene_variant_summary_{DATE}.tsv`: isolate x gene severity calls.",
        f"- `local_deep_mechanism_evidence_{DATE}.tsv`: isolate-level refined mechanism flags.",
        f"- `local_prediction_errors_with_deep_mechanism_{DATE}.tsv`: joined prediction-error table.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variant_rows, gene_summary = parse_local_target_variants()
    mechanisms = summarize_isolate_mechanisms(gene_summary)

    first = pd.read_csv(FIRST_PASS_MECHANISMS, sep="\t", low_memory=False)
    first = first[first["cohort"].eq("locked_local_validation")].copy()
    merged_mechanisms = first.merge(
        mechanisms.drop(columns=["cohort"]),
        on="genome_id",
        how="left",
        validate="one_to_one",
    )
    merged_mechanisms["mechanism_subtype_refined"] = derive_refined_subtype(merged_mechanisms)
    merged_mechanisms["mechanism_subtype_high_confidence"] = derive_high_confidence_subtype(merged_mechanisms)

    predictions = pd.read_csv(PREDICTION_ERRORS, sep="\t", low_memory=False)
    local_predictions = predictions[predictions["cohort"].eq("locked_local_validation")].copy()
    joined = local_predictions.merge(
        merged_mechanisms.drop(columns=["cohort"]),
        on="genome_id",
        how="left",
        validate="many_to_one",
    )

    local_summary = summarize_by(joined, ["drug", "mechanism_subtype_refined"])
    local_high_conf_summary = summarize_by(joined, ["drug", "mechanism_subtype_high_confidence"])
    pair_table = build_paired_error_table(local_predictions, merged_mechanisms)
    pair_refined = summarize_paired_by(pair_table, "mechanism_subtype_refined")
    pair_high_conf = summarize_paired_by(pair_table, "mechanism_subtype_high_confidence")
    pair_ampc = summarize_paired_by(pair_table, "ampc_core_driver_non_syn_any")
    pair_efflux = summarize_paired_by(pair_table, "efflux_strict_driver_non_syn_any")
    pair_ampc_disruptive = summarize_paired_by(pair_table, "ampc_core_driver_disruptive_any")
    pair_efflux_disruptive = summarize_paired_by(pair_table, "efflux_strict_driver_disruptive_any")

    variant_rows.to_csv(OUT / f"target_variant_rows_{DATE}.tsv", sep="\t", index=False)
    gene_summary.to_csv(OUT / f"target_gene_variant_summary_{DATE}.tsv", sep="\t", index=False)
    merged_mechanisms.to_csv(OUT / f"local_deep_mechanism_evidence_{DATE}.tsv", sep="\t", index=False)
    joined.to_csv(OUT / f"local_prediction_errors_with_deep_mechanism_{DATE}.tsv", sep="\t", index=False)
    local_summary.to_csv(OUT / f"local_refined_subtype_prediction_performance_{DATE}.csv", index=False)
    local_high_conf_summary.to_csv(OUT / f"local_high_confidence_subtype_prediction_performance_{DATE}.csv", index=False)
    pair_table.to_csv(OUT / f"local_paired_ipm_mem_deep_mechanism_errors_{DATE}.csv", index=False)
    pair_refined.to_csv(OUT / f"local_paired_error_by_refined_subtype_{DATE}.csv", index=False)
    pair_high_conf.to_csv(OUT / f"local_paired_error_by_high_confidence_subtype_{DATE}.csv", index=False)
    pair_ampc.to_csv(OUT / f"local_paired_error_by_ampc_core_flag_{DATE}.csv", index=False)
    pair_efflux.to_csv(OUT / f"local_paired_error_by_efflux_strict_flag_{DATE}.csv", index=False)
    pair_ampc_disruptive.to_csv(OUT / f"local_paired_error_by_ampc_core_disruptive_flag_{DATE}.csv", index=False)
    pair_efflux_disruptive.to_csv(OUT / f"local_paired_error_by_efflux_strict_disruptive_flag_{DATE}.csv", index=False)

    write_report(
        variant_rows,
        gene_summary,
        merged_mechanisms,
        local_summary,
        local_high_conf_summary,
        pair_refined,
        pair_high_conf,
        pair_ampc,
        pair_efflux,
        pair_ampc_disruptive,
        pair_efflux_disruptive,
    )


if __name__ == "__main__":
    main()
