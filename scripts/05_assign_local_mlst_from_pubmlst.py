#!/usr/bin/env python3
"""Assign local *P. aeruginosa* MLST ST from downloaded PubMLST alleles.

This is a lightweight exact-match MLST caller for the IPM-GPT local cohort.
It uses the PubMLST P. aeruginosa scheme downloaded under
`data/reference/pubmlst_paeruginosa_mlst/`.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATE = "2026-06-05"
SCHEME_DIR = PROJECT / "data" / "reference" / "pubmlst_paeruginosa_mlst"
METADATA = PROJECT / "data" / "metadata" / "clean_metadata.tsv"
OUT = PROJECT / "results" / "05_mlst"
LOCI = ["acsA", "aroE", "guaA", "mutL", "nuoD", "ppsA", "trpE"]


def load_profiles() -> dict[tuple[str, ...], dict[str, str]]:
    profiles = pd.read_csv(SCHEME_DIR / "profiles.csv", sep="\t", dtype=str).fillna("")
    profile_map = {}
    for _, row in profiles.iterrows():
        key = tuple(str(row[locus]) for locus in LOCI)
        profile_map[key] = {
            "ST": str(row["ST"]),
            "clonal_complex": str(row.get("clonal_complex", "")),
        }
    return profile_map


def load_alleles() -> dict[str, dict[str, object]]:
    pattern_dir = OUT / "rg_patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    loci = {}
    for locus in LOCI:
        seq_to_ids: dict[str, set[str]] = {}
        fasta = SCHEME_DIR / f"{locus}.fasta"
        for record in SeqIO.parse(fasta, "fasta"):
            allele_id = record.id.split("_")[-1]
            seq = str(record.seq).upper()
            rc = str(Seq(seq).reverse_complement())
            seq_to_ids.setdefault(seq, set()).add(allele_id)
            seq_to_ids.setdefault(rc, set()).add(allele_id)
        pattern_file = pattern_dir / f"{locus}_allele_patterns.txt"
        pattern_file.write_text("\n".join(sorted(seq_to_ids)) + "\n", encoding="ascii")
        # Keep a compiled regex fallback, but the main caller uses ripgrep's
        # Aho-Corasick fixed-string search for speed on Windows.
        pattern = "|".join(re.escape(seq) for seq in sorted(seq_to_ids, key=len, reverse=True))
        loci[locus] = {
            "seq_to_ids": seq_to_ids,
            "pattern": re.compile(pattern.encode("ascii")),
            "pattern_file": pattern_file,
        }
    return loci


def read_assembly(path: str) -> bytes:
    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append(str(record.seq).upper())
    return "N".join(records).encode("ascii", errors="ignore")


def call_locus_regex(genome_seq: bytes, allele_info: dict[str, object]) -> tuple[str, str, int]:
    hits = set()
    seq_to_ids = allele_info["seq_to_ids"]
    pattern = allele_info["pattern"]
    assert isinstance(seq_to_ids, dict)
    assert hasattr(pattern, "finditer")
    for match in pattern.finditer(genome_seq):
        allele_seq = match.group(0).decode("ascii")
        hits.update(seq_to_ids.get(allele_seq, set()))
    if len(hits) == 1:
        return next(iter(hits)), "exact_single", 1
    if len(hits) > 1:
        return ";".join(sorted(hits, key=lambda x: int(x) if x.isdigit() else x)), "multiple_exact", len(hits)
    return "", "missing", 0


def write_temp_genome(path: str, temp_dir: Path, isolate_id: str) -> Path:
    temp_path = temp_dir / f"{isolate_id}.one_line.fasta"
    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append(str(record.seq).upper())
    temp_path.write_text("N".join(records) + "\n", encoding="ascii")
    return temp_path


def call_locus_rg(temp_genome: Path, allele_info: dict[str, object]) -> tuple[str, str, int]:
    seq_to_ids = allele_info["seq_to_ids"]
    pattern_file = allele_info["pattern_file"]
    assert isinstance(seq_to_ids, dict)
    assert isinstance(pattern_file, Path)
    cmd = [
        "rg",
        "--fixed-strings",
        "--only-matching",
        "--no-line-number",
        "--no-filename",
        "--color",
        "never",
        "-f",
        str(pattern_file),
        str(temp_genome),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"rg failed for {temp_genome}: {result.stderr.strip()}")
    hits = set()
    if result.stdout:
        for line in result.stdout.splitlines():
            hits.update(seq_to_ids.get(line.strip().upper(), set()))
    if len(hits) == 1:
        return next(iter(hits)), "exact_single", 1
    if len(hits) > 1:
        return ";".join(sorted(hits, key=lambda x: int(x) if x.isdigit() else x)), "multiple_exact", len(hits)
    return "", "missing", 0


def call_isolate(isolate_id: str, assembly_path: str, alleles, profiles) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "isolate_id": isolate_id,
        "assembly_path": assembly_path,
        "mlst_scheme": "PubMLST Pseudomonas aeruginosa scheme 1",
    }
    path = Path(assembly_path)
    if not path.exists():
        row.update({"ST": "", "clonal_complex": "", "mlst_call_status": "assembly_missing"})
        return row

    temp_dir = OUT / "tmp_one_line_genomes"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_genome = write_temp_genome(str(path), temp_dir, isolate_id)
    allele_profile = []
    locus_statuses = []
    total_hits = 0
    for locus in LOCI:
        allele, status, n_hits = call_locus_rg(temp_genome, alleles[locus])
        row[locus] = allele
        row[f"{locus}_status"] = status
        row[f"{locus}_n_exact_hits"] = n_hits
        allele_profile.append(allele)
        locus_statuses.append(status)
        total_hits += n_hits
    try:
        temp_genome.unlink()
    except OSError:
        pass

    key = tuple(allele_profile)
    profile = profiles.get(key, {"ST": "", "clonal_complex": ""})
    row["ST"] = profile["ST"]
    row["clonal_complex"] = profile["clonal_complex"]
    row["allelic_profile"] = "-".join(allele_profile)
    row["n_loci_exact_single"] = sum(status == "exact_single" for status in locus_statuses)
    row["n_loci_missing"] = sum(status == "missing" for status in locus_statuses)
    row["n_loci_multiple"] = sum(status == "multiple_exact" for status in locus_statuses)
    row["n_total_exact_hits"] = total_hits
    if row["ST"]:
        row["mlst_call_status"] = "ST_exact_profile"
    elif row["n_loci_exact_single"] == len(LOCI):
        row["mlst_call_status"] = "novel_or_unprofiled_exact_alleles"
    else:
        row["mlst_call_status"] = "incomplete_or_ambiguous"
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    safe = frame.copy().astype(str)
    cols = list(safe.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in safe.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_report(calls: pd.DataFrame) -> None:
    scheme = json.loads((SCHEME_DIR / "scheme_1.json").read_text(encoding="utf-8"))
    status = calls["mlst_call_status"].value_counts().reset_index()
    status.columns = ["mlst_call_status", "isolates"]
    top_st = calls[calls["ST"].astype(str).str.len().gt(0)]["ST"].value_counts().head(20).reset_index()
    top_st.columns = ["ST", "isolates"]
    lines = [
        "# Local MLST Assignment",
        "",
        f"Date: {DATE}",
        "",
        "## Scheme",
        "",
        "- Source: PubMLST REST API, `pubmlst_paeruginosa_seqdef`, scheme 1.",
        f"- Scheme last updated: {scheme.get('last_updated', '')}",
        f"- PubMLST message: {scheme.get('message', '')}",
        "",
        "## Call Status",
        "",
        markdown_table(status),
        "",
        "## Top STs",
        "",
        markdown_table(top_st),
        "",
        "## Method",
        "",
        "Each MLST allele was called by exact nucleotide match against the assembly, including reverse-complement sequences. ST was assigned only when all seven exact allele calls matched a PubMLST profile.",
    ]
    (OUT / f"local_mlst_report_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(METADATA, sep="\t", dtype=str).fillna("")
    local = meta[meta["data_origin"].eq("local_clinical")].copy()
    alleles = load_alleles()
    profiles = load_profiles()
    rows = []
    for _, rec in local.sort_values("isolate_id").iterrows():
        rows.append(call_isolate(rec["isolate_id"], rec["assembly_path"], alleles, profiles))
    calls = pd.DataFrame(rows)
    calls.to_csv(OUT / f"local_mlst_pubmlst_exact_{DATE}.tsv", sep="\t", index=False)
    summary = calls["ST"].replace("", pd.NA).value_counts(dropna=True).reset_index()
    summary.columns = ["ST", "isolates"]
    summary.to_csv(OUT / f"local_mlst_st_counts_{DATE}.csv", index=False)
    write_report(calls)
    print(calls["mlst_call_status"].value_counts().to_string())
    print(summary.head(20).to_string(index=False))
    print(f"Wrote local MLST calls to {OUT}")


if __name__ == "__main__":
    main()
