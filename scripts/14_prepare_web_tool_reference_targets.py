#!/usr/bin/env python3
"""Prepare compact PAO1 target-gene references for the FASTA web tool."""

from __future__ import annotations

from pathlib import Path
import json


PROJECT = Path(__file__).resolve().parents[1]
FASTA = PROJECT / "data" / "reference" / "pao1" / "PAO1.fasta"
OUT = PROJECT / "web_fasta_tool" / "assets" / "reference_targets.json"
OUT_JS = PROJECT / "web_fasta_tool" / "assets" / "reference_targets.js"


TARGETS = [
    {"gene": "oprD", "locus_tag": "PA0958", "axis": "oprd", "strict_driver": True, "start": 1060548, "end": 1062245, "strand": "-"},
    {"gene": "mexR", "locus_tag": "PA0424", "axis": "efflux", "strict_driver": True, "start": 470060, "end": 470596, "strand": "-"},
    {"gene": "nalC", "locus_tag": "PA3720", "axis": "efflux_context", "strict_driver": False, "start": 4185165, "end": 4185905, "strand": "+"},
    {"gene": "nalD", "locus_tag": "PA3574", "axis": "efflux", "strict_driver": True, "start": 4018226, "end": 4018960, "strand": "-"},
    {"gene": "mexT", "locus_tag": "PA2492", "axis": "efflux_context", "strict_driver": False, "start": 2788905, "end": 2790266, "strand": "-"},
    {"gene": "mexS", "locus_tag": "PA2491", "axis": "efflux", "strict_driver": True, "start": 2788163, "end": 2788900, "strand": "-"},
    {"gene": "mexZ", "locus_tag": "PA2020", "axis": "efflux", "strict_driver": True, "start": 2251397, "end": 2252014, "strand": "+"},
    {"gene": "nfxB", "locus_tag": "PA4596", "axis": "efflux", "strict_driver": True, "start": 5165027, "end": 5165758, "strand": "+"},
    {"gene": "ampR", "locus_tag": "PA4109", "axis": "ampc_axis", "strict_driver": True, "start": 4592990, "end": 4593880, "strand": "-"},
    {"gene": "ampC", "locus_tag": "PA4110", "axis": "ampc_axis_context", "strict_driver": False, "start": 4594029, "end": 4595222, "strand": "+"},
    {"gene": "ampD", "locus_tag": "PA4522", "axis": "ampc_axis", "strict_driver": True, "start": 5064774, "end": 5065340, "strand": "-"},
    {"gene": "dacB", "locus_tag": "PA3047", "axis": "ampc_axis", "strict_driver": True, "start": 3410264, "end": 3411694, "strand": "+"},
]


def read_fasta(path: Path) -> str:
    parts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            continue
        parts.append(line.strip().upper())
    return "".join(parts)


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def main() -> None:
    genome = read_fasta(FASTA)
    records = []
    for target in TARGETS:
        seq = genome[target["start"] - 1 : target["end"]]
        if target["strand"] == "-":
            seq = reverse_complement(seq)
        rec = dict(target)
        rec["sequence"] = seq
        rec["length_bp"] = len(seq)
        records.append(rec)

    payload = {
        "reference": "Pseudomonas aeruginosa PAO1",
        "source_fasta": str(FASTA.relative_to(PROJECT)).replace("\\", "/"),
        "kmer_size": 21,
        "version_date": "2026-06-05",
        "targets": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    OUT_JS.write_text(
        "window.IPM_GPT_REFERENCE_TARGETS = " + text + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} targets -> {OUT}")
    print(f"Wrote browser bundle -> {OUT_JS}")


if __name__ == "__main__":
    main()
