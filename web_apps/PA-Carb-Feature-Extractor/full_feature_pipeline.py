from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
MODEL_PATH = ASSET_DIR / "ipm_mem_unified_public_only_model_2026-06-05.joblib"
FEATURE_META_PATH = ASSET_DIR / "locked_model_feature_metadata.json"
ARO_NORM_REF_PATH = ASSET_DIR / "aro_norm_reference_public_training.csv"
PROTEIN_INDEX_PATH = ASSET_DIR / "protein_index.csv"
PROTEIN_EMBEDDINGS_PATH = ASSET_DIR / "protein_embeddings.npy"
CARD_DIR = ASSET_DIR / "card"
PAO1_DIR = ASSET_DIR / "pao1"
PAO1_REF = PAO1_DIR / "PAO1.embl"
CARD_PROTEIN_DB = CARD_DIR / "localDB" / "protein.db.dmnd"
CARD_PROTEIN_FASTA = CARD_DIR / "localDB" / "proteindb.fsa"
CARD_NUCLEOTIDE_FASTA = CARD_DIR / "nucleotide_fasta_protein_homolog_model.fasta"
CARD_JSON = CARD_DIR / "card.json"

MAX_SEQ_LEN = 1000
OPRD_FULL_LEN = 457.0
OPRD_TRUNCATION_FRACTION = 0.9
DRUGS = ["IPM", "MEM"]
EFFLUX_MUT_COLS = ["mexR_mut", "nalC_mut", "nalD_mut", "mexT_mut", "mexS_mut", "mexZ_mut", "nfxB_mut", "efflux_regulator_any"]
REGULATOR_GENES = ["mexR", "nalC", "nalD", "mexT", "mexS", "mexZ", "nfxB"]


class FullFeatureUnavailable(RuntimeError):
    pass


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def tool_path(*names: str) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def diamond_tool() -> str | None:
    return tool_path("diamond", "diamond-aligner")


def full_feature_ready() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not tool_available("snippy"):
        missing.append("snippy")
    rgi_ready = tool_available("rgi")
    diamond_ready = diamond_tool() is not None and tool_available("prodigal") and CARD_PROTEIN_DB.exists() and CARD_PROTEIN_FASTA.exists() and CARD_JSON.exists()
    blastn_ready = tool_available("blastn") and tool_available("makeblastdb") and CARD_NUCLEOTIDE_FASTA.exists() and CARD_JSON.exists()
    if not rgi_ready and not diamond_ready and not blastn_ready:
        amr_missing = []
        if not tool_available("rgi"):
            amr_missing.append("rgi")
        if diamond_tool() is None:
            amr_missing.append("diamond or diamond-aligner")
        if not tool_available("prodigal"):
            amr_missing.append("prodigal")
        if not tool_available("blastn"):
            amr_missing.append("blastn")
        if not tool_available("makeblastdb"):
            amr_missing.append("makeblastdb")
        if not CARD_PROTEIN_DB.exists():
            amr_missing.append("assets/card/localDB/protein.db.dmnd")
        if not CARD_PROTEIN_FASTA.exists():
            amr_missing.append("assets/card/localDB/proteindb.fsa")
        if not CARD_NUCLEOTIDE_FASTA.exists():
            amr_missing.append("assets/card/nucleotide_fasta_protein_homolog_model.fasta")
        missing.append("AMR detector unavailable; missing " + ", ".join(amr_missing))
    for path in [MODEL_PATH, FEATURE_META_PATH, ARO_NORM_REF_PATH, PROTEIN_INDEX_PATH, PROTEIN_EMBEDDINGS_PATH, CARD_DIR, PAO1_REF]:
        if not path.exists():
            missing.append(str(path.relative_to(APP_DIR)))
    return not missing, missing


def read_feature_columns() -> list[str]:
    payload = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    return list(payload["feature_columns"])


def load_norm_reference() -> dict[str, float]:
    if not ARO_NORM_REF_PATH.exists():
        return {}
    ref = pd.read_csv(ARO_NORM_REF_PATH)
    return {
        str(row["aro"]): float(row["median_norm_when_present"])
        for _, row in ref.iterrows()
        if pd.notna(row.get("aro")) and pd.notna(row.get("median_norm_when_present"))
    }


def load_embedding_cache() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not PROTEIN_INDEX_PATH.exists() or not PROTEIN_EMBEDDINGS_PATH.exists():
        return {}, {}
    index = pd.read_csv(PROTEIN_INDEX_PATH)
    embeddings = np.load(PROTEIN_EMBEDDINGS_PATH, mmap_mode="r")
    seq_hash_to_embedding: dict[str, np.ndarray] = {}
    seq_hash_to_norm: dict[str, float] = {}
    for i, row in index.iterrows():
        if i >= len(embeddings):
            break
        seq_hash = str(row["seq_hash"])
        emb = np.asarray(embeddings[i], dtype=np.float32)
        seq_hash_to_embedding[seq_hash] = emb
        seq_hash_to_norm[seq_hash] = float(np.linalg.norm(emb))
    return seq_hash_to_embedding, seq_hash_to_norm


def sequence_hash(seq: Any) -> str:
    seq_s = "" if pd.isna(seq) else str(seq)[:MAX_SEQ_LEN]
    return hashlib.md5(seq_s.encode()).hexdigest()[:12] if seq_s else ""


def run_command(cmd: list[str], *, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)


def run_rgi(fasta: Path, work_dir: Path, threads: int = 2) -> Path:
    out_prefix = work_dir / "rgi" / "query"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rgi",
        "main",
        "-i",
        str(fasta),
        "-o",
        str(out_prefix),
        "-t",
        "contig",
        "--local",
        "--clean",
        "-n",
        str(threads),
        "--low_quality",
    ]
    result = run_command(cmd, cwd=CARD_DIR if CARD_DIR.exists() else None, timeout=900)
    out_txt = out_prefix.with_suffix(".txt")
    if result.returncode != 0 or not out_txt.exists():
        message = (result.stderr or result.stdout or "RGI produced no output").strip()[-1200:]
        raise FullFeatureUnavailable(f"RGI failed: {message}")
    return out_txt


def read_fasta_sequences(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and parts:
                    records[name] = "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name is not None and parts:
        records[name] = "".join(parts)
    return records


def load_card_subject_metadata() -> dict[str, dict[str, float | str]]:
    card_payload = json.loads(CARD_JSON.read_text(encoding="utf-8"))
    meta: dict[str, dict[str, float | str]] = {}
    with CARD_PROTEIN_FASTA.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if not raw.startswith(">"):
                continue
            header = raw[1:].strip()
            subject_id = header.split()[0]
            model_id = subject_id.split("_", 1)[0]
            payload = card_payload.get(model_id, {})
            if not isinstance(payload, dict):
                payload = {}
            aro_name = str(payload.get("ARO_name") or header.split("|")[-1].strip())
            pass_bitscore = 0.0
            match = re.search(r"pass_bitscore:\s*([0-9.]+)", header)
            if match:
                pass_bitscore = float(match.group(1))
            meta[subject_id] = {"aro_name": aro_name, "pass_bitscore": pass_bitscore}
    return meta


def load_card_aro_accession_metadata() -> dict[str, str]:
    card_payload = json.loads(CARD_JSON.read_text(encoding="utf-8"))
    meta: dict[str, str] = {}
    for payload in card_payload.values():
        if not isinstance(payload, dict):
            continue
        accession = payload.get("ARO_accession")
        name = payload.get("ARO_name")
        if accession and name:
            meta[f"ARO:{accession}"] = str(name)
            meta[str(accession)] = str(name)
    return meta


def run_card_diamond_fallback(fasta: Path, work_dir: Path, threads: int = 2) -> Path:
    """Approximate RGI strict/perfect AMR hits with bundled CARD DIAMOND assets.

    This is used only when the RGI binary is unavailable in the Space runtime.
    It preserves the feature interface expected by the locked model by emitting
    a small RGI-like table with Best_Hit_ARO, Predicted_Protein, and Cut_Off.
    """
    out_dir = work_dir / "card_diamond"
    out_dir.mkdir(parents=True, exist_ok=True)
    diamond_exe = diamond_tool()
    if diamond_exe is None:
        raise FullFeatureUnavailable("DIAMOND executable was not found as 'diamond' or 'diamond-aligner'.")
    runtime_db_prefix = out_dir / "card_runtime_protein"
    runtime_db = runtime_db_prefix.with_suffix(".dmnd")
    makedb_cmd = [
        diamond_exe,
        "makedb",
        "--in",
        str(CARD_PROTEIN_FASTA),
        "--db",
        str(runtime_db_prefix),
        "--quiet",
    ]
    makedb_result = run_command(makedb_cmd, timeout=600)
    if makedb_result.returncode != 0 or not runtime_db.exists():
        message = (makedb_result.stderr or makedb_result.stdout or "DIAMOND makedb produced no database").strip()[-1200:]
        raise FullFeatureUnavailable(f"DIAMOND CARD database rebuild failed: {message}")

    proteins = out_dir / "query.faa"
    prodigal_cmd = [
        "prodigal",
        "-i",
        str(fasta),
        "-a",
        str(proteins),
        "-p",
        "meta",
        "-q",
    ]
    prodigal_result = run_command(prodigal_cmd, timeout=900)
    if prodigal_result.returncode != 0 or not proteins.exists():
        message = (prodigal_result.stderr or prodigal_result.stdout or "Prodigal produced no protein FASTA").strip()[-1200:]
        raise FullFeatureUnavailable(f"Prodigal failed for CARD fallback: {message}")

    blast_out = out_dir / "card_diamond.tsv"
    diamond_cmd = [
        diamond_exe,
        "blastp",
        "--db",
        str(runtime_db_prefix),
        "--query",
        str(proteins),
        "--out",
        str(blast_out),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
        "--max-target-seqs",
        "1",
        "--evalue",
        "1e-10",
        "--threads",
        str(threads),
        "--quiet",
    ]
    diamond_result = run_command(diamond_cmd, timeout=900)
    if diamond_result.returncode != 0 or not blast_out.exists():
        message = (diamond_result.stderr or diamond_result.stdout or "DIAMOND produced no output").strip()[-1200:]
        raise FullFeatureUnavailable(f"DIAMOND CARD fallback failed: {message}")

    protein_seqs = read_fasta_sequences(proteins)
    card_meta = load_card_subject_metadata()
    rows: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    with blast_out.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            qseqid, sseqid = parts[0], parts[1]
            if qseqid in seen_queries:
                continue
            pident = float(parts[2])
            align_len = float(parts[3])
            qlen = float(parts[4])
            slen = float(parts[5])
            bitscore = float(parts[7])
            qcov = align_len / qlen if qlen else 0.0
            scov = align_len / slen if slen else 0.0
            subject_meta = card_meta.get(sseqid, {})
            pass_bitscore = float(subject_meta.get("pass_bitscore", 0.0) or 0.0)
            passes_bitscore = pass_bitscore > 0 and bitscore >= pass_bitscore
            passes_identity = pident >= 95.0 and qcov >= 0.80 and scov >= 0.80
            if not (passes_bitscore or passes_identity):
                continue
            seen_queries.add(qseqid)
            rows.append(
                {
                    "Best_Hit_ARO": subject_meta.get("aro_name", sseqid),
                    "Predicted_Protein": protein_seqs.get(qseqid, ""),
                    "Cut_Off": "Strict" if passes_bitscore else "Perfect",
                    "CARD_fallback_identity": pident,
                    "CARD_fallback_qcov": round(qcov, 4),
                    "CARD_fallback_scov": round(scov, 4),
                    "CARD_fallback_bitscore": bitscore,
                }
            )

    out_txt = out_dir / "query.card_fallback.txt"
    pd.DataFrame(rows).to_csv(out_txt, sep="\t", index=False)
    return out_txt


def run_card_blastn_fallback(fasta: Path, work_dir: Path, threads: int = 2) -> Path:
    """AMR gene fallback using BLASTN against bundled CARD nucleotide homologs.

    This is less rich than RGI but preserves the manuscript feature interface
    better than returning to the lightweight target-locus screen.
    """
    out_dir = work_dir / "card_blastn"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = out_dir / "card_nucleotide"
    make_db_cmd = [
        "makeblastdb",
        "-in",
        str(CARD_NUCLEOTIDE_FASTA),
        "-dbtype",
        "nucl",
        "-out",
        str(db_prefix),
    ]
    make_db = run_command(make_db_cmd, timeout=300)
    if make_db.returncode != 0:
        message = (make_db.stderr or make_db.stdout or "makeblastdb failed").strip()[-1200:]
        raise FullFeatureUnavailable(f"CARD BLASTN fallback database build failed: {message}")

    blast_out = out_dir / "card_blastn.tsv"
    blast_cmd = [
        "blastn",
        "-query",
        str(fasta),
        "-db",
        str(db_prefix),
        "-out",
        str(blast_out),
        "-outfmt",
        "6 qseqid sseqid pident length qlen slen evalue bitscore",
        "-max_target_seqs",
        "5",
        "-evalue",
        "1e-20",
        "-num_threads",
        str(threads),
    ]
    blast = run_command(blast_cmd, timeout=1200)
    if blast.returncode != 0 or not blast_out.exists():
        message = (blast.stderr or blast.stdout or "blastn produced no output").strip()[-1200:]
        raise FullFeatureUnavailable(f"CARD BLASTN fallback failed: {message}")

    aro_meta = load_card_aro_accession_metadata()
    rows: list[dict[str, object]] = []
    seen_aro: set[str] = set()
    with blast_out.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            sseqid = parts[1]
            pident = float(parts[2])
            align_len = float(parts[3])
            qlen = float(parts[4])
            slen = float(parts[5])
            bitscore = float(parts[7])
            scov = align_len / slen if slen else 0.0
            if pident < 95.0 or scov < 0.80:
                continue
            aro_match = re.search(r"ARO:\d+", sseqid)
            aro_id = aro_match.group(0) if aro_match else ""
            aro_name = aro_meta.get(aro_id, "")
            if not aro_name:
                # Fall back to the terminal header field before any organism label.
                fields = sseqid.split("|")
                aro_name = fields[-1].split("[", 1)[0].strip() if fields else sseqid
            if aro_name in seen_aro:
                continue
            seen_aro.add(aro_name)
            rows.append(
                {
                    "Best_Hit_ARO": aro_name,
                    "Cut_Off": "Strict",
                    "CARD_blastn_identity": pident,
                    "CARD_blastn_subject_coverage": round(scov, 4),
                    "CARD_blastn_bitscore": bitscore,
                }
            )

    out_txt = out_dir / "query.card_blastn_fallback.txt"
    pd.DataFrame(rows).to_csv(out_txt, sep="\t", index=False)
    return out_txt


def run_amr_detection(fasta: Path, work_dir: Path, threads: int = 2) -> tuple[Path, str]:
    if tool_available("rgi"):
        try:
            return run_rgi(fasta, work_dir, threads=threads), "RGI/CARD"
        except FullFeatureUnavailable:
            pass
    if diamond_tool() is not None and tool_available("prodigal"):
        try:
            return run_card_diamond_fallback(fasta, work_dir, threads=threads), "DIAMOND/CARD fallback"
        except FullFeatureUnavailable:
            pass
    if tool_available("blastn") and tool_available("makeblastdb"):
        return run_card_blastn_fallback(fasta, work_dir, threads=threads), "BLASTN/CARD nucleotide fallback"
    raise FullFeatureUnavailable("No AMR gene detector available: install rgi, or provide diamond+prodigal or blastn+makeblastdb with bundled CARD assets.")


def run_snippy(fasta: Path, work_dir: Path, threads: int = 2) -> Path:
    out_dir = work_dir / "snippy"
    cmd = [
        "snippy",
        "--ctgs",
        str(fasta),
        "--ref",
        str(PAO1_REF),
        "--outdir",
        str(out_dir),
        "--prefix",
        "snps",
        "--cpus",
        str(threads),
        "--force",
        "--quiet",
    ]
    result = run_command(cmd, timeout=1200)
    out_tab = out_dir / "snps.tab"
    if result.returncode != 0 or not out_tab.exists():
        message = (result.stderr or result.stdout or "Snippy produced no output").strip()[-1200:]
        raise FullFeatureUnavailable(f"Snippy failed: {message}")
    return out_tab


def load_rgi_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "Cut_Off" in df.columns:
        df = df[df["Cut_Off"].isin(["Strict", "Perfect"])].copy()
    if "Predicted_Protein" in df.columns:
        df = df[df["Predicted_Protein"].astype(str).str.len().gt(20)].copy()
    return df


def rgi_to_amr_features(rgi: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, float], list[str]]:
    features = {col: 0.0 for col in feature_cols if col.endswith("__present") or col.endswith("__norm")}
    notes: list[str] = []
    norm_ref = load_norm_reference()
    seq_hash_to_embedding, seq_hash_to_norm = load_embedding_cache()

    if rgi.empty or "Best_Hit_ARO" not in rgi.columns:
        return features, ["RGI returned no strict/perfect AMR gene hits used by the locked model."]

    rgi = rgi.copy()
    rgi["Best_Hit_ARO"] = rgi["Best_Hit_ARO"].astype(str)
    if "Predicted_Protein" in rgi.columns:
        rgi["seq_hash"] = rgi["Predicted_Protein"].map(sequence_hash)
    else:
        rgi["seq_hash"] = ""

    unseen = 0
    for aro, group in rgi.groupby("Best_Hit_ARO"):
        present_col = f"{aro}__present"
        norm_col = f"{aro}__norm"
        if present_col not in features and norm_col not in features:
            continue
        if present_col in features:
            features[present_col] = 1.0
        embeddings = [seq_hash_to_embedding[h] for h in group["seq_hash"] if h in seq_hash_to_embedding]
        if embeddings:
            mean_emb = np.stack(embeddings).mean(axis=0)
            features[norm_col] = float(np.linalg.norm(mean_emb))
        elif norm_col in features:
            hashes = [h for h in group["seq_hash"] if h]
            known_norms = [seq_hash_to_norm[h] for h in hashes if h in seq_hash_to_norm]
            if known_norms:
                features[norm_col] = float(np.mean(known_norms))
            else:
                features[norm_col] = float(norm_ref.get(aro, 0.0))
                unseen += 1
    if unseen:
        notes.append(f"{unseen} AMR hit(s) used public-training median norm because the exact protein sequence was not in the ESM2 cache.")
    return features, notes


def beta_summary_features(rgi: pd.DataFrame) -> dict[str, float]:
    names = rgi.get("Best_Hit_ARO", pd.Series(dtype=str)).astype(str).str.upper()
    out = {
        "vim_any": names.str.contains(r"\bVIM-", regex=True).any(),
        "imp_any": names.str.contains(r"\bIMP-", regex=True).any(),
        "ndm_any": names.str.contains(r"\bNDM-", regex=True).any(),
        "kpc_any": names.str.contains(r"\bKPC-", regex=True).any(),
        "oxa48_any": names.str.contains(r"OXA-48|OXA48", regex=True).any(),
        "gim_any": names.str.contains(r"\bGIM-", regex=True).any(),
        "spm_any": names.str.contains(r"\bSPM-", regex=True).any(),
        "blaPDC_any": names.str.contains(r"\bPDC-", regex=True).any(),
        "blaOXA_any": names.str.contains(r"\bOXA-", regex=True).any(),
        "blaTEM_any": names.str.contains(r"\bTEM-", regex=True).any(),
        "blaSHV_any": names.str.contains(r"\bSHV-", regex=True).any(),
        "blaCTX_any": names.str.contains(r"\bCTX", regex=True).any(),
        "blaGES_any": names.str.contains(r"\bGES-", regex=True).any(),
        "blaVEB_any": names.str.contains(r"\bVEB-", regex=True).any(),
    }
    out["carbapenemase_any"] = any(out[k] for k in ["vim_any", "imp_any", "ndm_any", "kpc_any", "oxa48_any", "gim_any", "spm_any"])
    return {k: float(v) for k, v in out.items()}


AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Glu": "E",
    "Gln": "Q",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}
GYRA_RESIST_AA = {83: ["I", "N", "G", "Y", "V", "C"], 87: ["N", "G", "Y", "V"]}
PARC_RESIST_AA = {80: ["I", "F", "R"], 84: ["V", "K", "G"]}


def parse_aa_change(effect: Any) -> tuple[int, str, str] | None:
    if not isinstance(effect, str):
        return None
    match = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)", effect)
    if not match:
        return None
    ref3, pos, alt3 = match.group(1), int(match.group(2)), match.group(3)
    return pos, AA3_TO_1.get(ref3, "?"), AA3_TO_1.get(alt3, alt3[0] if alt3 else "?")


def effect_category(effect: Any, typ: Any) -> str:
    effect_s = str(effect).lower()
    type_s = str(typ).lower()
    if any(token in effect_s for token in ["stop_gained", "frameshift", "start_lost", "stop_lost", "*"]):
        return "high_conf_disruptive_effect"
    if "missense_variant" in effect_s or "missense" in effect_s:
        return "missense"
    if "synonymous_variant" in effect_s or "synonymous" in effect_s:
        return "synonymous"
    if type_s in {"del", "ins", "complex"}:
        return "indel_or_complex"
    return "other"


def variant_length_delta(ref: Any, alt: Any) -> int:
    ref_s = "" if pd.isna(ref) else str(ref)
    alt_s = "" if pd.isna(alt) else str(alt)
    return abs(len(alt_s) - len(ref_s))


def parse_aa_pos(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else math.nan


def load_snippy_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return pd.DataFrame()


def snippy_to_mutation_features(snippy: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {
        "oprD_disrupted": 0.0,
        "oprD_mutated": 0.0,
        "gyrA_QRDR_mut": 0.0,
        "parC_QRDR_mut": 0.0,
        "any_QRDR_mut": 0.0,
        **{f"{gene}_mut": 0.0 for gene in REGULATOR_GENES},
        "efflux_regulator_any": 0.0,
    }
    if snippy.empty or "GENE" not in snippy.columns:
        return features

    oprd = snippy[snippy["GENE"].eq("oprD")].copy()
    if not oprd.empty:
        effects = oprd.get("EFFECT", pd.Series(dtype=str)).astype(str)
        types = oprd.get("TYPE", pd.Series(dtype=str)).astype(str).str.lower()
        has_disruptive = effects.str.contains(r"stop_gained|frameshift|start_lost|\*", regex=True, na=False).any() or types.isin(["del", "ins", "complex"]).any()
        has_missense = effects.str.contains("missense", regex=False, na=False).any()
        features["oprD_disrupted"] = float(has_disruptive)
        features["oprD_mutated"] = float(has_disruptive or has_missense)

    for gene, positions in [("gyrA", GYRA_RESIST_AA), ("parC", PARC_RESIST_AA)]:
        sub = snippy[snippy["GENE"].eq(gene)]
        flag = False
        for _, row in sub.iterrows():
            parsed = parse_aa_change(row.get("EFFECT", ""))
            if parsed is None:
                continue
            pos, _, alt = parsed
            if pos in positions and alt in positions[pos]:
                flag = True
                break
        features[f"{gene}_QRDR_mut"] = float(flag)
    features["any_QRDR_mut"] = float(bool(features["gyrA_QRDR_mut"] or features["parC_QRDR_mut"]))

    for gene in REGULATOR_GENES:
        sub = snippy[snippy["GENE"].eq(gene)]
        if sub.empty:
            continue
        effects = sub.get("EFFECT", pd.Series(dtype=str)).astype(str)
        ns = effects.str.contains("missense|stop_gained|frameshift|start_lost", regex=True, na=False).any()
        features[f"{gene}_mut"] = float(ns)
    features["efflux_regulator_any"] = float(any(features[f"{gene}_mut"] for gene in REGULATOR_GENES))
    return features


def refined_oprd_features(snippy: pd.DataFrame) -> dict[str, float]:
    out = {
        "oprd_len": OPRD_FULL_LEN,
        "oprd_truncated": 0.0,
        "has_high_conf_disruptive_effect": 0.0,
        "n_high_conf_disruptive_effect": 0.0,
        "has_large_indel_10bp": 0.0,
        "n_large_indel_10bp": 0.0,
        "max_length_delta": 0.0,
    }
    if snippy.empty or "GENE" not in snippy.columns:
        return out
    oprd = snippy[snippy["GENE"].eq("oprD")].copy()
    if oprd.empty:
        return out
    oprd["effect_category"] = [effect_category(e, t) for e, t in zip(oprd.get("EFFECT", ""), oprd.get("TYPE", ""))]
    oprd["aa_pos_num"] = oprd.get("AA_POS", pd.Series([math.nan] * len(oprd))).map(parse_aa_pos)
    oprd["length_delta"] = [variant_length_delta(r, a) for r, a in zip(oprd.get("REF", ""), oprd.get("ALT", ""))]
    oprd["is_high_conf_disruptive"] = oprd["effect_category"].eq("high_conf_disruptive_effect").astype(int)
    oprd["is_large_indel_10bp"] = (oprd.get("TYPE", pd.Series(dtype=str)).astype(str).str.lower().isin(["del", "ins", "complex"]) & oprd["length_delta"].ge(10)).astype(int)
    disruptive = oprd[oprd["is_high_conf_disruptive"].eq(1)]
    if not disruptive.empty:
        positions = disruptive["aa_pos_num"].dropna()
        if not positions.empty:
            out["oprd_len"] = float(positions.min())
    out["has_high_conf_disruptive_effect"] = float(oprd["is_high_conf_disruptive"].any())
    out["n_high_conf_disruptive_effect"] = float(oprd["is_high_conf_disruptive"].sum())
    out["has_large_indel_10bp"] = float(oprd["is_large_indel_10bp"].any())
    out["n_large_indel_10bp"] = float(oprd["is_large_indel_10bp"].sum())
    out["max_length_delta"] = float(oprd["length_delta"].max()) if not oprd.empty else 0.0
    out["oprd_truncated"] = float(out["oprd_len"] < OPRD_FULL_LEN * OPRD_TRUNCATION_FRACTION)
    return out


def add_derived_features(df: pd.DataFrame) -> None:
    amino_cols = [
        col
        for col in df.columns
        if re.search(r"^(AAC|APH|ANT|aadA|rmt|arm|aphA|ant|aac|aph).+__present$", col)
        and "APH(3')-IIb" not in col
    ]
    if amino_cols:
        amino = df[amino_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["acq_amino_any_for_cap"] = amino.gt(0).any(axis=1)
        df["derived_acq_amino_count"] = amino.sum(axis=1)
    else:
        df["acq_amino_any_for_cap"] = False
        df["derived_acq_amino_count"] = 0

    beta_cols = [col for col in df.columns if re.search(r"^(PDC-|OXA-|GES-|VEB-|PER-|TEM-|SHV-|CTX|CARB-).+__present$", col)]
    carb_cols = [col for col in df.columns if re.search(r"^(VIM-|IMP-|NDM-|KPC-|SPM-|GIM-).+__present$", col)]
    df["derived_beta_burden_count"] = df[beta_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if beta_cols else 0
    df["derived_carbapenemase_gene_count"] = df[carb_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if carb_cols else 0

    efflux_cols = [col for col in EFFLUX_MUT_COLS if col in df.columns]
    df["derived_efflux_mut_count"] = df[efflux_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if efflux_cols else 0

    if "oprd_len" in df.columns:
        df["derived_oprd_len_lt_430"] = (pd.to_numeric(df["oprd_len"], errors="coerce") < 430).fillna(False).astype(int)
    else:
        df["derived_oprd_len_lt_430"] = 0

    parts = [col for col in ["oprD_disrupted", "oprd_truncated", "has_high_conf_disruptive_effect", "has_large_indel_10bp", "derived_oprd_len_lt_430"] if col in df.columns]
    df["derived_oprd_severity_count"] = df[parts].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if parts else 0


def build_feature_row(rgi: pd.DataFrame, snippy: pd.DataFrame, sample_name: str) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = read_feature_columns()
    row: dict[str, float | str] = {"genome_id": sample_name}
    row.update({col: 0.0 for col in feature_cols})
    amr_features, notes = rgi_to_amr_features(rgi, feature_cols)
    row.update(amr_features)
    row.update(beta_summary_features(rgi))
    row.update(snippy_to_mutation_features(snippy))
    row.update(refined_oprd_features(snippy))
    df = pd.DataFrame([row])
    add_derived_features(df)
    return df, notes


def aligned_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = {}
    for col in cols:
        if col in df.columns:
            data[col] = df[col].to_numpy()
        else:
            data[col] = np.zeros(len(df), dtype=float)
    return pd.DataFrame(data, index=df.index).apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)


def nearest_train_level(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(sorted(np.unique(levels)), dtype=float)
    if levels.size == 0:
        return pred
    idx = np.abs(pred.reshape(-1, 1) - levels.reshape(1, -1)).argmin(axis=1)
    return levels[idx]


def endpoint_base_predictions(df: pd.DataFrame, endpoint: dict[str, Any]) -> pd.DataFrame:
    x_gate = aligned_frame(df, endpoint["gate_feature_cols"])
    x_stage2 = aligned_frame(df, endpoint["stage2_feature_cols"])
    prob_ns = endpoint["gate_model"].predict_proba(x_gate)[:, 1]
    global_reg = endpoint["global_regressor"]
    s_reg = endpoint["s_regressor"] if endpoint["s_regressor"] is not None else global_reg
    ns_reg = endpoint["ns_regressor"] if endpoint["ns_regressor"] is not None else global_reg
    pred_ungated = global_reg.predict(x_stage2)
    pred_s = s_reg.predict(x_stage2)
    pred_ns = ns_reg.predict(x_stage2)
    pred_soft = (1.0 - prob_ns) * pred_s + prob_ns * pred_ns
    return pd.DataFrame(
        {
            "prob_ns": prob_ns,
            "pred_ungated": pred_ungated,
            "pred_s_branch": pred_s,
            "pred_ns_branch": pred_ns,
            "pred_soft_gate": pred_soft,
            "acq_amino_any_for_cap": df.get("acq_amino_any_for_cap", pd.Series([False] * len(df))).to_numpy(dtype=bool),
        }
    )


def apply_model_policy(base: pd.DataFrame, endpoint: dict[str, Any]) -> np.ndarray:
    policy = endpoint["policy"]
    breakpoint = float(policy["breakpoint_log2"])
    threshold = float(policy["threshold"])
    approach = str(policy["approach"])

    if approach == "ungated":
        pred = base["pred_ungated"].to_numpy(dtype=float).copy()
    elif approach == "soft_gate":
        pred = base["pred_soft_gate"].to_numpy(dtype=float).copy()
    elif approach == "hard_gate":
        hard_gate = (base["prob_ns"].to_numpy(dtype=float) >= threshold).astype(int)
        pred = np.where(hard_gate == 1, base["pred_ns_branch"].to_numpy(dtype=float), base["pred_s_branch"].to_numpy(dtype=float))
        pred[hard_gate == 0] = np.minimum(pred[hard_gate == 0], breakpoint)
        pred[hard_gate == 1] = np.maximum(pred[hard_gate == 1], breakpoint + 1.0)
    else:
        raise ValueError(f"Unknown model approach: {approach}")

    if bool(policy["snapped_to_train_levels"]):
        pred = nearest_train_level(pred, np.asarray(endpoint["train_mic_levels"], dtype=float))
    return pred


def format_mic(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value >= 1 and abs(value - round(value)) < 0.02:
        return str(int(round(value)))
    return f"{value:.3g}"


def predict_from_features(features: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise FullFeatureUnavailable(f"Locked model could not be loaded in this runtime: {exc}") from exc
    predictions: list[dict[str, Any]] = []
    for drug in DRUGS:
        endpoint = artifact["endpoints"][drug]
        policy = endpoint["policy"]
        base = endpoint_base_predictions(features, endpoint)
        pred_log2 = float(apply_model_policy(base, endpoint)[0])
        breakpoint_log2 = float(policy["breakpoint_log2"])
        sns = "NS" if pred_log2 > breakpoint_log2 else "S"
        predictions.append(
            {
                "drug": drug,
                "pred_mic_log2": round(pred_log2, 4),
                "pred_mic_mg_l": format_mic(float(2**pred_log2)),
                "breakpoint_mg_l": format_mic(float(2**breakpoint_log2)),
                "pred_sns": sns,
                "prob_ns": round(float(base["prob_ns"].iloc[0]), 4),
                "model_status": policy["status"],
                "prediction_type": "Full-feature FASTA pipeline estimate",
            }
        )
    return predictions


def mechanism_summary_from_features(features: pd.DataFrame) -> dict[str, Any]:
    row = features.iloc[0]
    oprd_disrupted = float(row.get("oprD_disrupted", 0)) > 0 or float(row.get("derived_oprd_severity_count", 0)) >= 2
    efflux = float(row.get("efflux_regulator_any", 0)) > 0
    ampc = any(float(row.get(col, 0)) > 0 for col in ["Pseudomonas aeruginosa ampR with mutation conferring resistance to aztreonam__present", "blaPDC_any"])
    carb = float(row.get("carbapenemase_any", 0)) > 0 or float(row.get("derived_carbapenemase_gene_count", 0)) > 0
    axes = []
    if oprd_disrupted:
        axes.append("OprD-loss")
    if efflux:
        axes.append("efflux-regulator")
    if ampc:
        axes.append("AmpC-axis")
    if carb:
        axes.append("acquired carbapenemase")
    if len(axes) >= 2:
        subtype = "High-confidence composite"
    elif oprd_disrupted:
        subtype = "OprD-loss"
    elif ampc:
        subtype = "AmpC-axis disruptive"
    else:
        subtype = "No high-confidence driver"
    return {
        "mechanism_subtype_high_confidence": subtype,
        "driver_axes": "; ".join(axes) if axes else "none detected",
        "oprd_len": float(row.get("oprd_len", math.nan)) if pd.notna(row.get("oprd_len", math.nan)) else "",
        "derived_oprd_severity_count": float(row.get("derived_oprd_severity_count", 0)),
    }


def analyze_fasta_full_feature(fasta: Path, sample_name: str | None = None, threads: int = 2) -> dict[str, Any]:
    ready, missing = full_feature_ready()
    if not ready:
        raise FullFeatureUnavailable("Full-feature pipeline assets/tools missing: " + ", ".join(missing))
    sample = sample_name or fasta.stem
    with tempfile.TemporaryDirectory(prefix="carbapenem_full_feature_") as tmp:
        work_dir = Path(tmp)
        rgi_path, amr_detector = run_amr_detection(fasta, work_dir, threads=threads)
        snippy_path = run_snippy(fasta, work_dir, threads=threads)
        rgi = load_rgi_table(rgi_path)
        snippy = load_snippy_table(snippy_path)
        features, notes = build_feature_row(rgi, snippy, sample)
        predictions = predict_from_features(features)
        summary = mechanism_summary_from_features(features)
        return {
            "mode": "full_feature",
            "sample_name": sample,
            "predictions": predictions,
            "mechanism_summary": summary,
            "feature_notes": notes,
            "amr_detector": amr_detector,
            "rgi_hit_count": int(len(rgi)),
            "snippy_target_variant_count": int(len(snippy)),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full-feature IPM/MEM FASTA analysis.")
    parser.add_argument("fasta")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("APP_THREADS", "2")))
    args = parser.parse_args()
    result = analyze_fasta_full_feature(Path(args.fasta), threads=args.threads)
    print(json.dumps(result, indent=2))
