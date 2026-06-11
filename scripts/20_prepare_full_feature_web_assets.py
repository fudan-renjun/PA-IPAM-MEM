#!/usr/bin/env python3
"""Prepare lightweight assets for the full-feature FASTA web deployment.

This script does not modify the locked model. It exports metadata required to
reconstruct the model input matrix in a Docker Space:

- The exact locked model feature columns.
- Public-training ARO-specific median `__norm` values.

Heavy assets such as CARD, PAO1, and cached ESM2 embeddings are intentionally
not copied here; they are listed in the deployment upgrade plan and should be
bundled only when building the full Docker package.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEPLOY_ASSETS = PROJECT / "投稿" / "huggingface_docker_upload" / "assets"
MODEL_PATH = DEPLOY_ASSETS / "ipm_mem_unified_public_only_model_2026-06-05.joblib"
PUBLIC_TRAIN = WORKSPACE / "data" / "processed" / "features" / "feature_matrix_pamic_public_training_2026-05-19.csv"


def feature_columns(artifact: dict[str, object]) -> list[str]:
    cols: set[str] = set()
    for endpoint in artifact["endpoints"].values():
        cols.update(endpoint["gate_feature_cols"])
        cols.update(endpoint["stage2_feature_cols"])
    return sorted(cols)


def write_feature_metadata(artifact: dict[str, object], cols: list[str]) -> None:
    payload = {
        "artifact_name": artifact.get("artifact_name"),
        "created_date": artifact.get("created_date"),
        "feature_count": len(cols),
        "feature_columns": cols,
        "endpoints": {
            drug: {
                "gate_feature_count": len(endpoint["gate_feature_cols"]),
                "stage2_feature_count": len(endpoint["stage2_feature_cols"]),
                "policy": endpoint["policy"],
                "n_training_rows": endpoint.get("n_training_rows"),
                "n_s": endpoint.get("n_s"),
                "n_ns": endpoint.get("n_ns"),
            }
            for drug, endpoint in artifact["endpoints"].items()
        },
    }
    out = DEPLOY_ASSETS / "locked_model_feature_metadata.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)


def write_aro_norm_reference(cols: list[str]) -> None:
    train = pd.read_csv(PUBLIC_TRAIN, low_memory=False)
    norm_cols = [col for col in cols if col.endswith("__norm") and col in train.columns]
    rows = []
    for norm_col in norm_cols:
        aro = norm_col.removesuffix("__norm")
        present_col = f"{aro}__present"
        values = pd.to_numeric(train[norm_col], errors="coerce")
        if present_col in train.columns:
            present = pd.to_numeric(train[present_col], errors="coerce").fillna(0).gt(0)
            present_values = values[present]
        else:
            present_values = values[values.gt(0)]
        rows.append(
            {
                "aro": aro,
                "norm_feature": norm_col,
                "present_feature": present_col if present_col in train.columns else "",
                "public_present_n": int(present_values.notna().sum()),
                "median_norm_when_present": float(present_values.median()) if present_values.notna().any() else 0.0,
                "mean_norm_when_present": float(present_values.mean()) if present_values.notna().any() else 0.0,
            }
        )
    out = DEPLOY_ASSETS / "aro_norm_reference_public_training.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, "rows", len(rows))


def main() -> None:
    DEPLOY_ASSETS.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(MODEL_PATH)
    cols = feature_columns(artifact)
    write_feature_metadata(artifact, cols)
    write_aro_norm_reference(cols)


if __name__ == "__main__":
    main()

