# HuggingFace Upload Checklist: PA-Carb MIC Interpreter

Upload the contents of this folder as a separate HuggingFace Docker Space.

Required root files:

- `README.md`
- `Dockerfile`
- `requirements.txt`
- `app.py`

Required assets:

- `assets/ipm_mem_unified_public_only_model_2026-06-05.joblib`
- `assets/locked_model_feature_metadata.json`

After deployment:

1. Open `/health`.
2. Confirm `model_loaded` is `true`.
3. Upload the single-sample or batch `feature_matrix.csv` exported by the PA-Carb Feature Extractor Space.
4. Confirm prediction is not blocked by missing features.

Do not upload raw FASTA files to this interpreter. FASTA files must first be processed by the PA-Carb Feature Extractor Space.
