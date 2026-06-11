# HuggingFace Upload Checklist: Feature Extractor

Upload the contents of this folder as one HuggingFace Docker Space.

Required root files:

- `README.md`
- `Dockerfile`
- `requirements.txt`
- `app.py`
- `full_feature_pipeline.py`

Required assets:

- `assets/locked_model_feature_metadata.json`
- `assets/aro_norm_reference_public_training.csv`
- `assets/protein_index.csv`
- `assets/protein_embeddings.npy`
- `assets/card/`
- `assets/pao1/`

After deployment:

1. Open `/health`.
2. Confirm `feature_extraction_ready` is `true`.
3. Check `amr_detector`. Accepted values are `RGI/CARD`, `DIAMOND/CARD fallback`, or `BLASTN/CARD nucleotide fallback`.
4. Upload one FASTA, multiple FASTA files, or a ZIP archive containing FASTA files.
5. Keep each run to no more than 10 assemblies.
6. Download `feature_matrix.csv`.
7. Upload that CSV to the separate MIC Analyzer Space.

Do not use feature matrices if the manifest reports `manuscript_equivalent_feature_schema: false`.
