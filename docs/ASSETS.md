# External Assets

This GitHub upload folder is a code-oriented package. It intentionally excludes large deployment assets from the PA-Carb Feature Extractor because the asset bundle is approximately 200 MB and includes CARD database files, PAO1 reference files, and protein embedding resources.

## Included Small Assets

The Feature Extractor folder includes small configuration files:

- `locked_model_feature_metadata.json`
- `reference_targets.json`
- `protein_index.csv`
- `aro_norm_reference_public_training.csv`

The MIC Interpreter folder includes the small locked model package used by the hosted interpreter:

- `ipm_mem_unified_public_only_model_2026-06-05.joblib`
- `locked_model_feature_metadata.json`

## Required Large Assets For Full Feature Extractor Deployment

To run the Feature Extractor exactly as deployed, the following assets must be supplied under:

`web_apps/PA-Carb-Feature-Extractor/assets/`

Required large files/directories:

- `protein_embeddings.npy`
- `card/`
- `pao1/`

These are available in the deployed HuggingFace Space asset bundle or should be distributed through Git LFS, Zenodo, Figshare, institutional storage, or GitHub Releases.

## Recommended GitHub Strategy

For a public manuscript repository:

- Keep this repository focused on scripts and web-app source code.
- Put large assets in a release or data archive.
- Add a DOI link to the asset archive in the final manuscript and repository README.
