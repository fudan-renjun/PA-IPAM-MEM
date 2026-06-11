# Web Deployment Notes

The project has two separate HuggingFace Docker Spaces.

## 1. PA-Carb Feature Extractor

Folder:

`web_apps/PA-Carb-Feature-Extractor`

Purpose:

- Upload one FASTA, multiple FASTA files, or a ZIP archive.
- Maximum batch size: 10 assemblies per run.
- Run assembly QC, RGI/CARD AMR detection with fallback paths, and PAO1/Snippy variant calling.
- Export `feature_matrix.csv` and `feature_manifest.json`.

Public deployment:

https://huggingface.co/spaces/fudan-renjun/PA-Carb-Feature-Extractor

## 2. PA-Carb MIC Interpreter

Folder:

`web_apps/PA-Carb-MIC-Interpreter`

Purpose:

- Upload the single-sample or batch `feature_matrix.csv` from the Feature Extractor.
- Validate the 455-feature manuscript schema.
- Apply the locked public-only IPM/MEM model.
- Return predicted MICs, S/NS calls, P(NS), driver axes, mechanism labels, and reliability flags.

Public deployment:

https://huggingface.co/spaces/fudan-renjun/PA-Carb-MIC-Interpreter

## Clinical Use Boundary

Both tools are research-use implementations and are not validated clinical antimicrobial susceptibility testing systems.
