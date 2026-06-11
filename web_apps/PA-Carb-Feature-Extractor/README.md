---
title: "Carbapenem Feature Extractor RGI"
emoji: "🧬"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Carbapenem Feature Extractor RGI

Step 1 of the IPM/MEM predictability workflow.

This Space accepts one or more *Pseudomonas aeruginosa* assembly FASTA files and exports a manuscript-schema feature matrix for downstream MIC analysis. Upload can be performed by file picker or drag-and-drop. Batch mode accepts FASTA files directly or a ZIP archive containing FASTA files, with a maximum of 10 assemblies per run.

This experimental build attempts to install RGI/CARD in the Docker image. AMR gene features are extracted with the first available detector in this order: RGI/CARD, DIAMOND/CARD protein fallback, then BLASTN/CARD nucleotide fallback.

## Outputs

- `feature_matrix.csv`
- `feature_manifest.json`

Intermediate AMR and Snippy files are used internally to build the feature matrix and are not exposed as required downloads.

## Required Check

Open `/health` after deployment. Use the extractor for manuscript-equivalent prediction only when:

```json
"feature_extraction_ready": true
```

The exported `feature_matrix.csv` can contain one row or multiple genome rows and should then be uploaded to the companion MIC Analyzer Space.

This tool is research-use only and is not a validated clinical AST system.
