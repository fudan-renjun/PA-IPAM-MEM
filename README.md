# PA-Carb IPM/MEM Predictability Study

This repository contains the analysis scripts and research-use web application code for a mechanism-resolved IPM/MEM MIC predictability study in *Pseudomonas aeruginosa*.

The study evaluates a locked public-training-only IPM/MEM model, joins prediction errors to genome-defined carbapenem resistance mechanism strata, and implements a two-stage web workflow:

1. `PA-Carb-Feature-Extractor`: FASTA or ZIP upload to manuscript-schema feature matrix.
2. `PA-Carb-MIC-Interpreter`: feature matrix upload to locked IPM/MEM MIC prediction and reliability labels.

## Repository Layout

```text
scripts/
  00-20 analysis, figure-data, table, manuscript, and web-asset preparation scripts

web_apps/
  PA-Carb-Feature-Extractor/
    FastAPI/Docker code for FASTA-to-feature extraction
  PA-Carb-MIC-Interpreter/
    FastAPI/Docker code for feature-matrix IPM/MEM interpretation

docs/
  Notes on required external assets and reproduction workflow
```

## Web Tools

Public research-use deployments:

- PA-Carb Feature Extractor: https://huggingface.co/spaces/fudan-renjun/PA-Carb-Feature-Extractor
- PA-Carb MIC Interpreter: https://huggingface.co/spaces/fudan-renjun/PA-Carb-MIC-Interpreter

The web tools are research-use only and are not validated clinical AST systems.

## Data And Large Assets

Raw clinical metadata, local assemblies, manuscript Word files, generated figure outputs, and large CARD/PAO1/embedding assets are not included in this code package.

See `docs/ASSETS.md` for the large assets required to run the full Feature Extractor deployment.

## Analysis Order

The numbered scripts in `scripts/` reflect the analysis order used for the manuscript:

1. Metadata cleaning and annotation audit.
2. Locked public-only IPM/MEM model evaluation.
3. Mechanism and high-confidence subtype annotation.
4. Predictability statistics, paired IPM/MEM analysis, ST sensitivity, and clinical warning rules.
5. Figure/table data export and manuscript/supplement generation.

Most scripts assume the original project data layout and are provided for transparency and reproducibility of the analysis workflow.

## Minimal Python Dependencies

Install the common Python dependencies with:

```bash
pip install -r requirements.txt
```

The Feature Extractor web app additionally requires external command-line tools inside Docker, including RGI/CARD or fallback AMR tools, Snippy, BLAST+, DIAMOND, and Prodigal.

## Citation

Please cite the associated manuscript when using this code or the web tools.
