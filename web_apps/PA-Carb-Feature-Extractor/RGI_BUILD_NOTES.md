# RGI Build Notes

This folder is an experimental RGI-enabled version of the Feature Extractor.

It differs from the stable DIAMOND build by adding:

- `git`
- a separate `rgi` micromamba environment installed from `bioconda`
- a small `/usr/local/bin/rgi` wrapper that runs `micromamba run -n rgi rgi`
- `rgi load --card_json /app/assets/card/card.json --local`

RGI is intentionally isolated from the web-app Python environment so that
RGI dependencies do not conflict with the locked scikit-learn/joblib model
runtime.

The app still keeps a fallback chain:

1. RGI/CARD
2. DIAMOND/CARD protein fallback
3. BLASTN/CARD nucleotide fallback

If HuggingFace build fails because the RGI dependency chain is too heavy, keep using the stable `huggingface_feature_extractor_upload` folder.

After deployment, open `/health` and confirm:

```json
"rgi": true
```

Then run one FASTA and check that the result reports:

```text
AMR detector: RGI/CARD
```
