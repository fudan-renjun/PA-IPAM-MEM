from __future__ import annotations

import html
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
MODEL_PATH = ASSET_DIR / "ipm_mem_unified_public_only_model_2026-06-05.joblib"
FEATURE_META_PATH = ASSET_DIR / "locked_model_feature_metadata.json"
APP_TITLE = "PA-Carb MIC Interpreter"
DRUGS = ["IPM", "MEM"]

app = FastAPI(title=APP_TITLE)

MODEL_LOAD_ERROR = ""
try:
    MODEL_ARTIFACT = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL_ARTIFACT = None
    MODEL_LOAD_ERROR = str(exc)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def feature_metadata() -> dict[str, object]:
    return json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))


def required_feature_columns() -> list[str]:
    return list(feature_metadata()["feature_columns"])


def nearest_train_level(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    levels = np.asarray(sorted(np.unique(levels)), dtype=float)
    if levels.size == 0:
        return pred
    idx = np.abs(pred.reshape(-1, 1) - levels.reshape(1, -1)).argmin(axis=1)
    return levels[idx]


def aligned_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)


def endpoint_base_predictions(df: pd.DataFrame, endpoint: dict[str, object]) -> pd.DataFrame:
    x_gate = aligned_frame(df, endpoint["gate_feature_cols"])  # type: ignore[index]
    x_stage2 = aligned_frame(df, endpoint["stage2_feature_cols"])  # type: ignore[index]
    prob_ns = endpoint["gate_model"].predict_proba(x_gate)[:, 1]  # type: ignore[index,union-attr]
    global_reg = endpoint["global_regressor"]  # type: ignore[index]
    s_reg = endpoint["s_regressor"] if endpoint["s_regressor"] is not None else global_reg  # type: ignore[index]
    ns_reg = endpoint["ns_regressor"] if endpoint["ns_regressor"] is not None else global_reg  # type: ignore[index]
    pred_ungated = global_reg.predict(x_stage2)  # type: ignore[union-attr]
    pred_s = s_reg.predict(x_stage2)  # type: ignore[union-attr]
    pred_ns = ns_reg.predict(x_stage2)  # type: ignore[union-attr]
    pred_soft = (1.0 - prob_ns) * pred_s + prob_ns * pred_ns
    return pd.DataFrame(
        {
            "prob_ns": prob_ns,
            "pred_ungated": pred_ungated,
            "pred_s_branch": pred_s,
            "pred_ns_branch": pred_ns,
            "pred_soft_gate": pred_soft,
        },
        index=df.index,
    )


def apply_model_policy(base: pd.DataFrame, endpoint: dict[str, object]) -> np.ndarray:
    policy = endpoint["policy"]  # type: ignore[index]
    breakpoint = float(policy["breakpoint_log2"])  # type: ignore[index]
    threshold = float(policy["threshold"])  # type: ignore[index]
    approach = str(policy["approach"])  # type: ignore[index]
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
    if bool(policy["snapped_to_train_levels"]):  # type: ignore[index]
        pred = nearest_train_level(pred, np.asarray(endpoint["train_mic_levels"], dtype=float))  # type: ignore[index]
    return pred


def format_mic(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value >= 1 and abs(value - round(value)) < 0.02:
        return str(int(round(value)))
    return f"{value:.3g}"


def mechanism_summary(row: pd.Series) -> tuple[str, str]:
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
    return subtype, "; ".join(axes) if axes else "none detected"


def reliability_label(drug: str, pred_sns: str, subtype: str) -> tuple[str, str]:
    subtype_lower = subtype.lower()
    oprd_or_composite = "oprd-loss" in subtype_lower or "composite" in subtype_lower
    if drug == "IPM" and pred_sns == "NS" and oprd_or_composite:
        return "Stratum-supported", "Mechanism-concordant / interpretable"
    if drug == "MEM" and pred_sns == "S" and oprd_or_composite:
        return "High-risk false-susceptible pattern", "Caution: large-underestimation and false-susceptible risk"
    if drug == "MEM":
        return "Caution", "Endpoint-specific caution"
    return "Caution", "Mechanism-stratum dependent"


def validate_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = required_feature_columns()
    missing = [col for col in required if col not in df.columns]
    extra = [col for col in df.columns if col not in required and col != "genome_id"]
    report = {
        "expected_feature_count": len(required),
        "missing_feature_count": len(missing),
        "extra_column_count": len(extra),
        "missing_features": missing[:50],
        "manuscript_equivalent_feature_schema": len(missing) == 0,
    }
    if missing:
        return df, report
    if "genome_id" not in df.columns:
        df = df.copy()
        df.insert(0, "genome_id", [f"sample_{i+1}" for i in range(len(df))])
    return df[["genome_id"] + required], report


def predict(df: pd.DataFrame) -> pd.DataFrame:
    if MODEL_ARTIFACT is None:
        raise RuntimeError(f"Locked model could not be loaded: {MODEL_LOAD_ERROR}")
    rows = []
    for idx, row in df.iterrows():
        subtype, axes = mechanism_summary(row)
        for drug in DRUGS:
            endpoint = MODEL_ARTIFACT["endpoints"][drug]
            base = endpoint_base_predictions(df.loc[[idx]], endpoint)
            pred_log2 = float(apply_model_policy(base, endpoint)[0])
            policy = endpoint["policy"]
            breakpoint_log2 = float(policy["breakpoint_log2"])
            sns = "NS" if pred_log2 > breakpoint_log2 else "S"
            flag, reliability = reliability_label(drug, sns, subtype)
            rows.append(
                {
                    "genome_id": row["genome_id"],
                    "drug": drug,
                    "predicted_mic_mg_l": format_mic(float(2**pred_log2)),
                    "predicted_log2_mic": round(pred_log2, 4),
                    "breakpoint_mg_l": format_mic(float(2**breakpoint_log2)),
                    "predicted_sns": sns,
                    "probability_ns": round(float(base["prob_ns"].iloc[0]), 4),
                    "mechanism_subtype": subtype,
                    "driver_axes": axes,
                    "reliability_flag": flag,
                    "stratum_specific_reliability": reliability,
                    "prediction_type": "Locked public-only full-feature model",
                }
            )
    return pd.DataFrame(rows)


def html_layout(body: str = "", error: str = "") -> str:
    error_html = f"<div class='alert'>{esc(error)}</div>" if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_TITLE}</title>
  <style>
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:#14212b; background:#fff; }}
    header {{ padding:28px clamp(20px,5vw,56px); border-bottom:1px solid #d7e0e8; background:#f4f9fc; }}
    main {{ max-width:1160px; margin:0 auto; padding:28px clamp(20px,5vw,56px) 52px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin-top:28px; font-size:20px; }}
    p {{ line-height:1.55; }}
    .muted {{ color:#5c6b76; }}
    .panel {{ border:1px solid #d7e0e8; border-radius:8px; padding:18px; background:#f8fafc; margin:18px 0; }}
    .alert {{ border-left:4px solid #a34c00; background:#fff8ef; padding:12px 14px; margin:18px 0; }}
    .upload {{ display:grid; grid-template-columns:1fr auto; gap:14px; align-items:center; }}
    .file-input {{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }}
    .drop-zone {{ border:2px dashed #b9c8d7; border-radius:8px; padding:18px; background:#fff; cursor:pointer; transition:border-color .15s, background-color .15s; }}
    .drop-zone:hover,.drop-zone.dragging {{ border-color:#0b6fa4; background:#eef7fc; }}
    .drop-title {{ font-weight:700; margin-bottom:4px; }}
    .file-name {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#5c6b76; }}
    button,.download {{ border:0; border-radius:6px; padding:11px 16px; color:#fff; background:#0b6fa4; font-weight:700; text-decoration:none; cursor:pointer; }}
    .download {{ background:#16815a; display:inline-block; margin:4px 8px 4px 0; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; margin:16px 0; }}
    .card {{ border:1px solid #d7e0e8; border-radius:8px; padding:14px; background:#fff; }}
    .card b {{ display:block; margin-bottom:6px; }}
    .card .value {{ font-size:24px; font-weight:700; margin:4px 0; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; }}
    th,td {{ border-bottom:1px solid #d7e0e8; padding:8px; text-align:left; vertical-align:top; }}
    th {{ background:#eef3f7; }}
    .table-wrap {{ overflow-x:auto; border:1px solid #d7e0e8; border-radius:8px; }}
    .flag {{ border-radius:999px; padding:4px 8px; font-weight:700; font-size:12px; display:inline-block; }}
    .flag-high {{ background:#fdecec; color:#9b1c1c; }}
    .flag-ok {{ background:#e8f5ef; color:#116849; }}
    .flag-caution {{ background:#fff4df; color:#8a5200; }}
    @media(max-width:700px) {{ .upload {{ grid-template-columns:1fr; }} .file-name {{ white-space:normal; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{APP_TITLE}</h1>
    <p class="muted">Step 2 of the PA-Carb workflow: upload the single-sample or batch feature_matrix.csv generated by PA-Carb Feature Extractor. The interpreter blocks prediction if required model features are missing.</p>
  </header>
  <main>
    {error_html}
    <form class="panel upload" action="/analyze" method="post" enctype="multipart/form-data">
      <input class="file-input" id="feature-file" type="file" name="file" accept=".csv,.txt" required>
      <label class="drop-zone" id="drop-zone" for="feature-file">
        <div class="drop-title">Choose or drag feature_matrix.csv here</div>
        <div class="file-name" id="file-name">No file selected</div>
        <div class="muted" style="font-size:13px;margin-top:6px">Single-sample and batch matrices are both supported; one row equals one genome.</div>
      </label>
      <button type="submit">Analyze Features</button>
    </form>
    {body}
  </main>
  <script>
    const featureInput = document.getElementById('feature-file');
    const fileName = document.getElementById('file-name');
    const form = document.querySelector('form.upload');
    const dropZone = document.getElementById('drop-zone');
    const setSelectedFileName = () => {{
      fileName.textContent = featureInput.files.length ? featureInput.files[0].name : 'No file selected';
      fileName.style.color = featureInput.files.length ? '#14212b' : '#5c6b76';
    }};
    if (featureInput && fileName) {{
      featureInput.addEventListener('change', setSelectedFileName);
    }}
    if (dropZone && featureInput) {{
      ['dragenter','dragover'].forEach(eventName => {{
        dropZone.addEventListener(eventName, event => {{
          event.preventDefault();
          dropZone.classList.add('dragging');
        }});
      }});
      ['dragleave','drop'].forEach(eventName => {{
        dropZone.addEventListener(eventName, event => {{
          event.preventDefault();
          dropZone.classList.remove('dragging');
        }});
      }});
      dropZone.addEventListener('drop', event => {{
        if (event.dataTransfer && event.dataTransfer.files.length) {{
          featureInput.files = event.dataTransfer.files;
          setSelectedFileName();
        }}
      }});
    }}
    if (form) {{
      form.addEventListener('submit', () => {{
        const button = form.querySelector('button[type="submit"]');
        if (button) {{
          button.disabled = true;
          button.textContent = 'Analyzing...';
        }}
      }});
    }}
  </script>
</body>
</html>"""


def flag_class(flag: object) -> str:
    text = str(flag).lower()
    if "high-risk" in text:
        return "flag-high"
    if "supported" in text:
        return "flag-ok"
    return "flag-caution"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return html_layout()


@app.get("/health")
def health() -> JSONResponse:
    meta = feature_metadata() if FEATURE_META_PATH.exists() else {}
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": MODEL_ARTIFACT is not None,
            "model_load_error": MODEL_LOAD_ERROR,
            "feature_count_expected": meta.get("feature_count"),
            "analyzer_mode": "feature_matrix_only",
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)) -> str:
    raw = await file.read()
    if not raw:
        return html_layout(error="The uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        return html_layout(error=f"Could not read CSV: {exc}")
    checked, report = validate_feature_matrix(df)
    if not report["manuscript_equivalent_feature_schema"]:
        missing_preview = ", ".join(report["missing_features"]) or "none"
        body = f"""
        <section>
          <h2>Prediction Blocked</h2>
          <div class="alert">
            <p><b>Reason:</b> the uploaded matrix is missing {esc(report['missing_feature_count'])} required model features.</p>
            <p><b>Missing feature preview:</b> {esc(missing_preview)}</p>
            <p>Generate the file with the companion Carbapenem Feature Extractor Space and upload the exported feature_matrix.csv.</p>
          </div>
        </section>
        """
        return html_layout(body)
    try:
        pred = predict(checked)
    except Exception as exc:
        return html_layout(error=str(exc))
    csv_blob = pred.to_csv(index=False)
    csv_href = "data:text/csv;charset=utf-8," + quote(csv_blob)
    rows = "".join(
        "<tr>"
        f"<td>{esc(r.genome_id)}</td>"
        f"<td>{esc(r.drug)}</td>"
        f"<td>{esc(r.predicted_mic_mg_l)}</td>"
        f"<td>{esc(r.predicted_log2_mic)}</td>"
        f"<td>{esc(r.breakpoint_mg_l)}</td>"
        f"<td>{esc(r.predicted_sns)}</td>"
        f"<td>{esc(r.probability_ns)}</td>"
        f"<td>{esc(r.mechanism_subtype)}</td>"
        f"<td>{esc(r.driver_axes)}</td>"
        f"<td><span class='flag {flag_class(r.reliability_flag)}'>{esc(r.reliability_flag)}</span></td>"
        f"<td>{esc(r.stratum_specific_reliability)}</td>"
        "</tr>"
        for r in pred.itertuples(index=False)
    )
    cards = "".join(
        f"""
        <div class="card">
          <b>{esc(r.drug)} predicted MIC</b>
          <div class="value">{esc(r.predicted_mic_mg_l)} mg/L</div>
          <div>log2 MIC {esc(r.predicted_log2_mic)}; {esc(r.predicted_sns)}; P(NS) {esc(r.probability_ns)}</div>
          <div style="margin-top:8px"><span class="flag {flag_class(r.reliability_flag)}">{esc(r.reliability_flag)}</span></div>
          <p class="muted">{esc(r.stratum_specific_reliability)}</p>
        </div>
        """
        for r in pred.itertuples(index=False)
    )
    body = f"""
    <section>
      <h2>MIC Prediction Result</h2>
      <div class="panel">
        <p><b>Rows analysed:</b> {esc(len(checked))}; <b>required features:</b> {esc(report['expected_feature_count'])}; <b>missing features:</b> 0.</p>
        <p>The numerical values were generated from a manuscript-schema feature matrix and the locked public-only IPM/MEM model. This remains a research-use prediction and is not a validated clinical AST result.</p>
        <a class="download" download="carbapenem_mic_predictions.csv" href="{csv_href}">Download predictions CSV</a>
      </div>
      <div class="cards">{cards}</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>genome_id</th><th>drug</th><th>predicted MIC</th><th>log2 MIC</th><th>breakpoint</th><th>S/NS</th><th>P(NS)</th><th>mechanism subtype</th><th>driver axes</th><th>reliability flag</th><th>stratum reliability</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>
    """
    return html_layout(body)
