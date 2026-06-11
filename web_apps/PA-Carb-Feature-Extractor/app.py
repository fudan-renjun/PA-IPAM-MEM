from __future__ import annotations

import html
import json
import os
import tempfile
import threading
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

import full_feature_pipeline as pipeline


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

APP_TITLE = "Carbapenem Feature Extractor"
MAX_BATCH_FASTA = 10
FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".fas", ".txt"}
ZIP_SUFFIXES = {".zip"}
app = FastAPI(title=APP_TITLE)
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


STEPS = [
    "Queued",
    "Reading FASTA and computing assembly QC",
    "Checking feature-extraction tools and bundled assets",
    "Running CARD AMR gene detection",
    "Running PAO1/Snippy variant calling",
    "Building manuscript-schema feature matrix",
    "Preparing downloadable files",
    "Complete",
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def update_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(fields)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def step_progress(step_index: int) -> int:
    if step_index <= 0:
        return 3
    return min(99, int(step_index / (len(STEPS) - 1) * 100))


def is_fasta_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in FASTA_SUFFIXES


def is_zip_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ZIP_SUFFIXES


def clean_sample_name(filename: str) -> str:
    stem = Path(filename).name
    for suffix in [".fasta", ".fna", ".fa", ".fas", ".txt"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._-")
    return safe or "sample"


def uniquify_sample_names(items: list[dict[str, object]]) -> None:
    seen: dict[str, int] = {}
    for item in items:
        base = str(item["sample_name"])
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            item["sample_name"] = f"{base}_{count + 1}"


def extract_fasta_members_from_zip(raw: bytes, archive_name: str) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name or name.startswith(".") or not is_fasta_filename(name):
                    continue
                with zf.open(info) as handle:
                    members.append((name, handle.read()))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{archive_name} is not a valid ZIP archive.") from exc
    return members


def prepare_uploaded_fastas(uploaded: list[tuple[str, bytes]], work_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for filename, raw in uploaded:
        if is_zip_filename(filename):
            for member_name, member_raw in extract_fasta_members_from_zip(raw, filename):
                items.append({"filename": member_name, "raw": member_raw})
        elif is_fasta_filename(filename):
            items.append({"filename": Path(filename).name, "raw": raw})
        else:
            raise ValueError(f"Unsupported file type: {filename}. Upload FASTA files or a ZIP archive containing FASTA files.")
    if not items:
        raise ValueError("No FASTA files were found. Upload .fa, .fna, .fasta, .fas, or a ZIP archive containing these files.")
    if len(items) > MAX_BATCH_FASTA:
        raise ValueError(f"Too many FASTA files were provided ({len(items)}). Please upload no more than {MAX_BATCH_FASTA} assemblies per run.")
    prepared: list[dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        raw = bytes(item["raw"])
        if not raw:
            raise ValueError(f"{item['filename']} is empty.")
        filename = str(item["filename"])
        sample_name = clean_sample_name(filename)
        path = work_dir / f"{index:02d}_{sample_name}.fasta"
        path.write_bytes(raw)
        prepared.append(
            {
                "filename": filename,
                "sample_name": sample_name,
                "path": path,
                "raw_text": raw.decode("utf-8", errors="ignore"),
            }
        )
    uniquify_sample_names(prepared)
    return prepared


def read_fasta_qc(text: str) -> dict[str, object]:
    lengths: list[int] = []
    current = 0
    gc = 0
    n_count = 0
    total = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                lengths.append(current)
            current = 0
            continue
        seq = "".join(ch for ch in line.upper() if not ch.isspace())
        current += len(seq)
        total += len(seq)
        gc += seq.count("G") + seq.count("C")
        n_count += seq.count("N")
    if current:
        lengths.append(current)
    n50 = 0
    if lengths:
        half = sum(lengths) / 2
        running = 0
        for length in sorted(lengths, reverse=True):
            running += length
            if running >= half:
                n50 = length
                break
    return {
        "contigs": len(lengths),
        "total_bp": total,
        "largest_contig": max(lengths) if lengths else 0,
        "n50": n50,
        "gc_percent": round(gc / total * 100, 2) if total else 0,
        "n_percent": round(n_count / total * 100, 3) if total else 0,
    }


def feature_extraction_ready() -> tuple[bool, list[str], str]:
    ready, missing = pipeline.full_feature_ready()
    if pipeline.tool_available("rgi"):
        detector = "RGI/CARD"
    elif pipeline.diamond_tool() is not None and pipeline.tool_available("prodigal"):
        detector = "DIAMOND/CARD fallback"
    elif pipeline.tool_available("blastn") and pipeline.tool_available("makeblastdb"):
        detector = "BLASTN/CARD nucleotide fallback"
    else:
        detector = "unavailable"
    return ready, missing, detector


def extract_feature_bundle(fasta: Path, sample_name: str, threads: int = 2) -> dict[str, object]:
    ready, missing, detector = feature_extraction_ready()
    if not ready:
        raise pipeline.FullFeatureUnavailable("Feature extraction tools/assets missing: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="carbapenem_feature_extract_") as tmp:
        work_dir = Path(tmp)
        amr_path, amr_detector = pipeline.run_amr_detection(fasta, work_dir, threads=threads)
        snippy_path = pipeline.run_snippy(fasta, work_dir, threads=threads)
        amr = pipeline.load_rgi_table(amr_path)
        snippy = pipeline.load_snippy_table(snippy_path)
        features, notes = pipeline.build_feature_row(amr, snippy, sample_name)
        feature_cols = pipeline.read_feature_columns()
        output_cols = ["genome_id"] + feature_cols
        for col in output_cols:
            if col not in features.columns:
                features[col] = 0.0
        features = features[output_cols]
        missing_features = [col for col in feature_cols if col not in features.columns]
        manifest = {
            "schema": "carbapenem_feature_matrix",
            "schema_version": "2026-06-11",
            "sample_name": sample_name,
            "feature_count_expected": len(feature_cols),
            "feature_count_exported": len(feature_cols) - len(missing_features),
            "missing_features": missing_features,
            "manuscript_equivalent_feature_schema": len(missing_features) == 0,
            "amr_detector": amr_detector,
            "amr_hit_count": int(len(amr)),
            "snippy_target_variant_count": int(len(snippy)),
            "feature_notes": notes,
        }
        return {
            "features": features,
            "amr": amr,
            "snippy": snippy,
            "manifest": manifest,
            "amr_detector": detector,
        }


def run_extraction_job(job_id: str, fasta_path: Path, filename: str, raw_text: str, threads: int = 2) -> None:
    sample_name = Path(filename).stem
    try:
        update_job(job_id, status="running", step=STEPS[1], step_index=1, progress=step_progress(1), log=["FASTA upload received."])
        qc = read_fasta_qc(raw_text)

        update_job(job_id, step=STEPS[2], step_index=2, progress=step_progress(2))
        ready, missing, detector = feature_extraction_ready()
        if not ready:
            raise pipeline.FullFeatureUnavailable("Feature extraction tools/assets missing: " + ", ".join(missing))

        with tempfile.TemporaryDirectory(prefix="carbapenem_feature_extract_") as tmp:
            work_dir = Path(tmp)

            update_job(job_id, step=STEPS[3], step_index=3, progress=step_progress(3), amr_detector=detector)
            amr_path, amr_detector = pipeline.run_amr_detection(fasta_path, work_dir, threads=threads)
            amr = pipeline.load_rgi_table(amr_path)

            update_job(
                job_id,
                step=STEPS[4],
                step_index=4,
                progress=step_progress(4),
                amr_detector=amr_detector,
                amr_hit_count=int(len(amr)),
            )
            snippy_path = pipeline.run_snippy(fasta_path, work_dir, threads=threads)
            snippy = pipeline.load_snippy_table(snippy_path)

            update_job(job_id, step=STEPS[5], step_index=5, progress=step_progress(5), snippy_target_variant_count=int(len(snippy)))
            features, notes = pipeline.build_feature_row(amr, snippy, sample_name)
            feature_cols = pipeline.read_feature_columns()
            output_cols = ["genome_id"] + feature_cols
            for col in output_cols:
                if col not in features.columns:
                    features[col] = 0.0
            features = features[output_cols]
            missing_features = [col for col in feature_cols if col not in features.columns]
            manifest = {
                "schema": "carbapenem_feature_matrix",
                "schema_version": "2026-06-11",
                "sample_name": sample_name,
                "feature_count_expected": len(feature_cols),
                "feature_count_exported": len(feature_cols) - len(missing_features),
                "missing_features": missing_features,
                "manuscript_equivalent_feature_schema": len(missing_features) == 0,
                "amr_detector": amr_detector,
                "amr_hit_count": int(len(amr)),
                "snippy_target_variant_count": int(len(snippy)),
                "feature_notes": notes,
            }

            update_job(job_id, step=STEPS[6], step_index=6, progress=step_progress(6))
            downloads = {
                "feature_matrix.csv": features.to_csv(index=False),
                "feature_manifest.json": json.dumps(manifest, indent=2),
            }
            update_job(
                job_id,
                status="complete",
                step=STEPS[7],
                step_index=7,
                progress=100,
                filename=filename,
                qc=qc,
                manifest=manifest,
                downloads=downloads,
                completed_at=time.time(),
            )
    except Exception as exc:
        update_job(job_id, status="error", step="Error", progress=100, error=str(exc))
    finally:
        try:
            fasta_path.unlink()
        except OSError:
            pass


def run_batch_extraction_job(job_id: str, items: list[dict[str, object]], temp_dir: Path, threads: int = 2) -> None:
    try:
        total = len(items)
        update_job(
            job_id,
            status="running",
            step=STEPS[1],
            step_index=1,
            progress=step_progress(1),
            log=[f"{total} FASTA file(s) received."],
            sample_count=total,
        )
        ready, missing, detector = feature_extraction_ready()
        if not ready:
            raise pipeline.FullFeatureUnavailable("Feature extraction tools/assets missing: " + ", ".join(missing))

        feature_frames: list[pd.DataFrame] = []
        sample_manifests: list[dict[str, object]] = []
        sample_qc: list[dict[str, object]] = []
        for index, item in enumerate(items, start=1):
            filename = str(item["filename"])
            sample_name = str(item["sample_name"])
            fasta_path = Path(item["path"])
            raw_text = str(item["raw_text"])
            base_progress = int((index - 1) / total * 92) + 3
            update_job(
                job_id,
                step=f"{STEPS[1]} ({index}/{total}: {filename})",
                step_index=1,
                progress=base_progress,
                current_sample=filename,
                current_sample_index=index,
            )
            qc = read_fasta_qc(raw_text)

            update_job(job_id, step=f"{STEPS[3]} ({index}/{total}: {filename})", step_index=3, progress=min(97, base_progress + 8), amr_detector=detector)
            with tempfile.TemporaryDirectory(prefix="carbapenem_feature_extract_") as tmp:
                work_dir = Path(tmp)
                amr_path, amr_detector = pipeline.run_amr_detection(fasta_path, work_dir, threads=threads)
                amr = pipeline.load_rgi_table(amr_path)

                update_job(
                    job_id,
                    step=f"{STEPS[4]} ({index}/{total}: {filename})",
                    step_index=4,
                    progress=min(98, base_progress + 18),
                    amr_detector=amr_detector,
                    amr_hit_count=int(len(amr)),
                )
                snippy_path = pipeline.run_snippy(fasta_path, work_dir, threads=threads)
                snippy = pipeline.load_snippy_table(snippy_path)

                update_job(
                    job_id,
                    step=f"{STEPS[5]} ({index}/{total}: {filename})",
                    step_index=5,
                    progress=min(99, base_progress + 28),
                    snippy_target_variant_count=int(len(snippy)),
                )
                features, notes = pipeline.build_feature_row(amr, snippy, sample_name)
                feature_cols = pipeline.read_feature_columns()
                output_cols = ["genome_id"] + feature_cols
                for col in output_cols:
                    if col not in features.columns:
                        features[col] = 0.0
                features = features[output_cols]
                missing_features = [col for col in feature_cols if col not in features.columns]
                manifest = {
                    "sample_name": sample_name,
                    "source_filename": filename,
                    "feature_count_expected": len(feature_cols),
                    "feature_count_exported": len(feature_cols) - len(missing_features),
                    "missing_features": missing_features,
                    "manuscript_equivalent_feature_schema": len(missing_features) == 0,
                    "amr_detector": amr_detector,
                    "amr_hit_count": int(len(amr)),
                    "snippy_target_variant_count": int(len(snippy)),
                    "feature_notes": notes,
                }
                feature_frames.append(features)
                sample_manifests.append(manifest)
                sample_qc.append({"sample_name": sample_name, "source_filename": filename, **qc})

        update_job(job_id, step=STEPS[6], step_index=6, progress=99)
        combined_features = pd.concat(feature_frames, ignore_index=True)
        feature_cols = pipeline.read_feature_columns()
        batch_manifest = {
            "schema": "carbapenem_feature_matrix",
            "schema_version": "2026-06-11",
            "sample_count": total,
            "max_batch_fasta": MAX_BATCH_FASTA,
            "feature_count_expected": len(feature_cols),
            "feature_count_exported": len(feature_cols),
            "manuscript_equivalent_feature_schema": all(bool(m["manuscript_equivalent_feature_schema"]) for m in sample_manifests),
            "samples": sample_manifests,
            "assembly_qc": sample_qc,
        }
        downloads = {
            "feature_matrix.csv": combined_features.to_csv(index=False),
            "feature_manifest.json": json.dumps(batch_manifest, indent=2),
        }
        update_job(
            job_id,
            status="complete",
            step=STEPS[7],
            step_index=7,
            progress=100,
            filename=f"{total} FASTA file(s)",
            sample_count=total,
            qc=sample_qc[0] if total == 1 else {},
            qc_rows=sample_qc,
            manifest=batch_manifest,
            downloads=downloads,
            completed_at=time.time(),
        )
    except Exception as exc:
        update_job(job_id, status="error", step="Error", progress=100, error=str(exc))
    finally:
        for item in items:
            try:
                Path(item["path"]).unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def html_layout(body: str = "", error: str = "") -> str:
    error_html = f"<div class='alert'>{esc(error)}</div>" if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_TITLE}</title>
  <style>
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:#14212b; background:#fff; }}
    header {{ padding:28px clamp(20px,5vw,56px); border-bottom:1px solid #d7e0e8; background:#f4f9fc; }}
    main {{ max-width:1100px; margin:0 auto; padding:28px clamp(20px,5vw,56px) 52px; }}
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
    .hint {{ font-size:13px; color:#5c6b76; margin-top:6px; }}
    button, .download {{ border:0; border-radius:6px; padding:11px 16px; color:#fff; background:#0b6fa4; font-weight:700; text-decoration:none; cursor:pointer; }}
    .download {{ background:#16815a; display:inline-block; margin:4px 8px 4px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
    .metric {{ border:1px solid #d7e0e8; border-radius:6px; padding:12px; background:#fff; }}
    .metric b {{ display:block; font-size:18px; margin-bottom:3px; }}
    .progress-shell {{ border:1px solid #c8d5e1; border-radius:999px; height:14px; overflow:hidden; background:#eef3f7; margin:12px 0; }}
    .progress-bar {{ height:100%; width:0%; background:#0b6fa4; transition:width .35s ease; }}
    .status-line {{ display:flex; justify-content:space-between; gap:16px; align-items:center; }}
    .step-list {{ margin:12px 0 0; padding-left:20px; color:#5c6b76; line-height:1.6; }}
    .step-current {{ color:#14212b; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; }}
    th,td {{ border-bottom:1px solid #d7e0e8; padding:8px; text-align:left; }}
    th {{ background:#eef3f7; }}
    .table-wrap {{ overflow-x:auto; border:1px solid #d7e0e8; border-radius:8px; }}
    @media(max-width:700px) {{ .upload {{ grid-template-columns:1fr; }} .file-name {{ white-space:normal; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{APP_TITLE}</h1>
    <p class="muted">Step 1 of the carbapenem MIC workflow: convert Pseudomonas aeruginosa assembly FASTA files into a manuscript-schema feature matrix. This page does not perform MIC prediction.</p>
  </header>
  <main>
    {error_html}
    <form class="panel upload" action="/extract" method="post" enctype="multipart/form-data">
      <input class="file-input" id="fasta-file" type="file" name="files" accept=".fa,.fna,.fasta,.fas,.txt,.zip" multiple required>
      <label class="drop-zone" id="drop-zone" for="fasta-file">
        <div class="drop-title">Choose or drag FASTA files / ZIP archive here</div>
        <div class="file-name" id="file-name">No file selected</div>
        <div class="hint">Batch mode accepts up to {MAX_BATCH_FASTA} assemblies per run. ZIP archives may contain .fa, .fna, .fasta, .fas, or .txt files.</div>
      </label>
      <button type="submit">Extract Features</button>
    </form>
    {body}
  </main>
  <script>
    const fastaInput = document.getElementById('fasta-file');
    const fileName = document.getElementById('file-name');
    const form = document.querySelector('form.upload');
    const dropZone = document.getElementById('drop-zone');
    const maxFiles = {MAX_BATCH_FASTA};
    const updateFileName = () => {{
      if (!fastaInput.files.length) {{
        fileName.textContent = 'No file selected';
        fileName.style.color = '#5c6b76';
        return;
      }}
      const names = Array.from(fastaInput.files).map(file => file.name);
      fileName.textContent = names.length === 1 ? names[0] : names.length + ' files selected: ' + names.slice(0, 3).join(', ') + (names.length > 3 ? ', ...' : '');
      fileName.style.color = '#14212b';
    }};
    if (fastaInput && fileName) {{
      fastaInput.addEventListener('change', updateFileName);
    }}
    if (dropZone && fastaInput) {{
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
          fastaInput.files = event.dataTransfer.files;
          updateFileName();
        }}
      }});
    }}
    if (form) {{
      form.addEventListener('submit', event => {{
        if (fastaInput.files.length > maxFiles) {{
          event.preventDefault();
          alert('Please upload no more than ' + maxFiles + ' FASTA files per run. For ZIP archives, the server will also enforce this limit after unpacking.');
          return;
        }}
        const button = form.querySelector('button[type="submit"]');
        if (button) {{
          button.disabled = true;
          button.textContent = 'Uploading...';
        }}
      }});
    }}
  </script>
</body>
</html>"""


def job_progress_body(job_id: str, filename: str) -> str:
    step_items = "".join(f"<li id='step-{i}'>{esc(step)}</li>" for i, step in enumerate(STEPS))
    return f"""
    <section class="panel" id="job-panel" data-job-id="{esc(job_id)}">
      <h2>Feature Extraction Progress</h2>
      <p><b>Uploaded input:</b> {esc(filename)}</p>
      <div class="status-line">
        <span id="job-step">Queued</span>
        <span id="job-progress-text">0%</span>
      </div>
      <div class="progress-shell"><div class="progress-bar" id="progress-bar"></div></div>
      <ol class="step-list">{step_items}</ol>
      <div id="job-message" class="muted">This may take several minutes for a complete bacterial assembly.</div>
      <div id="job-result"></div>
    </section>
    <script>
      const jobId = "{esc(job_id)}";
      const stepEl = document.getElementById('job-step');
      const progressText = document.getElementById('job-progress-text');
      const progressBar = document.getElementById('progress-bar');
      const messageEl = document.getElementById('job-message');
      const resultEl = document.getElementById('job-result');
      let polling = true;

      function markStep(index) {{
        for (let i = 0; i < {len(STEPS)}; i++) {{
          const item = document.getElementById('step-' + i);
          if (!item) continue;
          item.className = i === index ? 'step-current' : '';
        }}
      }}

      async function pollStatus() {{
        if (!polling) return;
        try {{
          const res = await fetch('/status/' + jobId, {{ cache: 'no-store' }});
          const data = await res.json();
          const progress = data.progress || 0;
          stepEl.textContent = data.step || data.status || 'Running';
          progressText.textContent = progress + '%';
          progressBar.style.width = progress + '%';
          markStep(data.step_index || 0);
          if (data.amr_detector) {{
            messageEl.textContent = 'AMR detector: ' + data.amr_detector + '. AMR hits: ' + (data.amr_hit_count ?? 'pending') + '. Snippy target variants: ' + (data.snippy_target_variant_count ?? 'pending') + '.';
          }}
          if (data.status === 'complete') {{
            polling = false;
            const result = await fetch('/result/' + jobId, {{ cache: 'no-store' }});
            resultEl.innerHTML = await result.text();
          }} else if (data.status === 'error') {{
            polling = false;
            resultEl.innerHTML = '<div class="alert"><b>Extraction failed:</b> ' + (data.error || 'Unknown error') + '</div>';
          }} else {{
            setTimeout(pollStatus, 2000);
          }}
        }} catch (err) {{
          messageEl.textContent = 'Waiting for server status...';
          setTimeout(pollStatus, 3000);
        }}
      }}
      pollStatus();
    </script>
    """


def result_fragment(job_id: str, job: dict[str, object]) -> str:
    qc = dict(job.get("qc", {}))
    manifest = dict(job.get("manifest", {}))
    qc_html = "".join(f"<div class='metric'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for k, v in qc.items())
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in manifest.items() if k not in {"missing_features", "samples", "assembly_qc"})
    sample_rows = ""
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        sample_rows += (
            "<tr>"
            f"<td>{esc(sample.get('sample_name', ''))}</td>"
            f"<td>{esc(sample.get('source_filename', ''))}</td>"
            f"<td>{esc(sample.get('amr_detector', ''))}</td>"
            f"<td>{esc(sample.get('amr_hit_count', ''))}</td>"
            f"<td>{esc(sample.get('snippy_target_variant_count', ''))}</td>"
            f"<td>{esc(sample.get('manuscript_equivalent_feature_schema', ''))}</td>"
            "</tr>"
        )
    qc_rows = ""
    for row in job.get("qc_rows", []):
        if not isinstance(row, dict):
            continue
        qc_rows += (
            "<tr>"
            f"<td>{esc(row.get('sample_name', ''))}</td>"
            f"<td>{esc(row.get('contigs', ''))}</td>"
            f"<td>{esc(row.get('total_bp', ''))}</td>"
            f"<td>{esc(row.get('largest_contig', ''))}</td>"
            f"<td>{esc(row.get('n50', ''))}</td>"
            f"<td>{esc(row.get('gc_percent', ''))}</td>"
            f"<td>{esc(row.get('n_percent', ''))}</td>"
            "</tr>"
        )
    sample_table = (
        f"""
        <h2>Sample Summary</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>sample</th><th>source file</th><th>AMR detector</th><th>AMR hits</th><th>Snippy target variants</th><th>schema ready</th></tr></thead>
          <tbody>{sample_rows}</tbody>
        </table></div>
        """
        if sample_rows
        else ""
    )
    batch_qc_table = (
        f"""
        <h2>Assembly QC</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>sample</th><th>contigs</th><th>total bp</th><th>largest contig</th><th>N50</th><th>GC %</th><th>N %</th></tr></thead>
          <tbody>{qc_rows}</tbody>
        </table></div>
        """
        if qc_rows
        else ""
    )
    single_qc = f"<h2>Assembly QC</h2><div class=\"grid\">{qc_html}</div>" if qc_html and not qc_rows else ""
    return f"""
    <section>
      <h2>Extraction Result</h2>
      <div class="panel">
        <p><b>Feature schema ready:</b> {esc(manifest.get('manuscript_equivalent_feature_schema', ''))}</p>
        <p><b>Samples:</b> {esc(manifest.get('sample_count', 1))}; <b>expected features:</b> {esc(manifest.get('feature_count_expected', ''))}; <b>exported features:</b> {esc(manifest.get('feature_count_exported', ''))}</p>
        <a class="download" href="/download/{esc(job_id)}/feature_matrix.csv">Download feature_matrix.csv</a>
        <a class="download" href="/download/{esc(job_id)}/feature_manifest.json">Download feature_manifest.json</a>
      </div>
      {sample_table}
      {batch_qc_table}
      {single_qc}
      <h2>Feature Manifest</h2>
      <div class="table-wrap"><table><tbody>{rows}</tbody></table></div>
    </section>
    """


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return html_layout()


@app.get("/health")
def health() -> JSONResponse:
    ready, missing, detector = feature_extraction_ready()
    return JSONResponse(
        {
            "status": "ok",
            "feature_extraction_ready": ready,
            "missing": missing,
            "amr_detector": detector,
            "tools": {
                "snippy": pipeline.tool_available("snippy"),
                "rgi": pipeline.tool_available("rgi"),
                "diamond": pipeline.tool_available("diamond"),
                "diamond-aligner": pipeline.tool_available("diamond-aligner"),
                "diamond_selected": pipeline.diamond_tool(),
                "prodigal": pipeline.tool_available("prodigal"),
                "blastn": pipeline.tool_available("blastn"),
                "makeblastdb": pipeline.tool_available("makeblastdb"),
            },
        }
    )


@app.post("/extract", response_class=HTMLResponse)
async def extract(files: list[UploadFile] = File(...)) -> str:
    if not files:
        return html_layout(error="No files were uploaded.")
    temp_dir = Path(tempfile.mkdtemp(prefix="carbapenem_upload_batch_"))
    uploaded: list[tuple[str, bytes]] = []
    job_id = uuid.uuid4().hex
    try:
        for file in files:
            raw = await file.read()
            if not raw:
                return html_layout(error=f"{file.filename or 'uploaded file'} is empty.")
            uploaded.append((file.filename or "uploaded.fasta", raw))
        items = prepare_uploaded_fastas(uploaded, temp_dir)
    except Exception as exc:
        try:
            for path in temp_dir.iterdir():
                path.unlink()
            temp_dir.rmdir()
        except OSError:
            pass
        return html_layout(error=str(exc))
    filename = str(items[0]["filename"]) if len(items) == 1 else f"{len(items)} FASTA files"
    update_job(job_id, status="queued", step=STEPS[0], step_index=0, progress=3, filename=filename, sample_count=len(items), created_at=time.time())
    worker = threading.Thread(
        target=run_batch_extraction_job,
        args=(job_id, items, temp_dir, int(os.environ.get("APP_THREADS", "2"))),
        daemon=True,
    )
    worker.start()
    return html_layout(job_progress_body(job_id, filename))


@app.get("/status/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if job is None:
        return JSONResponse({"status": "error", "error": "Job not found.", "progress": 100})
    safe = {
        key: value
        for key, value in job.items()
        if key not in {"downloads", "qc", "manifest"}
    }
    return JSONResponse(safe)


@app.get("/result/{job_id}", response_class=HTMLResponse)
def job_result(job_id: str) -> str:
    job = get_job(job_id)
    if job is None:
        return "<div class='alert'>Job not found.</div>"
    if job.get("status") != "complete":
        return "<div class='alert'>Job is not complete.</div>"
    return result_fragment(job_id, job)


@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str) -> Response:
    allowed = {
        "feature_matrix.csv": "text/csv",
        "feature_manifest.json": "application/json",
    }
    if filename not in allowed:
        return Response("File not available.", status_code=404)
    job = get_job(job_id)
    if job is None or job.get("status") != "complete":
        return Response("Job not complete.", status_code=404)
    downloads = dict(job.get("downloads", {}))
    content = downloads.get(filename)
    if content is None:
        return Response("File not available.", status_code=404)
    return Response(
        str(content),
        media_type=allowed[filename],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/extract.zip")
async def extract_zip(file: UploadFile = File(...)) -> Response:
    raw = await file.read()
    if not raw:
        return Response("The uploaded file is empty.", status_code=400)
    filename = file.filename or "uploaded.fasta"
    suffix = Path(filename).suffix or ".fasta"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        bundle = extract_feature_bundle(tmp_path, Path(filename).stem, threads=int(os.environ.get("APP_THREADS", "2")))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("feature_matrix.csv", bundle["features"].to_csv(index=False))  # type: ignore[index,union-attr]
        zf.writestr("feature_manifest.json", json.dumps(bundle["manifest"], indent=2))  # type: ignore[index]
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{Path(filename).stem}_carbapenem_features.zip"'},
    )
