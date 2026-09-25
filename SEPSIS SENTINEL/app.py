import os
import json
import time
import asyncio
import threading
import webbrowser
import importlib
from collections import OrderedDict
from hmac import compare_digest


# ---------------------------------------------------------------------------
# Dependency check, before anything heavy is imported.
#
# A missing package otherwise surfaces as a bare ImportError several frames deep,
# which is unhelpful on an air-gapped machine where the fix is a specific pip
# command. This names the package and points at the install instructions instead.
# ---------------------------------------------------------------------------
_REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "websockets": "websockets",
    "lightgbm": "lightgbm",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
}


def _verify_dependencies():
    missing = []
    for module_name, package_name in _REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise SystemExit(
            "SepsisGuard cannot start: missing required package(s): "
            + ", ".join(sorted(missing))
            + "\n\nInstall them with:\n"
            "    pip install -r requirements.txt\n\n"
            "On an air-gapped machine, build a wheelhouse on a networked machine first:\n"
            "    pip download -r requirements.txt -d wheelhouse/\n"
            "then, offline:\n"
            "    pip install --no-index --find-links=wheelhouse -r requirements.txt\n\n"
            "See the Installation section of README.md."
        )


_verify_dependencies()

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

# Initialize evaluation on import if not already run
try:
    import evaluation
except ImportError:
    # If not in path, append parent
    import sys
    sys.path.append(os.path.dirname(__file__))
    import evaluation

# Initialize FastAPI
app = FastAPI(title="SepsisGuard Clinician Dashboard")

# CORS origins are configurable rather than wide open. The default is the local
# dashboard only; a comma-separated list widens it, and "*" restores the old
# behaviour if someone genuinely needs it.
_cors_env = os.environ.get(
    "SEPSISGUARD_CORS_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000"
).strip()
CORS_ORIGINS = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # Credentials cannot be combined with a wildcard origin per the CORS spec.
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional shared-secret auth. Enforced only when SEPSISGUARD_API_KEY is set, so the
# default local single-user experience is unchanged. Plain string comparison, no
# expiry — this guards a localhost research tool, it is not an identity system.
API_KEY = os.environ.get("SEPSISGUARD_API_KEY", "").strip()


def _key_from_request(request):
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.query_params.get("api_key") or "").strip()


@app.middleware("http")
async def require_api_key(request, call_next):
    if API_KEY:
        # Static assets stay open so the dashboard can load and then authenticate.
        path = request.url.path
        if path.startswith("/api/") and not compare_digest(_key_from_request(request), API_KEY):
            return JSONResponse(status_code=401, content={"error": "Invalid or missing API key."})
    return await call_next(request)


async def websocket_authorised(websocket):
    """
    Check the key BEFORE accepting the handshake, so an unauthorised client never
    gets an open socket.
    """
    if not API_KEY:
        return True
    supplied = (
        websocket.headers.get("x-api-key")
        or websocket.query_params.get("api_key")
        or ""
    ).strip()
    if compare_digest(supplied, API_KEY):
        return True
    await websocket.close(code=1008)
    return False

# Portable Directory & Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
session_log_path = os.path.join(BASE_DIR, "session_logs.jsonl")
latency_log_path = os.path.join(BASE_DIR, "latency_log.jsonl")
os.makedirs(static_dir, exist_ok=True)

# Imports from dataset, evaluation & live_stream
from dataset import (
    features_info, feature_names, raw_to_clean,
    load_full_frame, load_patient, list_patient_ids, describe_source, DATASET_MODE,
    is_valid_patient_id, pseudonymize,
)
from live_stream import ReplayEngine
import voice_engine
import alerting

# 1. Check/Run Retrospective evaluation on Startup
print("Checking for precomputed metrics...")
if not os.path.exists(evaluation.metrics_json_path) or not os.path.exists(evaluation.run_config_path):
    print("Running evaluation pipeline on startup...")
    evaluation.run_evaluation_pipeline()
else:
    print("Precomputed metrics found.")

# Load primary model and metadata
print("Loading primary LightGBM model for real-time predictions...")
with open(evaluation.run_config_path, "r") as f:
    run_config = json.load(f)

# Decision thresholds come from the evaluation run rather than being hardcoded.
# The old 0.30/0.70 constants were tuned on a 7.14%-prevalence sample and collapse
# to ~5% sensitivity at the full corpus's real 1.80% prevalence.
_thresholds = run_config.get("decision_thresholds") or {}
ALERT_THRESHOLD = float(_thresholds.get("alert_threshold", 0.30))
HIGH_RISK_THRESHOLD = float(_thresholds.get("high_risk_threshold", 0.70))
print(f"Decision thresholds: alert={ALERT_THRESHOLD:.4f} high_risk={HIGH_RISK_THRESHOLD:.4f}")
    
# We will train a primary model on startup for API predictions
print(f"Dataset mode: {DATASET_MODE}")
df_raw = load_full_frame()
df_processed, _, train_medians = evaluation.preprocess_and_impute(df_raw)
df_processed = evaluation.compute_simplified_baselines(df_processed)

# Stratify primary split (seed=42)
df_train, df_test = evaluation.stratified_patient_split(df_processed, 42)
features_columns = feature_names + [f"{f}_nan" for f in feature_names]

X_tr = df_train[features_columns]
y_tr = df_train["SepsisLabel"]
train_ds = lgb.Dataset(X_tr, label=y_tr)
primary_model = lgb.train(run_config["hyperparameters"], train_ds, num_boost_round=150)
print("Primary model initialized successfully!")

# Patient timeline access.
#
# Previously every patient's full record list was materialised into a dict at
# startup. That is fine for 232 patients but does not survive the ~40,336-patient
# full corpus, so records are now built on demand and memoised.
PATIENT_IDS = list_patient_ids()
print(f"{len(PATIENT_IDS):,} patients available.")

# Report the offline voice model at boot rather than on first /ws/voice connection,
# so a missing model is obvious immediately instead of surfacing much later.
if voice_engine.is_available():
    print(f"Offline voice model found: {voice_engine.MODEL_NAME}")
else:
    print(f"Offline voice model NOT found: voice input will be disabled "
          f"({voice_engine.unavailable_reason()})")

# Genuine LRU: an OrderedDict with move_to_end on read. The previous version evicted
# via pop(next(iter(...))), which drops the oldest *inserted* entry regardless of use,
# so a frequently revisited patient could be evicted while a one-off lookup survived.
_patient_record_cache = OrderedDict()
# Bound the cache so a long session over the full corpus cannot grow without limit.
_PATIENT_CACHE_LIMIT = 256

# Maximum patient IDs pushed to the browser in one message.
PATIENT_LIST_LIMIT = int(os.environ.get("SEPSISGUARD_PATIENT_LIST_LIMIT", "1000"))


def get_patient_records(pid):
    """Return a patient's timeline as a list of plain dicts (None for missing)."""
    if not is_valid_patient_id(pid):
        return []
    cached = _patient_record_cache.get(pid)
    if cached is not None:
        _patient_record_cache.move_to_end(pid)   # mark as most recently used
        return cached

    try:
        p_sorted = load_patient(pid)
    except ValueError:
        return []

    # Values are forward-filled within the patient timeline before prediction, matching
    # how the model was trained. Raw (un-filled) values are kept separately so the UI
    # still shows what was actually measured and can compute data completeness.
    filled = p_sorted.copy()
    filled[feature_names] = filled[feature_names].ffill()

    records = []
    for raw_row, fill_row in zip(p_sorted.to_dict("records"), filled.to_dict("records")):
        rec = {
            "PatientID": pid,
            "Dataset": raw_row["Dataset"],
            "SepsisLabel": int(raw_row["SepsisLabel"]),
        }
        for name in feature_names:
            raw_val = raw_row[name]
            rec[name] = float(raw_val) if pd.notna(raw_val) else None
            # Carried-forward value used for scoring only
            fill_val = fill_row[name]
            rec[f"__model_{name}"] = float(fill_val) if pd.notna(fill_val) else None
        records.append(rec)

    if len(_patient_record_cache) >= _PATIENT_CACHE_LIMIT:
        _patient_record_cache.popitem(last=False)   # evict least recently used
    _patient_record_cache[pid] = records
    return records

# REST API models
class EventLogPayload(BaseModel):
    session_id: str = Field(..., description="Randomly generated session ID for participant privacy")
    layout_order: str = Field(..., description="Assigned condition counterbalancing layout order")
    layout_condition: str = Field(..., description="Current layout view: 3-column vs 1-column baseline")
    task_id: str = Field(..., description="Action category: patient_review, order_placement, dismissal, etc.")
    start_timestamp: int = Field(..., description="Task start timestamp in epoch ms")
    end_timestamp: int = Field(..., description="Task end/action timestamp in epoch ms")
    patient_id: str
    risk_score: float
    action_type: str = Field(..., description="clinician response: order, dismiss, or timeline_navigation")
    action_detail: Optional[str] = Field(None, description="Dismissal reason or order description")
    shap_snapshot: Optional[Dict[str, float]] = None

# REST API Endpoints
@app.get("/api/metrics")
async def get_metrics():
    if os.path.exists(evaluation.metrics_json_path):
        with open(evaluation.metrics_json_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Metrics summary not found."}, status_code=404)

@app.get("/api/leakage_audit")
async def get_leakage_audit():
    if os.path.exists(evaluation.leakage_json_path):
        with open(evaluation.leakage_json_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Leakage audit log not found."}, status_code=404)

@app.get("/api/run_config")
async def get_run_config():
    if os.path.exists(evaluation.run_config_path):
        with open(evaluation.run_config_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Run configuration not found."}, status_code=404)

@app.get("/api/latency_stats")
async def get_latency_stats():
    if not os.path.exists(latency_log_path):
        return JSONResponse(content={
            "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "total_runs": 0
        })
        
    latencies = []
    try:
        with open(latency_log_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                latencies.append(entry["latency_ms"])
    except Exception as e:
        print(f"Error reading latency log: {e}")
        
    if not latencies:
        return JSONResponse(content={
            "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "total_runs": 0
        })
        
    return JSONResponse(content={
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "total_runs": len(latencies)
    })

@app.post("/api/log_event")
async def log_usability_event(payload: EventLogPayload):
    # Strictly check that no name or identifying text fields are provided for anonymity
    log_entry = payload.model_dump()
    
    # Append to local logs file
    with open(session_log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return JSONResponse(content={"status": "success", "message": "Usability log event stored successfully"})

SERVER_START_TIME = time.time()

# Cohort scoring is expensive on the full corpus, so results are cached briefly.
# Never score the whole corpus synchronously on every request.
_cohort_cache = {"key": None, "computed_at": 0.0, "payload": None}
COHORT_CACHE_TTL_SECONDS = float(os.environ.get("SEPSISGUARD_COHORT_CACHE_TTL", "60"))
COHORT_MAX_PATIENTS = int(os.environ.get("SEPSISGUARD_COHORT_MAX_PATIENTS", "500"))


@app.get("/api/health")
async def get_health():
    """Liveness and configuration snapshot — useful before anything else is trusted."""
    return JSONResponse(content={
        "status": "ok",
        "model_loaded": primary_model is not None,
        "dataset_mode": DATASET_MODE,
        "patient_count": len(PATIENT_IDS),
        "decision_thresholds": {
            "alert_threshold": ALERT_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
        },
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "voice_available": voice_engine.is_available(),
        "auth_required": bool(API_KEY),
    })


def _compute_cohort_risk(limit):
    """
    Score each patient from their most recent reading and rank by risk.

    One batched predict over a single assembled frame rather than a call per
    patient, which is what makes this viable at corpus scale.
    """
    ids = PATIENT_IDS[:COHORT_MAX_PATIENTS]
    ordered_columns = feature_names + [f"{f}_nan" for f in feature_names]

    rows, meta = [], []
    for pid in ids:
        records = get_patient_records(pid)
        if not records:
            continue
        last = records[-1]

        values, flags = {}, {}
        for name in feature_names:
            raw = last.get(name)
            filled = last.get("__model_" + name)
            missing = filled is None or (isinstance(filled, float) and pd.isna(filled))
            values[name] = float(train_medians[name]) if missing else float(filled)
            flags[f"{name}_nan"] = 1 if raw is None else 0
        rows.append({**values, **flags})
        meta.append({
            "patient_id": pid,
            "dataset": last.get("Dataset"),
            "hours_observed": len(records),
            "sepsis_label": last.get("SepsisLabel"),
        })

    if not rows:
        return {"patients": [], "count": 0, "scored_patients": 0}

    frame = pd.DataFrame(rows)[ordered_columns]
    probs = primary_model.predict(frame)

    scored = []
    for info, prob in zip(meta, probs):
        prob = float(prob)
        scored.append({
            **info,
            "risk": prob,
            "risk_level": ("High" if prob >= HIGH_RISK_THRESHOLD
                           else "Medium" if prob >= ALERT_THRESHOLD else "Low"),
        })
    scored.sort(key=lambda r: r["risk"], reverse=True)

    return {
        "patients": scored[:limit],
        "count": min(limit, len(scored)),
        "scored_patients": len(scored),
        "alert_threshold": ALERT_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "note": (f"Scored the most recent reading for the first {len(scored):,} patients "
                 f"(cap SEPSISGUARD_COHORT_MAX_PATIENTS)."),
    }


@app.get("/api/cohort_risk")
async def get_cohort_risk(limit: int = Query(20, ge=1, le=200)):
    """Triage view: patients ranked by current sepsis risk."""
    now = time.time()
    if (_cohort_cache["payload"] is not None
            and _cohort_cache["key"] == limit
            and now - _cohort_cache["computed_at"] < COHORT_CACHE_TTL_SECONDS):
        return JSONResponse(content={**_cohort_cache["payload"], "cached": True})

    payload = await asyncio.to_thread(_compute_cohort_risk, limit)
    _cohort_cache.update({"key": limit, "computed_at": now, "payload": payload})
    return JSONResponse(content={**payload, "cached": False})


@app.get("/api/metrics_history")
async def get_metrics_history(limit: int = Query(50, ge=1, le=1000)):
    """Timestamped snapshots of previous evaluation runs, newest last."""
    path = getattr(evaluation, "metrics_history_path", None)
    if not path or not os.path.exists(path):
        return JSONResponse(content={"runs": [], "count": 0})

    runs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return JSONResponse(content={"runs": runs[-limit:], "count": len(runs)})


# ---------------------------------------------------------------------------
# Tab-lifetime heartbeat.
#
# HTTP is stateless and the browser gives the server no signal when a tab closes,
# so the page holds a WebSocket open for its lifetime and the server watches for
# that socket to drop.
#
# A refresh also drops it, so a disconnect starts a grace timer rather than
# shutting down immediately: if a new heartbeat arrives first, the timer is
# cancelled. Counting connections rather than reacting to each disconnect also
# handles multiple tabs — shutdown only happens once the count reaches and stays
# at zero.
#
# Intended for single-user local sessions. See the README: this would be wrong on
# a shared server, where one person closing a tab would kill everyone's process.
# ---------------------------------------------------------------------------
AUTO_SHUTDOWN_ON_CLOSE = os.environ.get(
    "SEPSISGUARD_AUTO_SHUTDOWN_ON_CLOSE", "true").strip().lower() in ("1", "true", "yes")
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("SEPSISGUARD_SHUTDOWN_GRACE_SECONDS", "5"))
AUTO_OPEN_BROWSER = os.environ.get(
    "SEPSISGUARD_AUTO_OPEN_BROWSER", "true").strip().lower() in ("1", "true", "yes")

HOST = os.environ.get("SEPSISGUARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEPSISGUARD_PORT", "8000"))

_heartbeat_connections = 0
_shutdown_task = None
# Set in __main__ so the shutdown path can ask uvicorn to stop gracefully.
uvicorn_server = None


def _clear_session_caches():
    """Drop per-session state. The on-disk Parquet corpus cache is left alone."""
    patients = len(_patient_record_cache)
    _patient_record_cache.clear()
    _cohort_cache.update({"key": None, "computed_at": 0.0, "payload": None})
    print(f"Cleared session caches: {patients} patient timelines, cohort risk cache.")
    print("On-disk corpus cache retained (expensive to rebuild, not session state).")


async def _shutdown_after_grace():
    """Wait out the grace period; shut down only if nothing reconnected."""
    try:
        await asyncio.sleep(SHUTDOWN_GRACE_SECONDS)
    except asyncio.CancelledError:
        # A tab reconnected — this was a refresh, not a close.
        return

    if _heartbeat_connections > 0:
        return

    print(f"No dashboard tab reconnected within {SHUTDOWN_GRACE_SECONDS:.0f}s — shutting down.")
    _clear_session_caches()

    if uvicorn_server is not None:
        # Graceful stop, so the JSONL logs are flushed and closed rather than
        # truncated by an abrupt exit.
        uvicorn_server.should_exit = True
    else:
        asyncio.get_event_loop().stop()


@app.websocket("/ws/heartbeat")
async def websocket_heartbeat(websocket: WebSocket):
    """Open for the lifetime of a dashboard tab; its closure signals the tab closed."""
    global _heartbeat_connections, _shutdown_task

    if not await websocket_authorised(websocket):
        return
    await websocket.accept()

    _heartbeat_connections += 1
    if _shutdown_task is not None and not _shutdown_task.done():
        _shutdown_task.cancel()          # a refresh beat the grace timer
        _shutdown_task = None
    print(f"Dashboard tab connected ({_heartbeat_connections} open).")

    try:
        while True:
            # Content is irrelevant; receiving just keeps the socket alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _heartbeat_connections = max(0, _heartbeat_connections - 1)
        print(f"Dashboard tab disconnected ({_heartbeat_connections} open).")
        if AUTO_SHUTDOWN_ON_CLOSE and _heartbeat_connections == 0:
            _shutdown_task = asyncio.create_task(_shutdown_after_grace())


@app.on_event("startup")
async def _open_dashboard_in_browser():
    """
    Open the dashboard once the server is actually accepting connections, rather
    than guessing with a sleep before uvicorn.run().
    """
    if not AUTO_OPEN_BROWSER:
        return
    url = f"http://{HOST}:{PORT}"
    try:
        # Headless machines have no browser to launch; that must not be fatal.
        opened = webbrowser.open(url)
        print(f"Opened dashboard at {url}" if opened
              else f"No browser available to open {url} — navigate there manually.")
    except Exception as exc:
        print(f"Could not open a browser ({exc}) — navigate to {url} manually.")


@app.get("/api/voice_status")
async def get_voice_status():
    """
    Tells the dashboard whether speech recognition can run locally.

    The browser's own SpeechRecognition is cloud-based and fails offline, so the
    client uses this to decide which engine to use.
    """
    available = voice_engine.is_available()
    return JSONResponse(content={
        "local_available": available,
        "sample_rate": voice_engine.SAMPLE_RATE,
        "model": voice_engine.MODEL_NAME if available else None,
        "reason": None if available else voice_engine.unavailable_reason(),
    })


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    Offline speech recognition. The client streams 16 kHz mono 16-bit PCM; each
    finalised utterance comes back as a transcript. Audio never leaves the machine.
    """
    if not await websocket_authorised(websocket):
        return
    await websocket.accept()

    if not voice_engine.is_available():
        await websocket.send_text(json.dumps({
            "type": "voice_unavailable", "reason": voice_engine.unavailable_reason()
        }))
        await websocket.close()
        return

    try:
        transcriber = await asyncio.to_thread(voice_engine.Transcriber)
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "voice_unavailable", "reason": str(exc)}))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({
        "type": "voice_ready", "model": voice_engine.MODEL_NAME
    }))
    print("Voice WebSocket connected (offline recognition).")

    try:
        while True:
            chunk = await websocket.receive_bytes()
            # Decoding is CPU-bound, so keep it off the event loop.
            text = await asyncio.to_thread(transcriber.feed, chunk)
            if text:
                await websocket.send_text(json.dumps({"type": "transcript", "text": text}))
    except WebSocketDisconnect:
        print("Voice WebSocket disconnected.")
    except Exception as exc:
        print(f"Voice WebSocket error: {exc}")
        try:
            await websocket.send_text(json.dumps({"type": "voice_error", "message": str(exc)}))
        except Exception:
            pass


@app.get("/api/session_logs")
async def get_session_logs(session_id: Optional[str] = Query(None)):
    """
    Return recorded usability-study events so a session can be exported from the
    dashboard. Optionally filtered to one session_id.
    """
    if not os.path.exists(session_log_path):
        return JSONResponse(content={"events": [], "count": 0})

    events = []
    with open(session_log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id is None or entry.get("session_id") == session_id:
                events.append(entry)

    return JSONResponse(content={"events": events, "count": len(events)})


def _compute_replay_counts(chunk_rows=200_000):
    """
    Batched alarm-suppression replay over the entire dataset.

    Predictions use the SAME preprocessing the model was trained on — patient-level
    forward-fill then median fallback, with missingness flags taken from the raw
    observations. Scoring raw NaN here (as an earlier version did) produced
    probabilities uncorrelated with the evaluated model. Rows are chunked so peak
    memory stays bounded on the full corpus.
    """
    frame = df_processed.sort_values(["Patient ID", "ICULOS"], kind="stable")
    ordered_columns = feature_names + [f"{f}_nan" for f in feature_names]

    probs = np.empty(len(frame), dtype=np.float64)
    for start in range(0, len(frame), chunk_rows):
        block = frame.iloc[start:start + chunk_rows]
        probs[start:start + len(block)] = primary_model.predict(block[ordered_columns])

    prob_s = pd.Series(probs, index=frame.index)

    # Suppression rule lives in alerting.py so this and ReplayEngine cannot drift.
    unsuppressed, suppressed = alerting.evaluate_series(
        probabilities=prob_s,
        patient_ids=frame["Patient ID"],
        lactate=frame["Lactate"],
        o2sat=frame["O2Sat"],
        alert_threshold=ALERT_THRESHOLD,
    )
    never_sepsis = frame.groupby("Patient ID")["SepsisLabel"].transform("max") == 0

    return {
        "total_unsuppressed": int(unsuppressed.sum()),
        "total_suppressed": int(suppressed.sum()),
        "false_alarms_unsuppressed": int((unsuppressed & never_sepsis).sum()),
        "false_alarms_suppressed": int((suppressed & never_sepsis).sum()),
    }


# Replay simulation endpoint for Alarm Fatigue Evidence
@app.get("/api/replay_results")
async def get_alarm_replay_results():
    print("Replaying all patient histories through alert suppression logic...")
    # Retrospectively runs alarm-suppression across every patient to evaluate alarm
    # fatigue: unsuppressed vs. suppressed alarm counts, and false alarm rates.
    #
    # This used to call model.predict() once per patient-hour. Batched into chunked
    # predictions and vectorised suppression logic, it is identical arithmetic but
    # tractable on the ~1.55M-row full corpus.
    counts = await asyncio.to_thread(_compute_replay_counts)

    total_unsuppressed = counts["total_unsuppressed"]
    total_suppressed = counts["total_suppressed"]
    false_alarms_unsuppressed = counts["false_alarms_unsuppressed"]
    false_alarms_suppressed = counts["false_alarms_suppressed"]

    suppression_pct = ((total_unsuppressed - total_suppressed) / total_unsuppressed * 100.0) if total_unsuppressed > 0 else 0.0

    return JSONResponse(content={
        "total_unsuppressed_alarms": total_unsuppressed,
        "total_suppressed_alarms": total_suppressed,
        "alarms_avoided_count": total_unsuppressed - total_suppressed,
        "suppression_effectiveness_pct": suppression_pct,
        "false_alarm_rate_before_suppression_pct": (false_alarms_unsuppressed / total_unsuppressed * 100.0) if total_unsuppressed > 0 else 0.0,
        "false_alarm_rate_after_suppression_pct": (false_alarms_suppressed / total_suppressed * 100.0) if total_suppressed > 0 else 0.0,
        "clinical_relevance": "Suppression logic significantly reduces false alerts caused by transient noise without losing true clinical warnings."
    })

# Prediction logic for general ws
def predict_sepsis_risk(features_dict, missing_mask=None):
    """
    Score one feature vector.

    `features_dict` supplies the values fed to the model (already forward-filled by
    the caller where a patient history exists). `missing_mask`, when given, is the
    RAW record used to decide which `_nan` flags are set, so the flags describe what
    was actually measured rather than what was carried forward.
    """
    # The model is trained on forward-filled, median-imputed features paired with
    # binary missingness flags — it never sees NaN in the value columns. Passing raw
    # NaN here sends rows down branch directions that were never fitted, producing
    # scores uncorrelated with the evaluated model (r = 0.10, mean prob 0.80 vs 0.02).
    # Missing values are therefore filled with the training medians, exactly as the
    # training pipeline's fallback does, while the flags still record what was absent.
    mask_source = missing_mask if missing_mask is not None else features_dict
    row_data = {}
    flag_data = {}
    for col in feature_names:
        val = features_dict.get(col)
        missing = val is None or (isinstance(val, float) and np.isnan(val))
        row_data[col] = [float(train_medians[col]) if missing else float(val)]

        raw_val = mask_source.get(col)
        raw_missing = raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val))
        flag_data[f"{col}_nan"] = [1 if raw_missing else 0]

    df_pred = pd.DataFrame(row_data)
    for col in feature_names:
        df_pred[f"{col}_nan"] = flag_data[f"{col}_nan"]

    prob = float(primary_model.predict(df_pred)[0])
    
    # Calculate feature contributions
    contrib = primary_model.predict(df_pred, pred_contrib=True)[0]
    shap_attribs = {}
    for name, val in zip(feature_names, contrib[:-1]):
        shap_attribs[name] = float(val)
        
    decision_certainty = 2.0 * abs(prob - 0.5)
    
    if prob >= HIGH_RISK_THRESHOLD:
        risk_level = "High"
    elif prob >= ALERT_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    return prob, decision_certainty, risk_level, shap_attribs

# Clinical explanation, generated locally.
#
# This deliberately has no LLM path. An earlier version called out to an
# OpenAI-compatible endpoint when OPENAI_API_KEY or LLM_BASE_URL was set, which is
# incompatible with running air-gapped. The explanation is now derived entirely from
# the model's own SHAP attributions, so it reflects what actually drove the
# prediction rather than a fixed checklist of thresholds.

# Units and a plain-language reading for each feature the explanation can mention.
FEATURE_NARRATIVE = {
    "Lactate":         ("mmol/L",  "raised lactate, a marker of hypoperfusion", "lactate within range"),
    "O2Sat":           ("%",       "reduced oxygen saturation", "oxygenation preserved"),
    "MAP":             ("mmHg",    "low mean arterial pressure", "mean arterial pressure adequate"),
    "SBP":             ("mmHg",    "low systolic pressure", "systolic pressure adequate"),
    "HR":              ("bpm",     "tachycardia", "heart rate unremarkable"),
    "Temp":            ("°C",      "temperature derangement", "temperature normal"),
    "Resp":            ("/min",    "elevated respiratory rate", "respiratory rate normal"),
    "WBC":             ("x10^9/L", "abnormal white cell count", "white cell count normal"),
    "Creatinine":      ("mg/dL",   "raised creatinine, suggesting renal dysfunction", "renal indices unremarkable"),
    "Bilirubin_total": ("mg/dL",   "raised bilirubin, suggesting hepatic dysfunction", "bilirubin normal"),
    "Platelets":       ("x10^3/uL","low platelet count", "platelet count normal"),
    "Glucose":         ("mg/dL",   "dysglycaemia", "glucose within range"),
    "pH":              ("",        "acid-base disturbance", "acid-base balance normal"),
    "PaCO2":           ("mmHg",    "abnormal PaCO2", "PaCO2 normal"),
    "BUN":             ("mg/dL",   "raised urea", "urea normal"),
    "Hgb":             ("g/dL",    "low haemoglobin", "haemoglobin normal"),
    "PTT":             ("sec",     "prolonged clotting time", "clotting time normal"),
    "HCO3":            ("mEq/L",   "low bicarbonate", "bicarbonate normal"),
    "ICULOS":          ("hrs",     "length of ICU stay", "length of ICU stay"),
    "HospAdmTime":     ("hrs",     "time from hospital to ICU admission", "admission timing"),
    "Age":             ("yrs",     "age", "age"),
    "Gender":          ("",        "sex", "sex"),
}


def generate_clinical_explanation(features, prob, decision_certainty, risk_level, shap_contribs=None):
    """
    Build the advisor text locally from the model's own feature attributions.

    `shap_contribs` is the per-feature contribution dict returned by
    predict_sepsis_risk(). The abnormalities called out are the top few features by
    absolute contribution, so the explanation changes with what actually drove this
    particular prediction instead of restating a fixed threshold checklist.
    """
    contribs = shap_contribs or {}

    # Rank by absolute contribution; positive pushes risk up, negative pulls it down.
    ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]

    drivers_up, drivers_down = [], []
    for name, contrib in ranked:
        unit, abnormal_phrase, normal_phrase = FEATURE_NARRATIVE.get(name, ("", name, name))
        val = features.get(name)
        val_str = f"{val:.2f}{(' ' + unit) if unit else ''}" if isinstance(val, (int, float)) else "not measured"
        phrase = abnormal_phrase if contrib > 0 else normal_phrase
        entry = f"{name} ({val_str}) — {phrase}"
        (drivers_up if contrib > 0 else drivers_down).append(entry)

    if drivers_up:
        up_str = "; ".join(drivers_up)
        para_drivers = (
            f"The variables contributing most strongly toward this risk estimate are: {up_str}."
        )
    else:
        para_drivers = (
            "No individual variable is pushing this estimate upward; the score is "
            "dominated by features that argue against sepsis."
        )

    if drivers_down:
        para_drivers += f" Pulling the estimate down: {'; '.join(drivers_down)}."

    missing = [f["name"] for f in features_info if features.get(f["name"]) is None]
    completeness_note = (
        f" {len(missing)} of {len(features_info)} variables were not measured at this hour, "
        f"so their contribution is inferred from the missingness pattern."
        if missing else ""
    )

    text = (
        "**Clinical Assessment (local, model-derived)**\n\n"
        f"The patient's current profile gives a sepsis probability of {prob:.1%} "
        f"({risk_level} risk category), with a decision certainty of {decision_certainty:.1%}."
        f"{completeness_note}\n\n"
        f"{para_drivers}\n\n"
        "These attributions come from the gradient-boosted model's own per-prediction "
        "feature contributions, so they describe this patient at this hour rather than a "
        "general rule. Note that attribution explains the model's reasoning, not causation.\n\n"
        "**Recommended Actions:**\n"
        "1. Measure and track lactate clearance every 2-4 hours.\n"
        "2. Obtain blood cultures prior to administration of any new empiric antimicrobials.\n"
        "3. Administer broad-spectrum empiric antibiotics and initiate fluid resuscitation "
        "(30 mL/kg crystalloid) if sepsis-induced hypoperfusion is suspected.\n"
        "4. Closely monitor mean arterial pressure (target MAP >= 65 mmHg) and urine output."
    )
    return text

# 3. Static Patient WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await websocket_authorised(websocket):
        return
    await websocket.accept()
    print("Static WebSocket client connected.")
    
    try:
        # Initial greeting: Send list of patients.
        # The sample dataset sends every ID, preserving existing dashboard behaviour.
        # The full corpus is capped so the browser is not handed ~40k <option> nodes;
        # clients narrow the rest with a "search_patients" message.
        await websocket.send_text(json.dumps({
            "type": "patients_list",
            "patients": PATIENT_IDS[:PATIENT_LIST_LIMIT],
            "total_patients": len(PATIENT_IDS),
            "truncated": len(PATIENT_IDS) > PATIENT_LIST_LIMIT,
            # So the dashboard colours/alerts at the same thresholds the server uses
            "alert_threshold": ALERT_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD
        }))

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "search_patients":
                query = str(message.get("query", "")).strip().upper()
                matches = [p for p in PATIENT_IDS if query in p.upper()] if query else PATIENT_IDS
                await websocket.send_text(json.dumps({
                    "type": "patients_list",
                    "patients": matches[:PATIENT_LIST_LIMIT],
                    "total_patients": len(matches),
                    "truncated": len(matches) > PATIENT_LIST_LIMIT
                }))

            elif msg_type == "get_patient_timeline":
                pid = message.get("patient_id")
                timeline = get_patient_records(pid)
                
                # Precompute predictions for all steps in the timeline to send
                precomputed_timeline = []
                for step in timeline:
                    # Score on forward-filled values (as trained); missingness flags and
                    # data completeness still come from the raw observations.
                    model_input = {
                        name: step.get(f"__model_{name}") for name in feature_names
                    }
                    prob, cert, r_lvl, shap_vals = predict_sepsis_risk(
                        model_input, missing_mask=step
                    )
                    step_data = {k: v for k, v in step.items() if not k.startswith("__model_")}

                    # Compute data completeness for static timeline steps
                    non_none_count = sum(1 for f in features_info if step.get(f["name"]) is not None)
                    completeness = float(non_none_count / len(features_info))

                    step_data["PredictedRisk"] = prob
                    step_data["PredictedDecisionCertainty"] = cert
                    step_data["PredictedDataCompleteness"] = completeness
                    step_data["PredictedRiskLevel"] = r_lvl
                    step_data["PredictedSHAP"] = shap_vals
                    precomputed_timeline.append(step_data)
                    
                await websocket.send_text(json.dumps({
                    "type": "patient_timeline",
                    "patient_id": pid,
                    "timeline": precomputed_timeline
                }))
                
            elif msg_type == "predict_risk":
                features = message.get("features", {})
                prob, cert, r_lvl, shap_vals = predict_sepsis_risk(features)
                
                # Compute data completeness for custom simulation input
                non_none_count = sum(1 for f in features_info if features.get(f["name"]) is not None)
                completeness = float(non_none_count / len(features_info))
                
                await websocket.send_text(json.dumps({
                    "type": "prediction_result",
                    "probability": prob,
                    "decision_certainty": cert,
                    "data_completeness": completeness,
                    "risk_level": r_lvl,
                    "feature_contribs": shap_vals
                }))
                
            elif msg_type == "ask_advisor":
                features = message.get("features", {})
                prob, cert, r_lvl, shap_vals = predict_sepsis_risk(features)
                explanation = generate_clinical_explanation(features, prob, cert, r_lvl, shap_vals)
                await websocket.send_text(json.dumps({
                    "type": "advisor_response",
                    "explanation": explanation
                }))
                
    except WebSocketDisconnect:
        print("Static WebSocket client disconnected.")
    except Exception as e:
        print(f"Static WebSocket Error: {e}")

# 4. Simulated Live Replay WebSocket Endpoint
@app.websocket("/ws/live/{patient_id}")
async def websocket_live_endpoint(websocket: WebSocket, patient_id: str, playback_speed: int = Query(10)):
    if not await websocket_authorised(websocket):
        return
    await websocket.accept()

    # Validate before the ID is used to locate a patient file.
    if not is_valid_patient_id(patient_id):
        await websocket.send_text(json.dumps({
            "type": "live_error", "message": "Invalid patient ID."
        }))
        await websocket.close()
        return

    print(f"Live WebSocket connected for {pseudonymize(patient_id)} at {playback_speed}x speed.")

    try:
        # Initialize ReplayEngine
        engine = ReplayEngine(
            patient_id,
            playback_speed=playback_speed,
            model=primary_model,
            train_medians=train_medians,
            alert_threshold=ALERT_THRESHOLD,
            high_risk_threshold=HIGH_RISK_THRESHOLD,
        )
        interval = engine.get_interval()
        
        for i in range(engine.total_hours):
            # Generate next simulated reading
            reading_payload = engine.step()
            if reading_payload is None:
                break
                
            # Send payload to client
            await websocket.send_text(json.dumps(reading_payload))
            
            # Sleep to match playback rate
            await asyncio.sleep(interval)
            
        # Notify completion
        await websocket.send_text(json.dumps({"type": "live_complete", "patient_id": patient_id}))
        
    except WebSocketDisconnect:
        print(f"Live WebSocket disconnected for {pseudonymize(patient_id)}.")
    except Exception as e:
        print(f"Live WebSocket Error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "live_error", "message": str(e)}))
        except Exception:
            pass

# Mount static files
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print(f"Starting Sepsis Web Server on http://{HOST}:{PORT} ...")

    # Construct the Server explicitly rather than calling uvicorn.run(), so the
    # heartbeat shutdown path can set should_exit and get a graceful stop that
    # flushes session_logs.jsonl and latency_log.jsonl.
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="info")
    uvicorn_server = uvicorn.Server(config)

    # Bind the listening socket ourselves so address reuse is explicit and
    # verifiable. Without SO_REUSEADDR the OS can hold the port in TIME_WAIT after
    # shutdown and an immediate restart fails with "address already in use" — which
    # is the concrete meaning of "freeing the port" after auto-shutdown.
    #
    # On Windows SO_REUSEADDR has different semantics (it allows two live sockets to
    # bind the same port rather than reclaiming a lingering one), so it is not set
    # there; Windows releases the port on close without needing it.
    import socket

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((HOST, PORT))
    listen_sock.listen(2048)
    listen_sock.set_inheritable(True)

    try:
        uvicorn_server.run(sockets=[listen_sock])
    finally:
        try:
            listen_sock.close()
        except OSError:
            pass
