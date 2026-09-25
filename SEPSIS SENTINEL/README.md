# SepsisGuard

**SepsisGuard** is an offline clinical decision support prototype for early sepsis detection in
ICU patients. It pairs a LightGBM classifier trained on the PhysioNet/CinC Challenge 2019 data
with a browser dashboard offering live patient replay, SHAP explanations, alarm-fatigue
suppression, hands-free voice control and a usability-study harness.

> **Fully offline at runtime by design.** Nothing in this codebase makes an outbound network
> call while running. The LLM advisor path has been removed and the browser's cloud-based speech
> recognition fallback has been removed — the advisor and voice input only ever run locally.

> **Research prototype.** Trained on retrospective data. **Not cleared for live diagnostic
> deployment or unmonitored clinical use.** See [Limitations](#limitations).

---

## Architecture

| Component | Role |
|---|---|
| `app.py` | FastAPI server: REST endpoints, WebSockets, model loading, auth, lifecycle |
| `dataset.py` | Single source of truth for data loading; switches between sample and full corpus |
| `evaluation.py` | Retrospective metrics, bootstrap CIs, threshold selection, calibration, external validation |
| `live_stream.py` | `ReplayEngine` — streams a patient timeline hour by hour with causal-only imputation |
| `alerting.py` | The alarm-suppression rule, defined once and shared by batch and streaming paths |
| `voice_engine.py` | Offline speech recognition via Vosk |
| `static/` | Dashboard: HTML, CSS, client JS, voice modules, locally vendored fonts and icons |

**Model.** LightGBM over 22 ICU vitals and labs, each paired with a binary missingness flag
(44 inputs). Missingness is signal — which labs a clinician ordered carries information — so
it is modelled explicitly rather than imputed away.

**Dataset modes.** `sample` (default) uses the bundled 232-patient Excel extract for fast setup.
`full` uses the complete ~40,336-patient PhysioNet corpus from per-patient `.psv` files. Results
are written to separate files per mode so they can never be confused.

**Live replay.** Streams a retrospective timeline at 1x/10x/60x over a WebSocket, forward-filling
only from past observations so no future information leaks into a prediction.

**Voice.** Recognition runs locally with Vosk. Audio never leaves the machine.

---

## Prerequisites

- **Python 3.10+** (developed against 3.13)
- ~2 GB free disk for dependencies, plus ~39 MB for the voice model and ~322 MB if you use the
  full PhysioNet corpus

---

## Installation

### Standard (machine with internet)

```bash
python -m venv .venv
```
```bash
# Windows
.venv\Scripts\Activate.ps1
```
```bash
# Linux / macOS
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

### Air-gapped install

`pip` itself needs to reach PyPI, so **running `pip install -r requirements.txt` directly on the
air-gapped machine will fail.** Build a wheelhouse on a networked machine first and carry it
across.

On a machine **with** internet access:

```bash
pip download -r requirements.txt -d wheelhouse/
```

Transfer `wheelhouse/` and `requirements.txt` to the air-gapped machine, then:

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

Every dependency is pinned to an exact version so the wheelhouse and the offline install cannot
drift apart.

### Fetch model files before going offline

Two artefacts are downloaded once, ahead of time, and must be present on disk before the machine
is disconnected — same principle as the pip wheelhouse:

```bash
python scripts/fetch_voice_model.py
```
Downloads the ~39 MB Vosk speech model to `models/`. Without it, voice input is disabled (the app
still runs normally).

For the full corpus, download `training_setA` and `training_setB` from
[physionet.org/content/challenge-2019](https://physionet.org/content/challenge-2019/) into
`data/physionet2019/`. Without it, the app uses the bundled sample.

---

## Running

```bash
python app.py
```

or

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

The server trains the primary model in memory on startup (~15 s on the sample, a few minutes on
the full corpus), then serves the dashboard at **http://127.0.0.1:8000**.

### Auto-open browser

On startup the dashboard opens in your default browser automatically. This fires from the
FastAPI startup event — when the server is actually accepting connections, not after a guessed
delay. It is wrapped so a headless machine with no browser to launch logs a message instead of
failing. Disable with `SEPSISGUARD_AUTO_OPEN_BROWSER=false`.

### Auto-shutdown when the tab closes

The page holds a WebSocket (`/ws/heartbeat`) open for its lifetime. When it closes, the server
waits out a short grace period and then shuts down cleanly, clearing in-memory caches.

- A **page refresh** reconnects within the grace period, so it does not shut down.
- With **multiple tabs open**, closing one does not shut down — only the last one does.
- Shutdown is graceful (`should_exit`), so `session_logs.jsonl` and `latency_log.jsonl` are
  flushed and closed rather than truncated.
- The listening socket sets `SO_REUSEADDR` on POSIX, so the port is immediately reusable by a
  fresh run rather than being held in `TIME_WAIT`.
- The on-disk Parquet corpus cache is **retained** — it is expensive to rebuild and is not
  per-session state.

> ⚠️ **Intended for single-user local sessions only.** This is inappropriate for a shared
> multi-user server: one person closing their last tab would kill the process for everyone.
> Disable with `SEPSISGUARD_AUTO_SHUTDOWN_ON_CLOSE=false`.

---

## Environment variables

All optional — the app runs with none of them set.

| Variable | Default | Purpose |
|---|---|---|
| `SEPSISGUARD_DATASET_MODE` | `sample` | `sample` (bundled extract) or `full` (PhysioNet corpus) |
| `SEPSISGUARD_PSV_ROOT` | `data/physionet2019` | Where the extracted PSV set directories live |
| `SEPSISGUARD_VOICE_MODEL` | `models/vosk-model-small-en-us-0.15` | Offline speech model directory |
| `SEPSISGUARD_API_KEY` | *(unset)* | When set, required on `/api/*` and every WebSocket handshake. Unset means no auth |
| `SEPSISGUARD_CORS_ORIGINS` | `http://127.0.0.1:8000,http://localhost:8000` | Comma-separated allowed origins; `*` allows all |
| `SEPSISGUARD_PATIENT_LIST_LIMIT` | `1000` | Max patient IDs pushed to the browser at once |
| `SEPSISGUARD_AUTO_OPEN_BROWSER` | `true` | Open the dashboard in the default browser on startup |
| `SEPSISGUARD_AUTO_SHUTDOWN_ON_CLOSE` | `true` | Shut down when the last dashboard tab closes |
| `SEPSISGUARD_SHUTDOWN_GRACE_SECONDS` | `5` | Grace period before shutdown, so a refresh does not trigger it |
| `SEPSISGUARD_HOST` / `SEPSISGUARD_PORT` | `127.0.0.1` / `8000` | Bind address |
| `SEPSISGUARD_COHORT_CACHE_TTL` | `60` | Seconds to cache cohort-risk results |
| `SEPSISGUARD_COHORT_MAX_PATIENTS` | `500` | Cap on patients scored per cohort request |

---

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Model loaded, dataset mode, patient count, thresholds, uptime, voice availability |
| `GET /api/metrics` | Full metrics summary: AUROC/AUPRC/CIs, subgroups, calibration, XAI |
| `GET /api/metrics_history` | Timestamped snapshots of previous evaluation runs (`?limit=N`) |
| `GET /api/cohort_risk` | Patients ranked by current risk for triage (`?limit=N`) |
| `GET /api/run_config` | Split sizes, feature list, hyperparameters, selected thresholds |
| `GET /api/leakage_audit` | SOFA-feature label-leakage correlations |
| `GET /api/latency_stats` | Inference latency mean/p50/p95/p99 |
| `GET /api/replay_results` | Alarm-suppression effectiveness across all patients |
| `GET /api/voice_status` | Whether local speech recognition is available, and why not if it isn't |
| `GET /api/session_logs` | Recorded usability-study events (`?session_id=` to filter) |
| `POST /api/log_event` | Append an anonymized usability-study event |
| `WS /ws` | Patient timelines, what-if prediction, advisor, patient search |
| `WS /ws/live/{patient_id}` | Simulated live replay stream (`?playback_speed=N`) |
| `WS /ws/voice` | Offline speech recognition: send 16 kHz mono PCM16, receive transcripts |
| `WS /ws/heartbeat` | Tab-lifetime signal used for auto-shutdown |

---

## Evaluation

```bash
python evaluation.py
```

Writes `metrics_summary.json`, `run_config.json`, `leakage_audit.json` and appends a snapshot to
`metrics_history.jsonl`. In `full` mode these carry a `_full` suffix so sample results are never
overwritten.

Includes patient-level stratified splits, 1,000 paired bootstrap resamples, Holm-Bonferroni
corrected baseline comparisons, multi-seed stability, decision-threshold selection on a held-out
validation split, calibration deciles, and cross-hospital external validation.

---

## Results — full PhysioNet corpus

All figures below are read from `metrics_summary_full.json`, produced by
`SEPSISGUARD_DATASET_MODE=full python evaluation.py`. Patient-level stratified 80/20 split
(split seed 42, model seed 42), 1,000 paired bootstrap resamples.

| Corpus | Value |
|---|---|
| Rows | 1,552,210 hourly observations |
| Patients | 40,336 (20,336 Set A + 20,000 Set B) |
| Patients who ever develop sepsis | 2,932 |
| Test split | 8,068 patients / 311,614 rows |
| Positive rate | **1.78%** |

### Discrimination

| Predictor | AUROC (95% CI) | AUPRC | Brier |
|---|---|---|---|
| **LightGBM (SepsisGuard)** | **0.814 (0.808–0.820)** | **0.099 (0.094–0.105)** | **0.017** |
| Simplified NEWS2 | 0.624 (0.617–0.632) | 0.029 | — |
| Simplified qSOFA | 0.579 (0.572–0.586) | 0.022 | — |

Across seeds `[42, 100, 2026, 999, 123]`: AUROC **0.821 ± 0.006**, AUPRC 0.098 ± 0.002.

AUPRC must be read against the **1.78% base rate** — 0.099 is a **5.6× lift over chance**, not a
poor score. It is not comparable to the sample-mode AUPRC, which is inflated by that extract's
7.14% positive rate.

### Against clinical baselines

| Comparison | AUROC difference (95% CI) | Adjusted p |
|---|---|---|
| vs. Simplified qSOFA | **+0.234 (0.227–0.242)** | `< 0.001` |
| vs. Simplified NEWS2 | **+0.190 (0.182–0.198)** | `< 0.001` |

The model significantly outperforms **both** baselines. Neither difference CI approaches zero.

> This is the claim the 232-patient sample could not support: there, the NEWS2 difference was
> `+0.014` with `p = 0.160`. The small enriched extract flattered NEWS2 (0.928 vs 0.624 here).
> Scale changed the conclusion, so quote the full-corpus figures.

*Two-sided paired bootstrap, Holm-Bonferroni corrected. With 1,000 resamples the smallest
resolvable p-value is 0.001; `< 0.001` is the floor and cannot honestly be quoted more precisely.*

### Operating points

Thresholds are selected on a validation split held out from the training patients
(6,454 patients / 248,648 rows) — never on the test set.

| Operating point | Threshold | Sensitivity | Specificity | PPV | NPV | Share of hours flagged |
|---|---|---|---|---|---|---|
| Alert (target 85% sens) | 0.0109 | 83.0% | 61.4% | 3.8% | 99.5% | **39.4%** |
| Youden's J | 0.0162 | 72.0% | 74.8% | 4.9% | 99.3% | 26.1% |
| High risk (target 50% sens) | 0.0416 | 47.6% | 91.4% | 9.1% | 99.0% | 9.3% |
| Legacy fixed 0.30 | 0.30 | **5.1%** | 99.6% | 19.7% | 98.3% | 0.5% |

Two things this table settles:

- **The legacy 0.30 cut-off is unusable at scale.** It looks excellent on specificity while
  catching only **1 in 20** septic patient-hours. Thresholds have to be selected against the
  actual base rate, not assumed.
- **Alarm burden is the real constraint**, not discrimination. At the sensitivity-optimised
  threshold the model flags ~39% of all patient-hours at 3.8% PPV. Choosing an operating point
  is a clinical trade-off, not a tuning detail.

### Alarm suppression

Every patient-hour in the corpus replayed through the suppression rule
(`GET /api/replay_results`, measured at the selected alert threshold of 0.0109):

| | Alarms | Share of all patient-hours |
|---|---|---|
| Raw (threshold only) | 608,989 | 39.2% |
| After suppression | 531,034 | 34.2% |
| **Avoided** | **77,955** | **12.8% of raw alarms** |

| False-alarm rate | Before | After |
|---|---|---|
| Alarms raised on patients who never develop sepsis | 76.8% | **74.7%** |

An alarm is suppressed as a transient spike unless it is *sustained* — the previous hour for the
same patient also crossed the threshold — or accompanied by severe organ dysfunction
(Lactate > 2.0 mmol/L **and** O₂Sat < 90%). "False alarm rate" here is patient-level: the share
of raised alarms belonging to patients who never develop sepsis at any point, not row-level PPV.

> **Suppression helps far less than the sample suggests.** It removes 12.8% of alarms and moves
> the false-alarm rate by 2.1 percentage points — from 76.8% to 74.7%. Roughly three in four
> alarms still fire on a patient who never becomes septic, and the dashboard still surfaces
> ~531,000 alarms across the corpus. Transient-spike filtering is not a solution to alarm
> fatigue at this operating point; the threshold choice dominates.

These figures are computed on the full corpus and are **not comparable to sample-mode results**,
which are measured over 232 patients at a different threshold.

### Cross-hospital external validation

Train on one hospital, test on the other — no patient overlap:

| Direction | Test rows | AUROC | AUPRC | Brier |
|---|---|---|---|---|
| Set A → Set B | 761,995 | 0.747 | 0.065 | 0.014 |
| Set B → Set A | 790,215 | 0.749 | 0.074 | 0.023 |

Both land **~0.065 below** the 0.814 within-corpus figure, and the drop is symmetric, so it
reflects genuine site transfer rather than one hospital being harder. **Report 0.75, not 0.81,
as the expected performance at a new site**, and recalibrate thresholds locally.

### Subgroups

| Subgroup | N rows | Positives | AUROC (95% CI) |
|---|---|---|---|
| Age < 65 | 159,716 | 2,861 | 0.814 (0.807–0.823) |
| Age ≥ 65 | 151,898 | 2,692 | 0.813 (0.806–0.821) |
| Female | 137,052 | 2,226 | 0.819 (0.810–0.827) |
| Male | 174,562 | 3,327 | 0.810 (0.802–0.817) |

Every subgroup is well powered (2,226–3,327 positives) and performance is consistent to within
0.01 AUROC. No subgroup disparity is detectable.

### Calibration & explainability

Calibration is close across all ten equal-count deciles; the model is mildly **under**-confident
at the low end (0.26% predicted vs 0.09% observed in decile 1) and mildly **over**-confident in
the top decile (11.0% predicted vs 8.8% observed).

SHAP top-5: `ICULOS, HospAdmTime, Temp, HR, WBC` — stability 0.802 under 1% σ input noise.

> ⚠️ **Time features dominate.** ICU length-of-stay and admission timing outrank every
> physiological variable, and agreement with the Cohen's *d* clinical ranking falls to **1 of 5**
> (WBC alone). Some of this model's discrimination comes from *how long a patient has been in the
> ICU* rather than from their physiology. This is a known characteristic of the PhysioNet 2019
> data and should be stated explicitly in any writeup rather than left for a reviewer to find.

### Sample mode, for contrast

The bundled 232-patient extract yields AUROC 0.943 — substantially optimistic, from 30
ever-sepsis patients and an enriched 7.14% positive rate. It exists for fast setup and UI work.
**Cite the full-corpus figures.** Sample results live in the unsuffixed
`metrics_summary.json` and are never overwritten by a full run.

---

## Limitations

- **Alarm burden.** At the sensitivity-optimised threshold the model flags **39.4% of all
  patient-hours at 3.8% PPV** — roughly 27 alerts per true positive. Transient-spike suppression
  removes only **12.8%** of those alarms and shifts the false-alarm rate from 76.8% to 74.7%, so
  it does not solve the problem: **~531,000 alarms remain across the corpus, three in four on
  patients who never become septic.** The operating point, not the suppression rule, is the lever
  that matters. See [Operating points](#operating-points) and
  [Alarm suppression](#alarm-suppression).
- **Site transfer.** Cross-hospital AUROC is **0.747–0.749 versus 0.814** within-corpus, in both
  directions. Expect ~0.75 at a new site, and recalibrate thresholds locally rather than carrying
  them across.
- **Time-based features dominate.** ICU length-of-stay and time-to-ICU-admission are the two
  strongest SHAP contributors, ranking above every physiological variable. Part of the model's
  discrimination reflects care-pathway timing rather than patient physiology.
- **Class imbalance.** Only 1.78% of patient-hours are positive. AUPRC (0.099) is far more
  informative than AUROC here, and headline AUROC alone will overstate clinical usefulness.
- **Voice recognition** is English-only and depends on a small acoustic model; expect errors in
  noisy environments. Ordering commands always require spoken confirmation.

---

## License

MIT. See `LICENSE`.
