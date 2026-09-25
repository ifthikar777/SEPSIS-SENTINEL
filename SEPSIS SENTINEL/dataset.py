"""
Single source of truth for loading the SepsisGuard clinical dataset.

Two modes, selected by DATASET_MODE (override with the SEPSISGUARD_DATASET_MODE
environment variable):

  "sample" -- the bundled Excel workbook (data/PhysioNet2019_Combined_SetsAB.xlsx),
              10,000 hourly rows across 232 patients. Default; keeps existing
              published results reproducible.

  "full"   -- the raw PhysioNet/CinC Challenge 2019 training corpus as per-patient
              pipe-separated .psv files (~40,336 patients / ~1.55M rows), read from
              SEPSISGUARD_PSV_ROOT (default: data/physionet2019/).

Both modes are normalised to one canonical frame so no downstream code needs to
know which source it came from:

    Patient ID | Dataset | SepsisLabel | <the 22 feature columns, clean names>

The 22 feature names below are deliberately identical to the PhysioNet PSV column
names, so "full" mode needs no column renaming at all.
"""

import os
import re
import glob
import json
import hashlib
import numpy as np
import pandas as pd

# Patient IDs are "<A|B>_<stem>" where the stem is the PSV filename. Because that stem
# is used to build a filesystem path, and IDs arrive from the client over the WebSocket
# URL, anything not matching this pattern is rejected before it reaches the filesystem.
PATIENT_ID_RE = re.compile(r"^[AB]_p\d{1,10}$")


def is_valid_patient_id(patient_id):
    return isinstance(patient_id, str) and bool(PATIENT_ID_RE.match(patient_id))


def pseudonymize(patient_id):
    """
    Short stable digest of a patient ID, for logs.

    Log files persist outside the application, so they should not carry record
    identifiers in clear text. The digest is stable, so per-patient grouping in
    latency analysis still works, but it is not reversible.
    """
    return "pt_" + hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()[:12]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mode & source configuration
DATASET_MODE = os.environ.get("SEPSISGUARD_DATASET_MODE", "sample").strip().lower()
PSV_ROOT = os.environ.get(
    "SEPSISGUARD_PSV_ROOT", os.path.join(BASE_DIR, "data", "physionet2019")
)

EXCEL_PATH = os.path.join(BASE_DIR, "data", "PhysioNet2019_Combined_SetsAB.xlsx")
EXCEL_SHEET = "Combined_Data"

# Cache for "full" mode, so the 40k-file directory walk happens once.
CACHE_PATH = os.path.join(BASE_DIR, "data", "_full_corpus_cache.parquet")

# Feature definitions.
#   raw     -- column header in the sample Excel workbook
#   name    -- canonical name (also the PhysioNet PSV column name)
#   default -- fallback when a feature is missing entirely
features_info = [
    {"raw": "Age\n(yrs)", "name": "Age", "default": 60.0},
    {"raw": "Gender\n(0=F,1=M)", "name": "Gender", "default": 1.0},
    {"raw": "ICU LOS\n(hrs)", "name": "ICULOS", "default": 24.0},
    {"raw": "Hosp→ICU\n(hrs)", "name": "HospAdmTime", "default": -48.0},
    {"raw": "HR\n(bpm)", "name": "HR", "default": 80.0},
    {"raw": "O₂ Sat\n(%)", "name": "O2Sat", "default": 97.0},
    {"raw": "Temp\n(°C)", "name": "Temp", "default": 37.0},
    {"raw": "SBP\n(mmHg)", "name": "SBP", "default": 120.0},
    {"raw": "MAP\n(mmHg)", "name": "MAP", "default": 80.0},
    {"raw": "Resp\n(/min)", "name": "Resp", "default": 16.0},
    {"raw": "WBC\n(×10⁹/L)", "name": "WBC", "default": 8.0},
    {"raw": "Creatinine\n(mg/dL)", "name": "Creatinine", "default": 1.0},
    {"raw": "Bilirubin\n(mg/dL)", "name": "Bilirubin_total", "default": 0.8},
    {"raw": "Platelets\n(×10³/µL)", "name": "Platelets", "default": 200.0},
    {"raw": "Glucose\n(mg/dL)", "name": "Glucose", "default": 100.0},
    {"raw": "Lactate\n(mmol/L)", "name": "Lactate", "default": 1.2},
    {"raw": "pH", "name": "pH", "default": 7.4},
    {"raw": "PaCO₂\n(mmHg)", "name": "PaCO2", "default": 40.0},
    {"raw": "BUN\n(mg/dL)", "name": "BUN", "default": 15.0},
    {"raw": "Hgb\n(g/dL)", "name": "Hgb", "default": 12.0},
    {"raw": "PTT\n(sec)", "name": "PTT", "default": 30.0},
    {"raw": "HCO₃\n(mEq/L)", "name": "HCO3", "default": 24.0},
]

feature_raws = [f["raw"] for f in features_info]
feature_names = [f["name"] for f in features_info]
raw_to_clean = {f["raw"]: f["name"] for f in features_info}

# Canonical, non-feature columns present on every loaded frame.
ID_COLUMNS = ["Patient ID", "Dataset", "SepsisLabel"]
CANONICAL_COLUMNS = ID_COLUMNS + feature_names

# Directory names the official challenge archives unpack into.
# training_setA.zip -> "training/", training_setB.zip -> "training_setB/".
_SET_DIR_CANDIDATES = {
    "Set A": ["training_setA", "training", "setA", "A"],
    "Set B": ["training_setB", "setB", "B"],
}


def resolve_psv_dirs():
    """Return [(set_label, directory), ...] for whichever set directories exist."""
    found = []
    for label, candidates in _SET_DIR_CANDIDATES.items():
        for cand in candidates:
            path = os.path.join(PSV_ROOT, cand)
            if os.path.isdir(path) and glob.glob(os.path.join(path, "*.psv")):
                found.append((label, path))
                break
    return found


def _patient_id_from_path(path, set_label):
    """training_setB/p012345.psv + 'Set B' -> 'B_p012345'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{set_label[-1]}_{stem}"


def _normalise_psv_frame(df, patient_id, set_label):
    """Coerce one raw PSV frame onto the canonical schema."""
    out = pd.DataFrame(index=df.index)
    out["Patient ID"] = patient_id
    out["Dataset"] = set_label
    out["SepsisLabel"] = (
        pd.to_numeric(df["SepsisLabel"], errors="coerce").fillna(0).astype(int)
        if "SepsisLabel" in df.columns
        else 0
    )
    for name in feature_names:
        # A column absent from this hospital's export becomes all-NaN, which the
        # missingness-flag pipeline already handles correctly. Use float NaN rather
        # than pd.NA so the column stays float64 and round-trips through Parquet.
        out[name] = (
            pd.to_numeric(df[name], errors="coerce").astype("float64")
            if name in df.columns
            else np.nan
        )
    return out[CANONICAL_COLUMNS]


def _read_psv(path, patient_id, set_label):
    df = pd.read_csv(path, sep="|")
    return _normalise_psv_frame(df, patient_id, set_label)


def _load_excel_frame():
    """Load the bundled sample workbook and normalise to canonical columns."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET, header=1)
    out = pd.DataFrame(index=df.index)
    out["Patient ID"] = df["Patient ID"]
    out["Dataset"] = df["Dataset"]
    out["SepsisLabel"] = df["Sepsis\nLabel"].astype(int)
    for f in features_info:
        out[f["name"]] = pd.to_numeric(df[f["raw"]], errors="coerce")
    return out[CANONICAL_COLUMNS]


# Module-level memo so repeated calls (e.g. one ReplayEngine per WebSocket
# connection) never re-read the source.
_frame_memo = None


def load_full_frame(use_cache=True, verbose=True):
    """Return the whole dataset as one canonical DataFrame."""
    global _frame_memo
    if _frame_memo is not None:
        return _frame_memo

    if DATASET_MODE == "full":
        if use_cache and os.path.exists(CACHE_PATH):
            if verbose:
                print(f"Loading full corpus from cache: {CACHE_PATH}")
            _frame_memo = pd.read_parquet(CACHE_PATH)
            return _frame_memo

        dirs = resolve_psv_dirs()
        if not dirs:
            raise FileNotFoundError(
                f"DATASET_MODE='full' but no PSV set directories found under {PSV_ROOT}.\n"
                f"Expected e.g. {os.path.join(PSV_ROOT, 'training')} and "
                f"{os.path.join(PSV_ROOT, 'training_setB')} containing *.psv files.\n"
                "Download training_setA.zip / training_setB.zip from "
                "https://physionet.org/content/challenge-2019/ and extract them there."
            )

        parts = []
        for set_label, directory in dirs:
            files = sorted(glob.glob(os.path.join(directory, "*.psv")))
            if verbose:
                print(f"Reading {len(files):,} patient files from {directory} ...")
            for i, path in enumerate(files):
                parts.append(_read_psv(path, _patient_id_from_path(path, set_label), set_label))
                if verbose and (i + 1) % 5000 == 0:
                    print(f"  ... {i + 1:,} / {len(files):,}")

        frame = pd.concat(parts, ignore_index=True)
        if verbose:
            print(f"Loaded {len(frame):,} rows / {frame['Patient ID'].nunique():,} patients.")
        if use_cache:
            frame.to_parquet(CACHE_PATH, index=False)
            if verbose:
                print(f"Cached to {CACHE_PATH}")
        _frame_memo = frame
        return _frame_memo

    _frame_memo = _load_excel_frame()
    return _frame_memo


# Index of patient ID -> (path on disk, set label), built by enumerating the dataset
# directories. Because every path here comes from glob() rather than from a client
# value, looking an ID up in this mapping cannot escape the dataset directories.
_psv_index_memo = None


def _psv_index():
    global _psv_index_memo
    if _psv_index_memo is None:
        index = {}
        for set_label, directory in resolve_psv_dirs():
            for path in glob.glob(os.path.join(directory, "*.psv")):
                index[_patient_id_from_path(path, set_label)] = (path, set_label)
        _psv_index_memo = index
    return _psv_index_memo


def list_patient_ids():
    """Patient IDs only. In 'full' mode this reads filenames, not file contents."""
    if DATASET_MODE == "full" and not (_frame_memo is not None or os.path.exists(CACHE_PATH)):
        ids = []
        for set_label, directory in resolve_psv_dirs():
            for path in sorted(glob.glob(os.path.join(directory, "*.psv"))):
                ids.append(_patient_id_from_path(path, set_label))
        return sorted(ids)
    return sorted(load_full_frame()["Patient ID"].unique().tolist())


def load_patient(patient_id):
    """
    Canonical frame for a single patient, sorted by ICU hour.

    In 'full' mode with no corpus loaded yet, this reads only that patient's
    file rather than pulling ~1.55M rows into memory.
    """
    # Reject anything that is not a well-formed ID before going further.
    if not is_valid_patient_id(patient_id):
        raise ValueError("Invalid patient ID.")

    if DATASET_MODE == "full" and _frame_memo is None and not os.path.exists(CACHE_PATH):
        # The path is never built from the supplied ID. It is looked up in an index
        # enumerated from the dataset directories, so every path this function can
        # open originates from the filesystem itself and the client value is only
        # ever used as a dictionary key.
        entry = _psv_index().get(patient_id)
        if entry is None:
            raise ValueError("Patient ID not found.")
        path, set_label = entry
        return _read_psv(path, patient_id, set_label).sort_values("ICULOS")

    frame = load_full_frame()
    rows = frame[frame["Patient ID"] == patient_id]
    if len(rows) == 0:
        raise ValueError("Patient ID not found in the dataset.")
    return rows.sort_values("ICULOS")


def describe_source():
    """Provenance metadata recorded into metrics_summary.json / run_config.json."""
    if DATASET_MODE == "full":
        return {
            "dataset_mode": "full",
            "sample_label": "Full PhysioNet/CinC 2019 corpus (training sets A+B)",
            "source": PSV_ROOT,
            "set_directories": [d for _, d in resolve_psv_dirs()],
        }
    return {
        "dataset_mode": "sample",
        "sample_label": "Sample Data (n=10,000 rows / 232 patients)",
        "source": EXCEL_PATH,
        "note": (
            "Bundled Excel extract of the PhysioNet/CinC 2019 challenge data. "
            "Set SEPSISGUARD_DATASET_MODE=full to train on the complete corpus."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(describe_source(), indent=2))
    frame = load_full_frame()
    print(f"\nrows      : {len(frame):,}")
    print(f"patients  : {frame['Patient ID'].nunique():,}")
    print(f"positives : {int(frame['SepsisLabel'].sum()):,} rows")
    print(f"ever-sepsis patients: {int(frame.groupby('Patient ID')['SepsisLabel'].max().sum()):,}")
