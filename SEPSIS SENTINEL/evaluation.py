import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from statistics import NormalDist
import sys

# Reconfigure stdout for UTF-8 output to support symbols
sys.stdout.reconfigure(encoding='utf-8')

# Portable Directory & Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data loading, feature definitions and mode switching all live in dataset.py.
# Re-exported here so existing imports from evaluation keep working.
from dataset import (
    features_info,
    feature_raws,
    feature_names,
    raw_to_clean,
    load_full_frame,
    describe_source,
    DATASET_MODE,
    EXCEL_PATH as excel_path,
)

# Result files are per-mode. Sample results keep the original unsuffixed names;
# full-corpus results land beside them as *_full.json. Without this, a full-corpus
# run would overwrite the sample results and the dashboard would serve full-corpus
# metrics alongside a sample-trained model.
_result_suffix = "" if DATASET_MODE == "sample" else f"_{DATASET_MODE}"
metrics_json_path = os.path.join(BASE_DIR, f"metrics_summary{_result_suffix}.json")
metrics_history_path = os.path.join(BASE_DIR, f"metrics_history{_result_suffix}.jsonl")
leakage_json_path = os.path.join(BASE_DIR, f"leakage_audit{_result_suffix}.json")
run_config_path = os.path.join(BASE_DIR, f"run_config{_result_suffix}.json")

# LightGBM hyperparameters. Built per-call so the model seed is always explicit
# and never inherited from a previous training run.
def make_lgb_params(seed):
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": seed
    }

# 1. Preprocessing and Imputation with Missingness Indicator Flags
def preprocess_and_impute(df_raw):
    """
    Takes a canonical frame from dataset.load_full_frame() (clean column names)
    and returns it imputed, with one binary <feature>_nan flag per feature.
    """
    print("Preprocessing and imputing data...")
    # Guarantee chronological order within each patient before forward-filling,
    # so the ffill can never pull a value backwards in time.
    df = df_raw.sort_values(["Patient ID", "ICULOS"], kind="stable").copy()

    # Calculate % missingness pre-imputation
    missingness = {}
    for name in feature_names:
        missingness[name] = {
            "percent_missing": float(df[name].isna().mean() * 100.0),
            "imputation_method": "patient-level forward-fill, then dataset-level median fallback"
        }

    # Missingness flags reflect the raw observation, before any filling
    flags = {f"{name}_nan": df[name].isna().astype("int8") for name in feature_names}

    # Forward-fill within patient timelines
    print("Performing forward-fill within patient timelines...")
    df_imputed = df.copy()
    df_imputed[feature_names] = df.groupby("Patient ID")[feature_names].ffill()

    # For any remaining gaps, fall back to the dataset-level median
    medians = {}
    for f in features_info:
        name = f["name"]
        med_val = float(df_imputed[name].median())
        if pd.isna(med_val):
            med_val = f["default"]
        medians[name] = med_val
        df_imputed[name] = df_imputed[name].fillna(med_val)

    # Attach all flag columns in one concat rather than 22 inserts
    df_imputed = pd.concat([df_imputed, pd.DataFrame(flags, index=df_imputed.index)], axis=1)

    return df_imputed, missingness, medians

# 2. Simplified Clinical Baselines
def compute_simplified_baselines(df):
    print("Computing simplified clinical baselines (qSOFA & NEWS2)...")
    
    # Fully vectorised: the previous df.iterrows() implementation cost minutes per
    # million rows. NaN contributes 0 points in every band, matching the original
    # `if pd.notna(...)` guards.
    def _band(series, bounds, points):
        """points[i] where series <= bounds[i]; final `points[-1]` is the else-branch."""
        conditions = [series <= b for b in bounds]
        return np.select(conditions, points[:-1], default=points[-1]) * series.notna().to_numpy()

    resp = df["Resp"]
    sbp = df["SBP"]
    o2 = df["O2Sat"]
    temp = df["Temp"]
    hr = df["HR"]

    # Simplified qSOFA (s-qSOFA)
    # RR >= 22 (1 pt), SBP <= 100 (1 pt)
    # GCS/Altered mental status is omitted
    df["s_qSOFA"] = (
        (resp >= 22).fillna(False).astype("int8") + (sbp <= 100).fillna(False).astype("int8")
    )

    # Simplified NEWS2
    # RR: <=8 (3), 9-11 (1), 12-20 (0), 21-24 (1), >=25 (3)
    # O2Sat: <=91 (3), 92-93 (2), 94-95 (1), >=96 (0)
    # Temp: <=35.0 (3), 35.1-36.0 (1), 36.1-38.0 (0), 38.1-39.0 (1), >=39.1 (2)
    # SBP: <=90 (3), 91-100 (2), 101-110 (1), 111-219 (0), >=220 (3)
    # HR: <=40 (3), 41-50 (1), 51-90 (0), 91-110 (1), 111-130 (2), >=131 (3)
    # AVPU (consciousness) and Supplemental Oxygen are omitted
    df["s_NEWS2"] = (
        _band(resp, [8, 11, 20, 24], [3, 1, 0, 1, 3])
        + _band(o2, [91, 93, 95], [3, 2, 1, 0])
        + _band(temp, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
        + _band(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
        + _band(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    ).astype("int16")

    return df

# 3. Label Leakage Audit
def leakage_audit(df_raw):
    print("Running Label Leakage Audit...")
    
    # Define SOFA-related features in our dataset (canonical names)
    sofa_features = ["Platelets", "Bilirubin_total", "Creatinine", "MAP", "Resp", "O2Sat"]
    target_col = "SepsisLabel"

    audit_results = {}

    # Calculate correlations at current time (t=0) and lagged time (t=-6 hours)
    # Sepsis Label is patient-hourly. Shift the target column backward to align with prior features.
    for col in sofa_features:
        clean_name = col

        # Current correlation (t=0)
        valid_idx_0 = df_raw[col].notna() & df_raw[target_col].notna()
        if valid_idx_0.sum() > 10:
            corr_0 = float(np.corrcoef(df_raw.loc[valid_idx_0, col], df_raw.loc[valid_idx_0, target_col])[0, 1])
        else:
            corr_0 = 0.0
            
        # Lagged correlation (t=-6): Sepsis label occurs 6 hours in the future
        # We shift Sepsis Label back by 6 hours per patient
        shifted_labels = df_raw.groupby("Patient ID")[target_col].shift(-6)
        valid_idx_lag = df_raw[col].notna() & shifted_labels.notna()
        if valid_idx_lag.sum() > 10:
            corr_lag = float(np.corrcoef(df_raw.loc[valid_idx_lag, col], shifted_labels[valid_idx_lag])[0, 1])
        else:
            corr_lag = 0.0
            
        # Leakage Risk evaluation
        # High risk: correlation is high (>0.4) and dramatically higher at t=0 than at t=-6
        corr_diff = abs(corr_0) - abs(corr_lag)
        if abs(corr_0) > 0.4 and corr_diff > 0.15:
            risk = "High"
        elif abs(corr_0) > 0.2:
            risk = "Medium"
        else:
            risk = "Low"
            
        audit_results[clean_name] = {
            "correlation_t0": corr_0,
            "correlation_t_minus_6": corr_lag,
            "correlation_difference": corr_diff,
            "leakage_risk_level": risk,
            "action_taken": "Retained with raw missingness flags. Imputed features lagged causally in replay engine."
        }
        
    print("\n--- Leakage Audit Summary ---")
    for feat, res in audit_results.items():
        print(f"Feature: {feat:<15} t0 Corr: {res['correlation_t0']:.3f} | t-6 Corr: {res['correlation_t_minus_6']:.3f} | Risk: {res['leakage_risk_level']}")
        
    # Write to file
    with open(leakage_json_path, "w") as f:
        json.dump(audit_results, f, indent=4)
        
    return audit_results

# 3b. Decision Threshold Selection
#
# The legacy 0.30 constant was inherited from the 232-patient sample, where sepsis
# prevalence was 7.14%. On the full corpus (1.80% prevalence) the model's probabilities
# sit much lower and a fixed 0.30 cut-off fires almost never — 5.1% sensitivity. Sepsis
# screening prioritises sensitivity, so thresholds are now *selected against explicit
# operating targets* rather than hardcoded.
#
# Selection happens on a validation split carved out of the training patients. Choosing
# a threshold on the test set would make the reported operating point optimistic.
TARGET_SENSITIVITY_ALERT = 0.85   # screening trigger: catch most septic hours
TARGET_SENSITIVITY_HIGH = 0.50    # escalation trigger: fewer, higher-precision alarms
LEGACY_THRESHOLD = 0.30           # kept only for before/after reporting


def calibration_bins(y_true, y_score, n_bins=10):
    """
    Reliability data: split predictions into equal-count deciles and compare the
    mean predicted probability against the observed sepsis rate in each bin.

    Equal-count (quantile) bins rather than equal-width, because at ~1.8%
    prevalence nearly every prediction falls in the lowest fixed-width bucket and
    the curve carries no information.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Rank-based split keeps bins populated even with heavily skewed scores.
    order = np.argsort(y_score, kind="stable")
    bins = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        bins.append({
            "count": int(len(chunk)),
            "mean_predicted": float(np.mean(y_score[chunk])),
            "observed_rate": float(np.mean(y_true[chunk])),
            "score_min": float(np.min(y_score[chunk])),
            "score_max": float(np.max(y_score[chunk])),
        })
    return {
        "n_bins": len(bins),
        "binning": "equal-count (quantile) over predicted probability",
        "bins": bins,
    }


def append_metrics_history(summary, path):
    """
    Append a compact timestamped snapshot of this run, so metric drift across runs
    is visible without diffing whole JSON files.
    """
    import datetime

    meta = summary.get("metadata", {})
    perf = summary.get("model_performance", {})
    snapshot = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_mode": meta.get("dataset_mode"),
        "total_rows": meta.get("total_rows"),
        "total_patients": meta.get("total_patients"),
        "split_seed": meta.get("split_seed"),
        "model_seed": meta.get("model_seed"),
        "auroc": perf.get("auroc"),
        "auprc": perf.get("auprc"),
        "brier_score": perf.get("brier_score"),
        "sensitivity_at_alert_threshold": perf.get("sensitivity_at_alert_threshold"),
        "specificity_at_alert_threshold": perf.get("specificity_at_alert_threshold"),
        "alert_threshold": (summary.get("decision_thresholds") or {}).get("alert_threshold"),
    }
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(snapshot) + "\n")
        print(f"Metrics history appended to {path}")
    except OSError as exc:
        print(f"Could not append metrics history: {exc}")


def operating_point(y_true, y_score, threshold):
    """Sensitivity / specificity / PPV / alert rate at a given threshold."""
    from sklearn.metrics import confusion_matrix
    y_hat = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "alert_rate": float((tp + fp) / len(y_true)) if len(y_true) else 0.0,
    }


def select_threshold_for_sensitivity(y_true, y_score, target_sensitivity):
    """
    Highest threshold that still achieves >= target_sensitivity.

    Taking the *highest* such threshold gives the best specificity among all
    thresholds meeting the sensitivity requirement.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sensitivity)[0]
    if len(ok) == 0:
        return float(np.min(y_score))
    # roc_curve returns thresholds in decreasing order; the first qualifying index
    # is therefore the highest threshold meeting the target.
    return float(thresholds[ok[0]])


def select_thresholds(y_true, y_score):
    """Pick alert/high thresholds plus Youden's J, all on validation data."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = float(thresholds[int(np.argmax(tpr - fpr))])
    alert = select_threshold_for_sensitivity(y_true, y_score, TARGET_SENSITIVITY_ALERT)
    high = select_threshold_for_sensitivity(y_true, y_score, TARGET_SENSITIVITY_HIGH)
    # Guarantee ordering even in degenerate cases
    if high <= alert:
        high = float(min(1.0, max(alert + 1e-6, youden)))
    return {
        "alert_threshold": alert,
        "high_risk_threshold": high,
        "youden_j_threshold": youden,
        "target_sensitivity_alert": TARGET_SENSITIVITY_ALERT,
        "target_sensitivity_high": TARGET_SENSITIVITY_HIGH,
        "selected_on": "validation split held out from training patients",
    }


# 3c. External Validation (train on one hospital, test on the other)
def external_validation(df_processed, features_columns, seed=42):
    """
    Train on Set A (Hospital A) and test on Set B (Hospital B), and vice versa.

    This is a stronger generalisation claim than a random patient-level split within
    one pooled corpus: the test hospital is never seen during training.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    results = {}
    for train_set, test_set in [("Set A", "Set B"), ("Set B", "Set A")]:
        tr = df_processed[df_processed["Dataset"] == train_set]
        te = df_processed[df_processed["Dataset"] == test_set]
        if len(tr) == 0 or len(te) == 0 or te["SepsisLabel"].nunique() < 2:
            continue

        model = lgb.train(
            make_lgb_params(seed),
            lgb.Dataset(tr[features_columns], label=tr["SepsisLabel"]),
            num_boost_round=150,
        )
        y_true = te["SepsisLabel"].values
        y_score = model.predict(te[features_columns])

        key = f"train_{train_set.replace(' ', '')}_test_{test_set.replace(' ', '')}"
        results[key] = {
            "train_set": train_set,
            "test_set": test_set,
            "train_patients": int(tr["Patient ID"].nunique()),
            "test_patients": int(te["Patient ID"].nunique()),
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "test_positive_rows": int(y_true.sum()),
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
            "brier_score": float(brier_score_loss(y_true, y_score)),
        }
        print(
            f"  {train_set} -> {test_set}: AUROC {results[key]['auroc']:.4f} "
            f"AUPRC {results[key]['auprc']:.4f} "
            f"({results[key]['train_patients']:,} train / {results[key]['test_patients']:,} test patients)"
        )
    return results


# 4. Stratified Patient-Level Split
def stratified_patient_split(df, seed):
    # Find patient labels: ever got sepsis (1) or never (0)
    patient_labels = df.groupby("Patient ID")["SepsisLabel"].max()
    # Force plain numpy arrays: with pyarrow installed, pandas 3 backs string
    # columns with ArrowStringArray, which sklearn's splitter cannot index.
    patients = np.asarray(patient_labels.index, dtype=object)
    labels = np.asarray(patient_labels, dtype=np.int64)
    
    # Stratified split on patient level
    from sklearn.model_selection import train_test_split
    train_pids, test_pids = train_test_split(
        patients, test_size=0.20, random_state=seed, stratify=labels
    )
    
    # Filter original df rows
    df_train = df[df["Patient ID"].isin(train_pids)]
    df_test = df[df["Patient ID"].isin(test_pids)]
    
    return df_train, df_test

# 5. Core Performance Evaluation & Paired Bootstrapping
def run_evaluation_pipeline():
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
    
    print("\n--- Starting Stratified Multi-Seed Evaluation ---")
    source = describe_source()
    print(f"Dataset mode: {source['dataset_mode']} ({source['sample_label']})")
    df_raw = load_full_frame()
    print(f"Loaded {len(df_raw):,} rows / {df_raw['Patient ID'].nunique():,} patients.")

    # Preprocess
    df_processed, missingness_log, medians = preprocess_and_impute(df_raw)
    df_processed = compute_simplified_baselines(df_processed)
    
    # Run Leakage Audit
    audit_log = leakage_audit(df_raw)
    
    # Multi-seed lists
    seeds = [42, 100, 2026, 999, 123]
    seed_results = []
    
    features_columns = feature_names + [f"{f}_nan" for f in feature_names]
    
    for seed in seeds:
        # Split
        df_train, df_test = stratified_patient_split(df_processed, seed)
        
        # Prepare arrays
        X_tr = df_train[features_columns]
        y_tr = df_train["SepsisLabel"]
        X_te = df_test[features_columns]
        y_te = df_test["SepsisLabel"]
        
        # Train LightGBM Model
        train_ds = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(make_lgb_params(seed), train_ds, num_boost_round=150)
        
        # Predictions
        y_pred = model.predict(X_te)
        
        # Metrics
        auroc = roc_auc_score(y_te, y_pred)
        auprc = average_precision_score(y_te, y_pred)
        brier = brier_score_loss(y_te, y_pred)
        
        # Sensitivity/Specificity at the legacy 0.30 constant. Retained across seeds
        # only so the threshold problem is visible in the seed sweep too; the headline
        # operating point comes from the selected alert threshold below.
        y_pred_class = (y_pred >= LEGACY_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred_class, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        seed_results.append({
            "seed": seed,
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "sensitivity_0.3": sens,
            "specificity_0.3": spec
        })
        
    # Summary across seeds. Named distinctly from the final metrics_summary built
    # further down: this dict used to share that name and was silently discarded
    # when the later assignment replaced it, losing the stability statistics.
    # It is now carried into metadata["multi_seed_stability"].
    seed_sweep_summary = {"seeds": list(seeds), "n_seeds": len(seeds)}
    for key in ["auroc", "auprc", "brier", "sensitivity_0.3", "specificity_0.3"]:
        vals = [r[key] for r in seed_results]
        seed_sweep_summary[f"{key}_mean"] = float(np.mean(vals))
        seed_sweep_summary[f"{key}_std"] = float(np.std(vals))

    print(f"Multi-Seed Results (n={len(seeds)}):")
    print(f"Mean AUROC: {seed_sweep_summary['auroc_mean']:.4f} ± {seed_sweep_summary['auroc_std']:.4f}")
    print(f"Mean AUPRC: {seed_sweep_summary['auprc_mean']:.4f} ± {seed_sweep_summary['auprc_std']:.4f}")
    
    # 6. Primary Seed Evaluation & Bootstrapping (for CIs and baselines)
    primary_seed = 42
    df_train, df_test = stratified_patient_split(df_processed, primary_seed)
    
    # Train primary model
    X_tr = df_train[features_columns]
    y_tr = df_train["SepsisLabel"]
    X_te = df_test[features_columns]
    y_te = df_test["SepsisLabel"]
    
    train_ds = lgb.Dataset(X_tr, label=y_tr)
    # Model seed is tied to the primary split seed rather than inherited from
    # the last multi-seed loop iteration.
    primary_lgb_params = make_lgb_params(primary_seed)
    primary_model = lgb.train(primary_lgb_params, train_ds, num_boost_round=150)
    
    # Predictions
    y_pred = primary_model.predict(X_te)
    
    # Baseline calculations on test split
    # NEWS2 score standard scaling (max NEWS2 = 20) is scaled down to [0, 1] for ROC calculation
    # qSOFA score (max qSOFA = 2) is scaled down to [0, 1]
    y_qsofa = df_test["s_qSOFA"].values / 2.0
    y_news2 = df_test["s_NEWS2"].values / 20.0
    
    # Primary point estimates
    model_auroc = roc_auc_score(y_te, y_pred)
    model_auprc = average_precision_score(y_te, y_pred)
    model_brier = brier_score_loss(y_te, y_pred)

    # --- Decision threshold selection -------------------------------------------
    # Carve a validation split out of the TRAINING patients, fit a model on the
    # reduced training set, and choose thresholds on validation predictions. The
    # test set is never used to pick the operating point.
    print("Selecting decision thresholds on a held-out validation split...")
    df_fit, df_val = stratified_patient_split(df_train, primary_seed)
    val_model = lgb.train(
        make_lgb_params(primary_seed),
        lgb.Dataset(df_fit[features_columns], label=df_fit["SepsisLabel"]),
        num_boost_round=150,
    )
    y_val = df_val["SepsisLabel"].values
    y_val_pred = val_model.predict(df_val[features_columns])
    thresholds = select_thresholds(y_val, y_val_pred)
    thresholds["validation_patients"] = int(df_val["Patient ID"].nunique())
    thresholds["validation_rows"] = int(len(df_val))
    print(
        f"  alert threshold    : {thresholds['alert_threshold']:.4f} "
        f"(target sensitivity {TARGET_SENSITIVITY_ALERT:.0%})"
    )
    print(
        f"  high-risk threshold: {thresholds['high_risk_threshold']:.4f} "
        f"(target sensitivity {TARGET_SENSITIVITY_HIGH:.0%})"
    )

    # Operating points on the TEST set at each threshold
    operating_points = {
        "alert": operating_point(y_te.values, y_pred, thresholds["alert_threshold"]),
        "high_risk": operating_point(y_te.values, y_pred, thresholds["high_risk_threshold"]),
        "youden_j": operating_point(y_te.values, y_pred, thresholds["youden_j_threshold"]),
        "legacy_0.30": operating_point(y_te.values, y_pred, LEGACY_THRESHOLD),
    }
    for name, op in operating_points.items():
        print(
            f"  [{name:11}] thr={op['threshold']:.4f} "
            f"sens={op['sensitivity']:.1%} spec={op['specificity']:.1%} "
            f"PPV={op['ppv']:.1%} alert_rate={op['alert_rate']:.1%}"
        )

    # Headline sensitivity/specificity now report the SELECTED alert threshold
    primary_sens = operating_points["alert"]["sensitivity"]
    primary_spec = operating_points["alert"]["specificity"]

    # --- External validation (hospital hold-out) ---------------------------------
    print("Running external validation (train on one hospital, test on the other)...")
    external = external_validation(df_processed, features_columns, seed=primary_seed)

    qsofa_auroc = roc_auc_score(y_te, y_qsofa)
    qsofa_auprc = average_precision_score(y_te, y_qsofa)
    
    news2_auroc = roc_auc_score(y_te, y_news2)
    news2_auprc = average_precision_score(y_te, y_news2)
    
    # Bootstrapping for 95% CIs (1,000 resamples)
    print("Running Paired Bootstrap Resampling (1,000 runs)...")
    np.random.seed(42)
    n_samples = len(y_te)
    y_te_arr = y_te.values
    
    boot_model_auroc = []
    boot_model_auprc = []
    boot_model_brier = []
    boot_qsofa_auroc = []
    boot_news2_auroc = []
    boot_diff_qsofa = []
    boot_diff_news2 = []
    
    for i in range(1000):
        # Draw paired bootstrap indices
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_te_b = y_te_arr[boot_idx]
        
        # Check class balance
        if len(np.unique(y_te_b)) < 2:
            continue
            
        y_pred_b = y_pred[boot_idx]
        y_qsofa_b = y_qsofa[boot_idx]
        y_news2_b = y_news2[boot_idx]
        
        # Model
        m_auc = roc_auc_score(y_te_b, y_pred_b)
        boot_model_auroc.append(m_auc)
        boot_model_auprc.append(average_precision_score(y_te_b, y_pred_b))
        boot_model_brier.append(brier_score_loss(y_te_b, y_pred_b))
        
        # Baselines
        q_auc = roc_auc_score(y_te_b, y_qsofa_b)
        n_auc = roc_auc_score(y_te_b, y_news2_b)
        boot_qsofa_auroc.append(q_auc)
        boot_news2_auroc.append(n_auc)
        
        # Differences (paired bootstrap)
        boot_diff_qsofa.append(m_auc - q_auc)
        boot_diff_news2.append(m_auc - n_auc)
        
    # Log a warning to show the paired bootstrap correction
    print("WARNING: Unpaired CI width must not be compared to paired CI width. Corrected paired bootstrap active.")
    
    # Compute CIs (2.5th and 97.5th percentiles)
    def get_ci(boot_list):
        return [float(np.percentile(boot_list, 2.5)), float(np.percentile(boot_list, 97.5))]
        
    ci_model_auroc = get_ci(boot_model_auroc)
    ci_model_auprc = get_ci(boot_model_auprc)
    ci_model_brier = get_ci(boot_model_brier)
    
    ci_qsofa_auroc = get_ci(boot_qsofa_auroc)
    ci_news2_auroc = get_ci(boot_news2_auroc)
    
    ci_diff_qsofa = get_ci(boot_diff_qsofa)
    ci_diff_news2 = get_ci(boot_diff_news2)
    
    # Calculate two-sided p-values for difference tests
    p_qsofa = float(2 * min(np.mean(np.array(boot_diff_qsofa) <= 0), np.mean(np.array(boot_diff_qsofa) >= 0)))
    p_news2 = float(2 * min(np.mean(np.array(boot_diff_news2) <= 0), np.mean(np.array(boot_diff_news2) >= 0)))
    
    # Subgroup Evaluation
    subgroups = {
        "Age_Under_65": df_test[df_test["Age"] < 65],
        "Age_Over_Equal_65": df_test[df_test["Age"] >= 65],
        "Sex_Female": df_test[df_test["Gender"] == 0],
        "Sex_Male": df_test[df_test["Gender"] == 1]
    }
    
    subgroup_metrics = {}
    raw_p_values = {
        "model_vs_qsofa": p_qsofa,
        "model_vs_news2": p_news2
    }
    
    for sub_name, sub_df in subgroups.items():
        sub_X = sub_df[features_columns]
        sub_y = sub_df["SepsisLabel"]
        
        n_sub = len(sub_df)
        pos_sub = int(sub_y.sum())
        underpowered = bool(pos_sub < 20)
        
        if len(np.unique(sub_y)) >= 2:
            sub_pred = primary_model.predict(sub_X)
            sub_auc = float(roc_auc_score(sub_y, sub_pred))
            sub_ap = float(average_precision_score(sub_y, sub_pred))
            
            # Simple bootstrap on subgroup to check significance
            boot_sub = []
            for _ in range(200):
                boot_i = np.random.choice(len(sub_y), len(sub_y), replace=True)
                if len(np.unique(sub_y.values[boot_i])) >= 2:
                    boot_sub.append(roc_auc_score(sub_y.values[boot_i], sub_pred[boot_i]))
            sub_ci = get_ci(boot_sub) if boot_sub else [0.0, 0.0]
            
            # Compare vs qSOFA
            sub_q = sub_df["s_qSOFA"].values / 2.0
            sub_q_auc = float(roc_auc_score(sub_y, sub_q))
            
            # Compute a quick p-value for the difference in this subgroup
            sub_diff = []
            for _ in range(200):
                boot_i = np.random.choice(len(sub_y), len(sub_y), replace=True)
                if len(np.unique(sub_y.values[boot_i])) >= 2:
                    sub_diff.append(roc_auc_score(sub_y.values[boot_i], sub_pred[boot_i]) - roc_auc_score(sub_y.values[boot_i], sub_q[boot_i]))
            sub_p = float(2 * min(np.mean(np.array(sub_diff) <= 0), np.mean(np.array(sub_diff) >= 0))) if sub_diff else 1.0
            
            raw_p_values[f"subgroup_{sub_name}_vs_qsofa"] = sub_p
        else:
            sub_auc = np.nan
            sub_ap = np.nan
            sub_ci = [np.nan, np.nan]
            sub_p = 1.0
            
        subgroup_metrics[sub_name] = {
            "N": n_sub,
            "positive_cases": pos_sub,
            "underpowered": underpowered,
            "auroc": sub_auc,
            "auprc": sub_ap,
            "auroc_95_ci": sub_ci,
            "raw_p_value_vs_qsofa": sub_p
        }
        
    # 7. Multiple Comparisons Correction (Holm-Bonferroni)
    p_keys = list(raw_p_values.keys())
    p_vals = [raw_p_values[k] for k in p_keys]
    
    # Sort indices
    sorted_indices = np.argsort(p_vals)
    m = len(p_vals)
    adjusted_p_vals = [1.0] * m
    
    for i, idx in enumerate(sorted_indices):
        raw_p = p_vals[idx]
        factor = m - i
        adj_p = min(raw_p * factor, 1.0)
        # Holm step-down requirement: adj_p[i] = max(adj_p[i], adj_p[i-1])
        if i > 0:
            adj_p = max(adj_p, adjusted_p_vals[sorted_indices[i-1]])
        adjusted_p_vals[idx] = float(adj_p)
        
    adjusted_p_dict = dict(zip(p_keys, adjusted_p_vals))
    
    # Add corrections to subgroups
    for sub_name in subgroup_metrics.keys():
        subgroup_metrics[sub_name]["adjusted_p_value_vs_qsofa"] = adjusted_p_dict[f"subgroup_{sub_name}_vs_qsofa"]
        
    # 8. SHAP Verification
    # Fit SHAP on test data
    print("Running SHAP clinical verification...")
    # Calculate feature importances based on LightGBM's pred_contrib
    test_contribs = primary_model.predict(X_te, pred_contrib=True)
    # Mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(test_contribs[:, :-1]), axis=0)
    shap_ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    top_5_shap = [item[0] for item in shap_ranking[:5]]
    
    # Cohen's d ranking from workbook: Lactate, O2Sat, Creatinine, Bilirubin_total, WBC
    cohens_d_top_5 = ["Lactate", "O2Sat", "Creatinine", "Bilirubin_total", "WBC"]
    agreement = [feat for feat in top_5_shap if feat in cohens_d_top_5]
    disagreement = [feat for feat in cohens_d_top_5 if feat not in top_5_shap]
    
    # SHAP Stability Check
    print("Running SHAP stability checks...")
    np.random.seed(42)
    sample_size = min(50, len(X_te))
    sample_indices = np.random.choice(len(X_te), size=sample_size, replace=False)
    
    correlations = []
    # Loop over sample patients and add 1% std noise
    feature_stds = X_te.std()
    for idx in sample_indices:
        x_orig = X_te.iloc[[idx]]
        # Compute original SHAP
        shap_orig = primary_model.predict(x_orig, pred_contrib=True)[0, :-1]
        
        # Add 1% std noise
        noise = np.random.normal(0, 0.01 * feature_stds)
        x_perturbed = x_orig + noise
        
        # Recompute SHAP
        shap_perturbed = primary_model.predict(x_perturbed, pred_contrib=True)[0, :-1]
        
        # Calculate Spearman correlation
        r_val, _ = spearmanr(shap_orig, shap_perturbed)
        if not np.isnan(r_val):
            correlations.append(r_val)
            
    shap_stability_score = float(np.mean(correlations)) if correlations else 1.0
    print(f"SHAP Stability Score (Mean Spearman rank-corr): {shap_stability_score:.4f}")
    
    # Build final metrics dict
    metrics_summary = {
        "metadata": {
            **source,
            "total_rows": int(len(df_raw)),
            "total_patients": int(df_raw["Patient ID"].nunique()),
            "ever_sepsis_patients": int(df_raw.groupby("Patient ID")["SepsisLabel"].max().sum()),
            "warning": "Unpaired CI width compared against paired CI width is invalid. Paired bootstrap active.",
            "simplified_qsofa_threshold": "Simplified qSOFA >= 2",
            "simplified_qsofa_citation": "Seymour et al. 2016 (JAMA)",
            "simplified_news2_threshold": "Simplified NEWS2 >= 5",
            "simplified_news2_citation": "Royal College of Physicians 2017 NEWS2",
            "primary_seed": primary_seed,
            "split_seed": primary_seed,
            "model_seed": primary_lgb_params["seed"],
            "train_size": len(df_train),
            "test_size": len(df_test),
            "multi_seed_stability": seed_sweep_summary
        },
        "model_performance": {
            "auroc": float(model_auroc),
            "auroc_95_ci": ci_model_auroc,
            "auprc": float(model_auprc),
            "auprc_95_ci": ci_model_auprc,
            "brier_score": float(model_brier),
            "brier_95_ci": ci_model_brier,
            "sensitivity_at_alert_threshold": primary_sens,
            "specificity_at_alert_threshold": primary_spec
        },
        "baseline_comparison": {
            "qsofa": {
                "auroc": float(qsofa_auroc),
                "auroc_95_ci": ci_qsofa_auroc,
                "auprc": float(qsofa_auprc),
                "auroc_difference_raw_p_value": raw_p_values["model_vs_qsofa"],
                "auroc_difference_adjusted_p_value": adjusted_p_dict["model_vs_qsofa"],
                "auroc_difference_95_ci": ci_diff_qsofa
            },
            "news2": {
                "auroc": float(news2_auroc),
                "auroc_95_ci": ci_news2_auroc,
                "auprc": float(news2_auprc),
                "auroc_difference_raw_p_value": raw_p_values["model_vs_news2"],
                "auroc_difference_adjusted_p_value": adjusted_p_dict["model_vs_news2"],
                "auroc_difference_95_ci": ci_diff_news2
            }
        },
        "decision_thresholds": thresholds,
        "operating_points": operating_points,
        "calibration": calibration_bins(y_te.values, y_pred, n_bins=10),
        "external_validation": external,
        "subgroups": subgroup_metrics,
        "xai_verification": {
            "shap_top_5": top_5_shap,
            "cohens_d_top_5": cohens_d_top_5,
            "plausibility_agreement": agreement,
            "plausibility_disagreement": disagreement,
            "shap_stability_score": shap_stability_score
        }
    }
    
    # Save to file
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to {metrics_json_path}")
    append_metrics_history(metrics_summary, metrics_history_path)
    
    # 9. Reproducibility Configuration
    run_config = {
        **source,
        "train_patients_count": int(df_train["Patient ID"].nunique()),
        "test_patients_count": int(df_test["Patient ID"].nunique()),
        "train_rows_count": len(df_train),
        "test_rows_count": len(df_test),
        "random_seeds_evaluated": seeds,
        "feature_list": features_columns,
        "missingness_rates": missingness_log,
        "imputation_strategy": "Patient-level forward-fill, followed by median fallback, with missingness-indicator flags",
        "split_seed": primary_seed,
        "model_seed": primary_lgb_params["seed"],
        "hyperparameters": primary_lgb_params,
        # Runtime reads these instead of hardcoding 0.30 / 0.70
        "decision_thresholds": thresholds,
    }
    
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=4)
    print(f"Run config saved to {run_config_path}")
    
    return metrics_summary, audit_log

if __name__ == "__main__":
    run_evaluation_pipeline()
