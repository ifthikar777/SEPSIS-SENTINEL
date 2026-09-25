import time
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# Portable Directory & Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
latency_log_path = os.path.join(BASE_DIR, "latency_log.jsonl")

# Feature definitions and dataset access
from dataset import features_info, feature_names, raw_to_clean, load_patient, pseudonymize
import alerting

class ReplayEngine:
    def __init__(self, patient_id, playback_speed=10, model=None, train_medians=None,
                 alert_threshold=0.30, high_risk_threshold=0.70):
        self.patient_id = patient_id
        self.playback_speed = playback_speed
        self.model = model
        # Supplied by the caller from run_config's selected thresholds; the defaults
        # are only a fallback for a config predating threshold selection.
        self.alert_threshold = float(alert_threshold)
        self.high_risk_threshold = float(high_risk_threshold)
        # Incremental form of the shared suppression rule (see alerting.py).
        self.alert_state = alerting.AlertState(self.alert_threshold, self.high_risk_threshold)
        self.train_medians = train_medians or {f["name"]: f["default"] for f in features_info}
        
        # Load this patient's history only. In "full" mode this reads a single
        # .psv file rather than pulling the whole corpus into memory.
        # Log a stable pseudonym, never the record identifier itself.
        self.patient_pseudonym = pseudonymize(patient_id)
        print(f"ReplayEngine: Loading history for {self.patient_pseudonym}...")
        self.patient_rows = load_patient(patient_id)

        self.total_hours = len(self.patient_rows)
        self.current_step = 0
        
        # Causal State
        self.max_timestamp_seen = -1.0
        
        # Rolling Imputation State
        self.rolling_values = {}  # Holds last known values (forward-fill)
        
        # Rolling Alert Window: store recent predictions
        # Buffer of dictionaries: { "hour": h, "risk": r }
        self.risk_history = []
        
        # Feature columns matching evaluation.py
        self.feature_columns = feature_names + [f"{f}_nan" for f in feature_names]
        
    def get_interval(self):
        # 1x = 5s, 10x = 1s, 60x = 0.1s
        if self.playback_speed == 1:
            return 5.0
        elif self.playback_speed == 10:
            return 1.0
        elif self.playback_speed == 60:
            return 0.1
        else:
            return max(0.01, 5.0 / self.playback_speed)
            
    def step(self):
        if self.current_step >= self.total_hours:
            return None  # Finished
            
        start_time_ms = time.perf_counter()
        
        # Extract row
        row = self.patient_rows.iloc[self.current_step]
        row_hour = float(row["ICULOS"])
        
        # Causal-only assertion check: ensure no future timestamps are seen
        if self.max_timestamp_seen >= 0 and row_hour < self.max_timestamp_seen:
            raise RuntimeError(f"Causal Violation: Replay hour {row_hour} is before max seen hour {self.max_timestamp_seen}")
        self.max_timestamp_seen = row_hour
        
        # Incremental Rolling Imputation
        imputed_features = {}
        missing_flags = {}
        
        for f in features_info:
            clean_name = f["name"]
            raw_val = row[clean_name]

            if pd.notna(raw_val):
                # Update rolling value (forward-fill)
                self.rolling_values[clean_name] = float(raw_val)
                imputed_features[clean_name] = float(raw_val)
                missing_flags[f"{clean_name}_nan"] = 0
            else:
                # Value is missing in current hour's reading
                missing_flags[f"{clean_name}_nan"] = 1
                if clean_name in self.rolling_values:
                    # Forward-fill from previous hours
                    imputed_features[clean_name] = self.rolling_values[clean_name]
                else:
                    # Fallback to train median
                    imputed_features[clean_name] = self.train_medians[clean_name]
                    
        # Package feature vector
        feature_vector = {}
        for col in feature_names:
            feature_vector[col] = imputed_features[col]
            feature_vector[f"{col}_nan"] = missing_flags[f"{col}_nan"]
            
        # Run Model Inference
        df_pred = pd.DataFrame([feature_vector])
        prob = float(self.model.predict(df_pred)[0])
        
        # Run SHAP attribution
        contrib = self.model.predict(df_pred, pred_contrib=True)[0]
        shap_attribs = {}
        for name, val in zip(feature_names, contrib[:-1]):
            shap_attribs[name] = float(val)
            
        # Decision Certainty: measures how far the risk probability is from the 0.5 decision boundary
        # (0% = coin-flip, 100% = fully decisive). This is NOT a statistical confidence interval
        # and is unrelated to the bootstrapped 95% CIs reported in evaluation.py.
        decision_certainty = 2.0 * abs(prob - 0.5)
        
        # Data Completeness: percentage of features with raw measurements (missing flag == 0)
        data_completeness = float(sum(1 for val in missing_flags.values() if val == 0) / len(missing_flags)) if missing_flags else 1.0
        
        if prob >= self.high_risk_threshold:
            risk_level = "High"
        elif prob >= self.alert_threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # Alarm Fatigue & Alert Suppression Logic.
        # The rule itself lives in alerting.py, shared with app.py's batch replay
        # scoring, so the two can never drift apart.
        alert_status, raw_alert, is_sustained, severe_organ = self.alert_state.step(
            probability=prob,
            lactate=imputed_features.get("Lactate"),
            o2sat=imputed_features.get("O2Sat"),
        )

        # Retained for the payload's rolling trace of recent risk.
        self.risk_history.append({"hour": row_hour, "risk": prob})
        if len(self.risk_history) > 4:
            self.risk_history.pop(0)  # Keep 4-hour window
                
        # Calculate latency
        end_time_ms = time.perf_counter()
        latency_ms = (end_time_ms - start_time_ms) * 1000.0
        
        # Log latency
        log_entry = {
            "timestamp": time.time(),
            "patient_pseudonym": self.patient_pseudonym,
            "hour": row_hour,
            "latency_ms": latency_ms
        }
        with open(latency_log_path, "a") as lf:
            lf.write(json.dumps(log_entry) + "\n")
            
        # Increment step
        self.current_step += 1
        
        # Construct message payload
        return {
            "type": "live_reading",
            "hour": row_hour,
            "step": self.current_step - 1,
            "total_steps": self.total_hours,
            "features": imputed_features,
            "missing_flags": missing_flags,
            "probability": prob,
            "decision_certainty": decision_certainty,
            "data_completeness": data_completeness,
            "risk_level": risk_level,
            "feature_contribs": shap_attribs,
            "alert_status": alert_status,
            "sepsis_label": int(row["SepsisLabel"]),
            "latency_ms": latency_ms
        }
