# SEPSIS-SENTINEL
# Sepsis Sentinel: Clinical ICU Risk Prediction Engine

## Overview
A clinical-grade machine learning pipeline designed to predict the onset of Sepsis in Intensive Care Unit (ICU) patients. This system processes multivariate time-series data and outperforms standard clinical baselines (qSOFA/NEWS2) by leveraging gradient boosting and advanced feature interpretability to provide actionable insights to healthcare providers.

## Core Architecture
* **Core Language:** Python
* **Modeling Framework:** LightGBM
* **Interpretability:** SHAP (SHapley Additive exPLanations)
* **Data Processing:** Pandas, NumPy

## Technical Implementations
1. **Multivariate Time-Series Processing:** Engineered a data pipeline to process 22 distinct clinical vitals and laboratory results in continuous time-series formats.
2. **Missing Data Engineering:** Handled inherently sparse clinical data by implementing explicit missingness flags, preventing data leakage while allowing the model to extract predictive value from the *absence* of clinical measurements.
3. **Clinical Interpretability:** Integrated SHAP to output feature attributions. Instead of acting as a "black box," the model explicitly quantifies which specific vitals (e.g., dropping blood pressure, spiking temperature) are driving the Sepsis risk score.
4. **Real-Time Replay Engine:** Designed a causal-only simulated inference engine to validate model performance, ensuring the architecture can handle true real-time clinical deployment conditions without future-data leakage.

## Project Structure
Contains the core Python notebooks/scripts for data preprocessing, model training (patient-level stratified splitting), and SHAP visualization outputs.
