# GridSense AI — 24-Hour Grid Stress Early Warning for ERCOT

> **ADS 508 · MLOps for Data Science · University of San Diego**  
> Alexander Zhuk · Michael Ha · Mark Villanueva

---

## Overview

GridSense AI is a machine learning pipeline built on Amazon SageMaker that predicts ERCOT grid stress events 24 hours in advance. A grid stress event is defined as any hour in which ERCOT electricity demand meets or exceeds the 90th-percentile historical threshold of **68,308 MW** — the top 10% highest-demand hours across 2022–2024, corresponding to the periods most likely to exceed renewable supply capacity.

When a stress event occurs without advance warning, grid operators are forced to purchase emergency backup power at real-time spot prices that can reach **$5,000/MWh**, compared to typical day-ahead prices of **$30–60/MWh**. A 24-hour early warning allows operators to pre-commit backup generation capacity at day-ahead prices, directly reducing emergency procurement costs.

---

## Model Results

Trained on 26 weather and temporal features using SageMaker XGBoost 1.7-1 with `seed=42`. Evaluated on a held-out test set spanning August–December 2024 (3,285 hours, 8.4% stress rate).

| Metric | Value | Target | Status |
|---|---|---|---|
| Average Precision (AP) | 0.8831 | — | — |
| AUC-ROC | 0.9859 | — | — |
| Train AUCPR | 0.9973 | — | — |
| Validation AUCPR | 0.8880 | — | — |
| Precision (threshold = 0.35) | 0.7762 | ≥ 0.70 | ✅ PASS |
| Recall (threshold = 0.35) | 0.7790 | ≥ 0.70 | ✅ PASS |
| F1 Score (threshold = 0.35) | 0.7776 | — | — |
| Precision (threshold = 0.50) | 0.8276 | ≥ 0.70 | ✅ PASS |
| Recall (threshold = 0.50) | 0.6957 | ≥ 0.70 | — |

**Recommended operating threshold: 0.35** — catches 78% of all real stress events (215 of 276 true positives) with approximately 12 false alarms per month.

**Top features by XGBoost gain:** temperature (840.0), month_cos (316.5), hour_cos (157.8), ghi_baseline_30d (108.1), dhi (86.8), hour_sin (61.9)

---

## Data Sources

| Source | Provider | Records | Description |
|---|---|---|---|
| EIA Electric Grid Monitor | U.S. Energy Information Administration | 26,304 (ERCOT) | Hourly electricity demand and generation by fuel type |
| Open-Meteo Historical Weather | Open-Meteo | 26,304 | Temperature, wind speed, cloud cover, solar radiation |
| NREL NSRDB Solar Radiation | National Renewable Energy Laboratory | 26,280 | GHI, DNI, DHI, and cloud type classification |

All data covers **January 1, 2022 – December 31, 2024**, geographically aligned to Austin, TX (30.3°N, −97.7°W) within the ERCOT footprint. Raw files are stored in the S3 bucket `gridsense-ai-data-team1`.

---

## Repository Structure

```
ADS-508-Final-Project/
│
├── 01_eia_exploration.ipynb          # EIA data quality checks, distributions, visualizations
├── 02_openmeteo_exploration.ipynb    # Weather data exploration and outlier analysis
├── 03_nrel_exploration.ipynb         # Solar radiation data exploration
├── 04_data_preparation.ipynb         # Merging, cleaning, feature engineering, SMOTE, train/val/test split
├── 05_model_training.ipynb           # SageMaker XGBoost training, evaluation, threshold analysis
│
└── README.md
```

---

## Pipeline

```
EIA / Open-Meteo / NREL APIs
        │
        ▼
  Python ingestion scripts
        │
        ▼
  S3: gridsense-ai-data-team1
  (raw/ → prepared/ → gridsense/models/)
        │
        ▼
  SageMaker Studio (Notebooks 01–04)
  · Merge on UTC timestamp
  · Engineer 34 features → remove 8 leakage columns → 26 final features
  · SMOTE on training split only (stress rate: 8.7% → ~30%)
  · Chronological split: 75% train / 12.5% val / 12.5% test
        │
        ▼
  SageMaker XGBoost 1.7-1 (Notebook 05)
  · ml.m5.xlarge · 4 vCPU / 16 GB
  · eval_metric: aucpr · seed=42
        │
        ▼
  SageMaker Serverless Inference Endpoint
  · Input: region ID + 26-feature weather forecast payload
  · Output: stress probability (0–1)
  · Alert threshold: 0.35
```

---

## Feature Engineering

The final model uses 26 weather and temporal features. Eight features were identified and removed as data leakage sources during training (they directly encode the target condition `demand ≥ 68,308 MW`):

**Removed (leakage):** `demand`, `net_gen`, `solar_gen`, `wind_gen`, `nuclear_gen`, `hydro_gen`, `shortfall_ratio`, `demand_temp_interaction`

**Key engineered features retained:**
- `ghi_baseline_30d` — 30-day rolling GHI baseline; measures sustained solar shortfall
- `ghi_deviation` — hourly GHI deviation from seasonal baseline
- `wind_variance_6h` — 6-hour rolling standard deviation of wind speed at 100m; captures sudden wind drops
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos` — cyclical encodings preserving temporal continuity
- Cloud type one-hot dummies (`cloud_type_0` through `cloud_type_9`)

AP dropped from 1.0 → 0.8831 after leakage removal, confirming that earlier performance was inflated rather than genuinely predictive.

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `binary:logistic` | Binary classification, outputs P(stress) |
| `num_round` | 300 | Max boosting rounds; early stopping prevents overfitting |
| `max_depth` | 6 | Captures multi-variable interactions without overfitting |
| `eta` | 0.1 | Conservative learning rate; gradual convergence |
| `subsample` | 0.8 | Row subsampling; stochastic regularisation |
| `colsample_bytree` | 0.8 | Column subsampling; improves ensemble diversity |
| `min_child_weight` | 5 | Prevents splits on fewer than 5 instances |
| `scale_pos_weight` | ~2.33 | Computed neg/pos ratio; up-weights stress event misclassifications |
| `eval_metric` | `aucpr` | Sensitive to minority-class performance on imbalanced data |
| `early_stopping_rounds` | 20 | Halts if validation AUCPR does not improve for 20 rounds |
| `seed` | 42 | Global random seed; guarantees full reproducibility |

---

## Setup and Reproduction

### Prerequisites
- AWS account with SageMaker Studio access
- S3 bucket: `gridsense-ai-data-team1` (or substitute your own)
- NREL developer API key ([register free](https://developer.nrel.gov/signup/))
- Python 3.8+, `boto3`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`

### Steps

**1. Ingest raw data**
- Download EIA Electric Grid Monitor 6-month CSV files (2022–2024) from the [EIA bulk download page](https://www.eia.gov/electricity/gridmonitor) and upload to `s3://gridsense-ai-data-team1/eia/`
- Run the Open-Meteo ingestion script to retrieve hourly weather data for Austin, TX and upload to `s3://gridsense-ai-data-team1/openmeteo/`
- Run the NREL ingestion script (requires API key) and upload to `s3://gridsense-ai-data-team1/nrel/`

**2. Data preparation**  
Open `04_data_preparation.ipynb` in SageMaker Studio and run all cells. Outputs `train.csv`, `validation.csv`, and `test.csv` to `s3://gridsense-ai-data-team1/prepared/`.

**3. Model training**  
Open `05_model_training.ipynb` and run all cells. Launches a SageMaker XGBoost training job on `ml.m5.xlarge`, evaluates on the test set, and saves the model artifact to `s3://gridsense-ai-data-team1/gridsense/models/`.

**4. Deploy endpoint**  
Using the saved model artifact, create a SageMaker Serverless Inference Endpoint. Send a JSON payload containing the 26 model features in the same order and Min-Max scale used during training. Apply threshold 0.35 to the returned probability score.

---

## Project Goals

| Goal | Status |
|---|---|
| Ingest EIA, Open-Meteo, and NREL into a structured S3 data lake | ✅ |
| Engineer feature set capturing renewable supply-demand imbalance signals | ✅ |
| Train XGBoost classifier with precision ≥ 0.70 and recall ≥ 0.70 | ✅ |
| Deploy SageMaker Serverless Inference Endpoint | ✅ |
| Complete all runs within $50 AWS student account budget | ✅ |

---

## Known Limitations

- **Geographic scope:** The model is trained on ERCOT data aligned to Austin, TX. It should not be applied to other grid operators without retraining on region-specific data.
- **Temporal staleness:** NREL solar data is available only through 2024. As ERCOT's installed renewable capacity grows, model performance may drift; quarterly retraining is recommended.
- **Same-hour features:** The current model classifies stress events using same-hour weather conditions. Introducing 24-hour lag features would convert it into a true day-ahead forecasting system — identified as the highest-priority future enhancement.
- **Batch pipeline:** There is no real-time streaming layer. A production deployment would benefit from AWS Kinesis + SageMaker Feature Store for continuous hourly inference.

---

## References

Potomac Economics. (2025). *2024 state of the market report for ERCOT*. https://www.potomaceconomics.com/wp-content/uploads/2025/06/2024-State-of-the-Market-Report.pdf

National Renewable Energy Laboratory. (n.d.). *NREL National Solar Radiation Database (NSRDB)*. https://developer.nrel.gov/docs/solar/nsrdb/

Open-Meteo. (n.d.). *Open-Meteo historical weather API*. https://open-meteo.com/en/docs/historical-weather-api

U.S. Energy Information Administration. (n.d.). *EIA electric grid monitor*. https://www.eia.gov/electricity/gridmonitor

---

## Authors

**Alexander Zhuk · Michael Ha · Mark Villanueva**  
ADS 508 — MLOps for Data Science · University of San Diego · 2026  
Company: GridSense AI (Energy Technology)
