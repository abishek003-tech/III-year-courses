# 📊 Student Performance Prediction System — MLOps

> **Build a Reproducible and Automated Student Performance Prediction System**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![Prefect](https://img.shields.io/badge/Prefect-Orchestration-purple.svg)](https://prefect.io)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green.svg)](https://dvc.org)

---

## 👤 Author

| Field | Details |
|---|---|
| **Name** | Abi (Abhishek S.) |
| **GitHub** | [abishek003-tech](https://github.com/abishek003-tech) |
| **College** | KGiSL Institute of Technology |
| **Department** | Artificial Intelligence and Machine Learning |
| **Course** | Machine Learning Operations (MLOps) |

---

## 📌 Problem Statement

Predict a student's **final academic grade (G3)** using demographic, social, and academic features — while excluding intermediate grades (G1, G2) to prevent target leakage and simulate a realistic early-prediction scenario.

---

## 📁 Project Structure

```
student-performance-mlops-abi/
├── configs/
│   └── params.yaml                # All configurable parameters
├── data/
│   └── student-mat.csv            # UCI dataset (auto-downloaded)
├── models/
│   ├── best_model.pkl             # Trained best model
│   └── preprocessor.pkl           # Fitted preprocessing pipeline
├── monitoring/
│   └── monitor.py                 # Model monitoring checks
├── pipeline/
│   └── flow.py                    # Prefect-orchestrated ML pipeline
├── reports/
│   ├── metrics.json               # Model comparison (JSON)
│   ├── model_comparison.csv       # Model comparison (CSV)
│   ├── monitoring_report.json     # Monitoring output
│   └── charts/
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── residuals_best_model.png
│       └── error_distribution.png
├── src/
│   ├── config.py                  # Central configuration loader
│   ├── data_ingestion.py          # Dataset download & extraction
│   ├── data_validation.py         # Data quality checks
│   ├── preprocessing.py           # Feature engineering & encoding
│   ├── train.py                   # Model training & MLflow logging
│   ├── evaluate.py                # Performance chart generation
│   ├── predict.py                 # Prediction module with validation
│   └── app.py                     # FastAPI prediction API
├── tests/
│   └── test_pipeline.py           # Automated pytest suite
├── screenshots/                   # Evidence for assignment
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**UCI Student Performance Dataset**  
P. Cortez and A. Silva (2008). *Using Data Mining to Predict Secondary School Student Performance.*

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **File used:** `student-mat.csv` (Mathematics course)
- **Records:** 395 students
- **Features:** 30 attributes (demographic, social, academic)
- **Target:** `G3` — final grade (0–20 scale)
- **Leakage prevention:** `G1` (1st period grade) and `G2` (2nd period grade) are excluded from features

### Feature Categories

| Type | Count | Examples |
|---|---|---|
| Numerical | 13 | age, Medu, Fedu, studytime, failures, absences |
| Categorical | 17 | school, sex, address, Mjob, Fjob, higher, internet |

---

## 🏗️ Architecture

```
Dataset Download ─→ Data Validation ─→ Preprocessing ─→ Feature Prep
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
Model Training (3 models) ─→ Evaluation ─→ Best Selection ─→ MLflow Logging
                                                                   │
     ┌─────────────────────────────────────────────────────────────┘
     ▼
Model Saving ─→ Prediction Service ─→ Monitoring
```

**Orchestration:** Prefect `@flow` + `@task`  
**Tracking:** MLflow experiments (parameters, metrics, model artifacts)  
**Versioning:** DVC for dataset and model files  

---

## 🧠 Models Trained

| # | Model | Key Hyperparameters |
|---|---|---|
| 1 | Linear Regression | fit_intercept=True |
| 2 | Random Forest Regressor | n_estimators=200, max_depth=10 |
| 3 | Gradient Boosting Regressor | n_estimators=200, lr=0.05, max_depth=4 |

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error magnitude |
| **MSE** | Mean Squared Error — penalises large errors more |
| **RMSE** | Root MSE — same unit as target (grade points) |
| **R²** | Coefficient of determination — proportion of variance explained |

### Selection Criterion

The model with the **lowest RMSE** is automatically selected as the best model.

> **Performance results table will be populated after running the pipeline with real data.**  
> See `reports/metrics.json` and `reports/model_comparison.csv` for actual values.

---

## 🚀 Installation & Setup (Windows)

### Prerequisites
- Python 3.10+ installed
- Git installed
- Internet connection (for first dataset download)

### Step-by-step

```powershell
# 1. Navigate to the project folder
cd "c:\Users\Lenovo\OneDrive\Desktop\III year\MLOPS\assignments\student-performance-mlops-abi"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Option 1: Full Automated Pipeline (Recommended)

```powershell
python pipeline/flow.py
```

This single command executes:
1. ✅ Dataset download & extraction
2. ✅ Data quality validation
3. ✅ Feature preprocessing
4. ✅ Training of all 3 models
5. ✅ Metric evaluation & comparison
6. ✅ MLflow experiment logging
7. ✅ Chart generation
8. ✅ Prediction demo
9. ✅ Model monitoring

### Option 2: Run Individual Steps

```powershell
# Download dataset
python src/data_ingestion.py

# Validate data
python src/data_validation.py

# Preprocess and train (includes MLflow logging)
python src/train.py

# Run a prediction demo
python src/predict.py

# Run monitoring
python monitoring/monitor.py
```

---

## 🔬 MLflow Experiment Tracking

```powershell
# Launch MLflow UI (after running the pipeline)
mlflow ui --backend-store-uri mlruns

# Open in browser: http://127.0.0.1:5000
```

MLflow logs:
- Model hyperparameters
- MAE, MSE, RMSE, R² for each model
- Trained model artifacts
- Run IDs for reproducibility

---

## 🌐 Prediction API

```powershell
# Start the FastAPI server
python src/app.py

# Open interactive docs: http://127.0.0.1:8000/docs
```

**Endpoints:**
| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict G3 for a student |
| GET | `/model-info` | View loaded model metrics |
| GET | `/sample` | Get a sample input |

---

## 📦 DVC Setup (Optional)

```powershell
# Initialize DVC
dvc init

# Track the dataset
dvc add data/student-mat.csv

# Commit the DVC file to Git
git add data/student-mat.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"
```

---

## 🧪 Running Tests

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📈 Output Files

After running the pipeline, these files are generated:

| File | Description |
|---|---|
| `models/best_model.pkl` | Trained best model (joblib) |
| `models/preprocessor.pkl` | Fitted ColumnTransformer |
| `reports/metrics.json` | All model metrics (JSON) |
| `reports/model_comparison.csv` | Comparison table (CSV) |
| `reports/monitoring_report.json` | Monitoring check results |
| `reports/charts/model_comparison.png` | Bar chart comparison |
| `reports/charts/actual_vs_predicted.png` | Scatter plots |
| `reports/charts/residuals_best_model.png` | Residual analysis |
| `reports/charts/error_distribution.png` | Error histogram |
| `mlruns/` | MLflow experiment tracking data |

---

## 🛡️ Security & Responsible MLOps

- No passwords, tokens, or API keys in the repository
- No real student PII used (UCI dataset is anonymised)
- Dataset is publicly licensed for academic research
- `.gitignore` excludes sensitive and generated files
- Model predictions are informational, not authoritative

---

## ⚠️ Limitations

1. **Small dataset** — 395 records limits model generalization
2. **Without G1/G2** — prediction accuracy is inherently lower
3. **No deep learning** — linear and ensemble methods only
4. **Single subject** — trained on Mathematics data only
5. **No real-time retraining** — batch pipeline execution
6. **Local monitoring** — no cloud-based alerting

---

## 🔮 Future Enhancements

- [ ] Add cross-validation for more robust evaluation
- [ ] Include the Portuguese language dataset for multi-subject analysis
- [ ] Deploy the FastAPI to a cloud platform (Heroku, GCP, AWS)
- [ ] Implement GitHub Actions CI/CD pipeline
- [ ] Add Evidently AI for advanced drift monitoring
- [ ] Hyperparameter tuning with Optuna
- [ ] Web-based dashboard (Streamlit or Gradio)
- [ ] SHAP feature importance analysis

---

## 📜 License

This project is developed for academic purposes as part of the MLOps course at KGiSL Institute of Technology.

The UCI Student Performance Dataset is used under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

*Built with ❤️ by Abi (Abhishek S.) — KGiSL Institute of Technology, AI & ML Department*
