# DVC + MLflow Machine Learning Experiment

> **A Beginner-Friendly Guide to Data, Code, Model Tracking & Reproducibility in MLOps**

---

## 1. Aim
The objective of this experiment is to build a fully reproducible Machine Learning pipeline using **DVC** (Data Version Control) and **MLflow** (Experiment Tracking). This ensures that data, code, model hyperparameters, evaluation metrics, and saved model artifacts are automatically tracked and can be reproduced by anyone on any machine.

---

## 2. What This Experiment Does
This experiment demonstrates the complete end-to-end MLOps lifecycle:
1. **Loads and inspects** any CSV dataset (`data/dataset.csv`).
2. **Handles missing data** automatically and separates features and target.
3. **Trains a Machine Learning Model** (Random Forest Classifier or Regressor) based on dataset target type.
4. **Tracks hyperparameter experiments** (`n_estimators = 10, 50, 100, 200`) and evaluation metrics using **MLflow**.
5. **Saves the trained model** as `models/model.pkl` using `joblib`.
6. **Tracks dataset versions** using **DVC**.
7. **Automates pipeline execution** using **DVC Pipeline** (`dvc repro`).
8. **Versions code and pipeline metadata** using **Git** and pushes to **GitHub**.

### End-to-End Workflow Diagram
```
CSV Dataset
    ↓
   DVC (Dataset Tracking)
    ↓
Training Script (src/train.py)
    ↓
Machine Learning Model (Random Forest)
    ↓
MLflow Tracking (Params & Metrics)
    ↓
Accuracy / Evaluation Metrics
    ↓
Saved Model (models/model.pkl)
    ↓
DVC Pipeline (dvc.yaml -> dvc repro)
    ↓
Git (Code & Metadata Versioning)
    ↓
GitHub (Remote Repository)
```

---

## 3. Tools Used

| Tool | Purpose | Simple Explanation |
| :--- | :--- | :--- |
| **Python 3.10+** | Programming Language | The language used to write our data processing and training scripts. |
| **Git** | Code Version Control | Tracks changes made to source code scripts and configuration files over time. |
| **GitHub** | Remote Code Hosting | Online platform to back up Git repositories and collaborate with others. |
| **DVC (Data Version Control)** | Dataset & Pipeline Versioning | Tracks large dataset files and manages pipeline dependencies that Git cannot efficiently handle. |
| **MLflow** | Experiment Tracking UI | Logs model hyperparameters, metrics (Accuracy, MAE, R²), and artifacts to compare different training runs. |
| **pandas** | Data Analysis | Loads CSV files into tabular DataFrames to inspect, clean, and manipulate data. |
| **numpy** | Numerical Computing | Provides fast array computations and mathematical operations. |
| **scikit-learn** | Machine Learning Library | Provides Random Forest algorithms, data splitters, and evaluation metric functions. |
| **joblib** | Model Serialization | Saves trained machine learning models to disk (`.pkl` file) so they can be reloaded later for prediction. |

---

## 4. Input

- **Input File:** `data/dataset.csv`
- **Features ($X$):** All columns in the CSV except the target column (e.g., `age`, `experience`, `salary`, `score`). Non-numeric categorical columns are automatically converted to numbers using One-Hot Encoding.
- **Target ($y$):** The column you want to predict (e.g., `target`).
  - *Automatic Detection:* If `TARGET_COLUMN = None` in `src/train.py`, the script automatically picks the **last column** of your CSV as the target.
  - *Manual Override:* You can set `TARGET_COLUMN = "your_column_name"` in `src/train.py`.

---

## 5. Output

1. **Trained Model Artifact:** Saved to `models/model.pkl`.
2. **MLflow Tracking Records:** Created inside the `mlruns/` directory (accessible via `mlflow ui` at `http://127.0.0.1:5000`).
3. **Experiment Metrics:** 
   - *Classification:* Accuracy, Precision, Recall, F1-Score.
   - *Regression:* MAE (Mean Absolute Error), MSE, RMSE, $R^2$ Score.
4. **DVC Tracking Files:** `data/dataset.csv.dvc` and `dvc.lock`.
5. **DVC Pipeline File:** `dvc.yaml`.
6. **Git Commits:** Version control history of code, pipeline files, and DVC metadata.

---

## 6. Project Structure

```
mlops-1/
│
├── data/
│   └── dataset.csv          # Main input CSV dataset (Tracked by DVC)
│
├── src/
│   └── train.py             # ML training & MLflow logging script
│
├── models/
│   ├── .gitkeep             # Ensures folder exists in Git
│   └── model.pkl            # Trained model artifact saved by joblib
│
├── create_dataset.py        # Helper script to create a sample CSV dataset
├── dvc.yaml                 # DVC pipeline stage definition file
├── requirements.txt         # Python package dependencies
├── .gitignore               # Files ignored by Git (venv, mlruns, dataset.csv)
└── README.md                # Comprehensive documentation
```

### Explanation of Key Files
- `data/dataset.csv`: Your raw data file. Git ignores this file; DVC tracks it.
- `src/train.py`: Contains the logic to load data, clean missing values, split data, train Random Forest, log to MLflow, and export `model.pkl`.
- `create_dataset.py`: Optional utility script that generates a sample CSV if you do not have one ready yet.
- `dvc.yaml`: Tells DVC how the dataset (`data/dataset.csv`) connects to the training code (`src/train.py`) and output model (`models/model.pkl`).

---

## 7. Prerequisites

Before starting, ensure you have installed on your Windows machine:
1. **Python 3.10+**: Download from [python.org](https://www.python.org/downloads/) (Make sure to check *"Add Python to PATH"* during installation).
2. **Git**: Download from [git-scm.com](https://git-scm.com/download/win).
3. **VS Code**: Recommended code editor.

Check your installations in Windows PowerShell:
```powershell
python --version
git --version
```

---

## 8. Create Virtual Environment

A virtual environment isolates project dependencies so they do not interfere with system Python.

Open **Windows PowerShell** inside your project folder and run:

```powershell
# Step 1: Create a virtual environment named 'venv'
python -m venv venv

# Step 2: Activate the virtual environment
venv\Scripts\activate
```
*(Meaning: `python -m venv venv` creates a clean isolated Python folder named `venv`. `venv\Scripts\activate` switches your PowerShell session to use Python and libraries inside `venv`.)*

> **Note:** When activated, your prompt will show `(venv)` at the beginning.

---

## 9. Install Libraries

Install all required Machine Learning, MLflow, and DVC packages:

```powershell
# Install required libraries
pip install pandas numpy scikit-learn mlflow dvc joblib

# Save installed dependencies to requirements.txt
pip freeze > requirements.txt
```
*(Meaning: Installs the exact libraries needed for ML, dataset tracking, and experiment logging, and saves their versions into `requirements.txt`.)*

To install dependencies later on a new machine:
```powershell
pip install -r requirements.txt
```

---

## 10. Initialize Git

Initialize a Git repository to version control your code:

```powershell
# Initialize Git in project directory
git init

# Add initial files to Git staging
git add .

# Create initial commit
git commit -m "Initial ML project setup"
```
*(Meaning: `git init` initializes a hidden `.git` repository folder. `git add .` prepares all code files to be saved. `git commit` takes a snapshot of your project code.)*

---

## 11. Initialize DVC

Initialize DVC (Data Version Control) inside the project:

```powershell
# Initialize DVC
dvc init

# Commit DVC configuration to Git
git add .dvc .gitignore
git commit -m "Initialize DVC"
```
*(Meaning: `dvc init` creates a `.dvc` configuration directory to track dataset versions alongside Git.)*

---

## 12. Add Dataset

Place your CSV dataset inside the `data/` directory (name it `dataset.csv`).

If you don't have a CSV file yet, create a sample dataset by running:
```powershell
python create_dataset.py
```

Now track the dataset using DVC:
```powershell
# Tell DVC to track dataset.csv
dvc add data/dataset.csv

# Stage the created .dvc tracking file and updated .gitignore in Git
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Track dataset using DVC"
```
*(Meaning: `dvc add data/dataset.csv` creates a tiny pointer file `data/dataset.csv.dvc` containing the hash/checksum of your dataset. Git tracks the small `.dvc` file while ignoring the large raw CSV file.)*

---

## 13. Configure Training Script

Open [`src/train.py`](file:///C:/Users/Lenovo/OneDrive/Desktop/Git%20Hub/mlops-1/src/train.py) in VS Code. At the top of the file, you will find clearly marked configuration variables:

```python
# ==============================================================================
# CONFIGURATION VARIABLES (Easy for Beginners to Change!)
# ==============================================================================
DATA_PATH = "data/dataset.csv"
TARGET_COLUMN = None     # Set to your column name, e.g. "target", or leave None for auto-detect
N_ESTIMATORS = 100       # Hyperparameter: Number of trees in Random Forest (Try 10, 50, 100, 200)
RANDOM_STATE = 42        # Random seed
# ==============================================================================
```

- **Target Column:** If your CSV target column is named `target`, you can set `TARGET_COLUMN = "target"`. If set to `None`, it automatically selects the **last column** in `dataset.csv`.
- **Hyperparameter:** You can change `N_ESTIMATORS` to test different values (`10`, `50`, `100`, `200`).

---

## 14. Run Training

Execute the training script from your activated terminal:

```powershell
python src/train.py
```

### Expected Command Output:
```text
============================================================
STEP 1: INSPECTING DATASET STRUCTURE
============================================================
Number of Rows: 200
Number of Columns: 5
Column Names: ['age', 'experience', 'salary', 'score', 'target']

Data Types:
age             int64
experience      int64
salary        float64
score         float64
target          int64
dtype: object

Missing Values Per Column:
age           0
experience    0
salary        1
score         1
target        0
dtype: int64
------------------------------------------------------------
Handling missing values...
  Filled missing numeric values in 'salary' with median (95830.0)
  Filled missing numeric values in 'score' with median (71.95)
Training set size: 160 samples
Testing set size:  40 samples

============================================================
STEP 2: TRAINING MODEL (CLASSIFICATION)
============================================================
Model: Random Forest (Classification)
Hyperparameter - n_estimators: 100

Evaluation Metrics:
  Accuracy:  0.9500
  Precision: 0.9531
  Recall:    0.9500
  F1 Score:  0.9497

Model saved successfully at: 'models\model.pkl'
Logged parameters, metrics, and model artifact to MLflow!
============================================================
```

---

## 15. Start MLflow UI

Launch the MLflow interactive web dashboard:

```powershell
mlflow ui
```
*(Meaning: Starts a local MLflow web server listening on port 5000.)*

Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

> ⚠️ **IMPORTANT:** Leave this terminal window running so the MLflow UI remains active. Open a new terminal window in VS Code for running further commands!

---

## 16. Run Multiple Hyperparameter Experiments

To demonstrate experiment tracking in MLflow, run multiple experiments with different values of `n_estimators`:

1. Open `src/train.py`.
2. Change `N_ESTIMATORS = 10`. Save the file and run in terminal:
   ```powershell
   python src/train.py
   ```
3. Change `N_ESTIMATORS = 50`. Save the file and run in terminal:
   ```powershell
   python src/train.py
   ```
4. Change `N_ESTIMATORS = 100`. Save the file and run in terminal:
   ```powershell
   python src/train.py
   ```
5. Change `N_ESTIMATORS = 200`. Save the file and run in terminal:
   ```powershell
   python src/train.py
   ```

Now refresh your MLflow UI (`http://127.0.0.1:5000`). You will see **4 separate runs** listed under `MLOps_Reproducibility_Experiment`, allowing you to compare accuracy and metrics across different hyperparameter values!

---

## 17. DVC Pipeline (`dvc repro`)

A DVC Pipeline links data, code, and model outputs so that the experiment can be reproduced with a single command.

The pipeline is defined in [`dvc.yaml`](file:///C:/Users/Lenovo/OneDrive/Desktop/Git%20Hub/mlops-1/dvc.yaml):
```yaml
stages:
  train:
    cmd: python src/train.py
    deps:
      - data/dataset.csv
      - src/train.py
    outs:
      - models/model.pkl
```

To run/reproduce the full pipeline:
```powershell
dvc repro
```

### What `dvc repro` does in simple words:
1. Checks if `data/dataset.csv` or `src/train.py` has changed.
2. If changes are detected, it automatically executes `python src/train.py` to regenerate `models/model.pkl`.
3. If no files changed, DVC skips execution because the pipeline outputs are already up to date!

---

## 18. Commit Pipeline Status to Git

After running experiments and generating your DVC pipeline lock file:

```powershell
# Check Git status
git status

# Stage updated code and DVC files
git add dvc.yaml dvc.lock src/train.py

# Commit snapshot
git commit -m "Complete MLflow experiments and DVC pipeline setup"
```

---

## 19. Push to GitHub

To store your project on GitHub:

1. Go to [GitHub.com](https://github.com/) and create a new empty repository named `ml-dvc-mlflow-project`. (Do **not** initialize with README or license).
2. Copy your GitHub repository URL (e.g., `https://github.com/YOUR_USERNAME/ml-dvc-mlflow-project.git`).
3. Run the following commands in PowerShell (replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual URL):

```powershell
# Add remote GitHub repository link
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>

# Rename current branch to main
git branch -M main

# Push project code to GitHub
git push -u origin main
```
*(Example: `git remote add origin https://github.com/john-doe/ml-dvc-mlflow-project.git`)*

---

## 20. Complete Run Order (Step-by-Step Cheat Sheet)

Follow this exact sequence from start to finish:

```powershell
# 1. Open VS Code terminal in project folder: C:\Users\Lenovo\OneDrive\Desktop\Git Hub\mlops-1

# 2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy scikit-learn mlflow dvc joblib

# 4. Save requirements
pip freeze > requirements.txt

# 5. Initialize Git and DVC
git init
dvc init

# 6. Generate sample dataset (or copy your own dataset.csv into data/ folder)
python create_dataset.py

# 7. Track dataset with DVC
dvc add data/dataset.csv
git add data/dataset.csv.dvc data/.gitignore .dvc .gitignore
git commit -m "Initialize project and track dataset with DVC"

# 8. Run initial training script
python src/train.py

# 9. Start MLflow server (leave running in terminal)
mlflow ui
# (Open http://127.0.0.1:5000 in browser)

# 10. Run hyperparameter experiments (In a NEW terminal window with venv activated)
# Change N_ESTIMATORS in src/train.py to 10, 50, 100, 200 and run:
python src/train.py

# 11. Run reproducible DVC pipeline
dvc repro

# 12. Save final progress to Git
git add .
git commit -m "Completed reproducible ML experiment with DVC and MLflow"

# 13. Push code to GitHub (Replace URL with your own)
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git branch -M main
git push -u origin main
```

---

## 21. Expected Output

### Example Terminal Training Output:
```text
============================================================
STEP 1: INSPECTING DATASET STRUCTURE
============================================================
Number of Rows: 200
Number of Columns: 5
Column Names: ['age', 'experience', 'salary', 'score', 'target']

Data Types:
...
Missing Values Per Column:
...
============================================================
STEP 2: TRAINING MODEL (CLASSIFICATION)
============================================================
Model: Random Forest (Classification)
Hyperparameter - n_estimators: 100

Evaluation Metrics:
  Accuracy:  0.9500
  Precision: 0.9531
  Recall:    0.9500
  F1 Score:  0.9497

Model saved successfully at: 'models\model.pkl'
Logged parameters, metrics, and model artifact to MLflow!
```

### Example MLflow Comparison Table (`http://127.0.0.1:5000`):

| Run Name | `problem_type` | `n_estimators` | `accuracy` / Metric | Saved Model Artifact |
| :--- | :--- | :--- | :--- | :--- |
| Run 1 | Classification | 10 | 0.9000 (XX.XX%) | model.pkl |
| Run 2 | Classification | 50 | 0.9250 (XX.XX%) | model.pkl |
| Run 3 | Classification | 100 | 0.9500 (XX.XX%) | model.pkl |
| Run 4 | Classification | 200 | 0.9500 (XX.XX%) | model.pkl |

*(Note: Exact numerical accuracy values will depend on your specific dataset values.)*

---

## 22. How to Compare Experiments in MLflow

1. Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.
2. Select **`MLOps_Reproducibility_Experiment`** from the left sidebar.
3. Check the boxes next to all 4 runs in the table.
4. Click the **Compare** button at the top of the table.
5. MLflow will display:
   - **Parallel Coordinates Plot / Bar Charts** comparing `n_estimators` vs `accuracy`.
   - **Side-by-side metric comparison** to highlight which hyperparameter achieved the highest accuracy without manually checking code logs!

---

## 23. Git vs DVC vs MLflow Summary

| Tool | Primary Purpose | What Files It Manages | Where Data Is Stored |
| :--- | :--- | :--- | :--- |
| **Git** | Code Version Control | `.py` scripts, `dvc.yaml`, `README.md`, `.gitignore` | Local `.git` folder & GitHub cloud |
| **DVC** | Data & Model Version Control | `data/dataset.csv`, `models/model.pkl`, `.dvc` tracking files | Local `.dvc/cache` or DVC remote storage |
| **MLflow** | Experiment Tracking & Model Registry | Hyperparameters, metrics, run timestamps, logs | Local `mlruns/` folder or MLflow server |

---

## 24. Reproducibility

### What is Reproducibility in Machine Learning?
Reproducibility means that any engineer or researcher can run your project on a completely different machine months later and achieve the **exact same model training results**.

### How This Project Guarantees Full Reproducibility:
1. **Git** tracks exact code versions (`src/train.py`).
2. **DVC** tracks exact dataset versions (`data/dataset.csv.dvc`).
3. **`dvc repro`** automates pipeline execution in the exact dependency order.
4. **MLflow** logs exact hyperparameters (`n_estimators = 100`) and metric outputs.
5. **Random Seed (`RANDOM_STATE = 42`)** ensures deterministic data splits and model training.

---

## 25. Common Errors & Easy Troubleshooting Solutions

### 1. `'python' is not recognized as an internal or external command`
- **Cause:** Python is not added to Windows PATH environment variables.
- **Fix:** Reinstall Python from python.org and ensure you check the box: **"Add Python to PATH"**.

### 2. `venv\Scripts\activate` fails with script execution policy error
- **Cause:** Windows PowerShell restricts running unverified scripts.
- **Fix:** Run PowerShell as Administrator and execute:
  `Set-ExecutionPolicy Unrestricted -Scope Process` then activate again.

### 3. `'dvc' or 'mlflow' is not recognized`
- **Cause:** Virtual environment is not activated in your terminal.
- **Fix:** Run `venv\Scripts\activate` first before calling `dvc` or `mlflow`.

### 4. `FileNotFoundError: Dataset not found at 'data/dataset.csv'`
- **Cause:** The CSV file is missing from the `data/` folder.
- **Fix:** Place your CSV file inside `data/` and name it `dataset.csv`, or run `python create_dataset.py`.

### 5. Target Column Not Found Error
- **Cause:** `TARGET_COLUMN` in `src/train.py` specifies a column name that does not exist in your CSV.
- **Fix:** Set `TARGET_COLUMN = None` in `src/train.py` to auto-detect the last column, or update `TARGET_COLUMN` to match your exact CSV header.

### 6. `Port 5000 is already in use` (MLflow UI Error)
- **Cause:** Another process or previous MLflow instance is occupying port 5000.
- **Fix:** Run MLflow on a different port:
  `mlflow ui --port 5001` then open `http://127.0.0.1:5001`.

### 7. Git Remote Origin Error (`remote origin already exists`)
- **Cause:** Git remote origin was added previously.
- **Fix:** Remove and re-add:
  `git remote remove origin`
  `git remote add origin <YOUR_GITHUB_REPOSITORY_URL>`

### 8. `dvc add` error: `data/dataset.csv is ignored by git`
- **Cause:** DVC automatically updates `.gitignore` to prevent raw data from being committed to Git.
- **Fix:** This is normal behavior! Run `git add data/dataset.csv.dvc .gitignore` to stage the DVC tracker instead.

---

## 26. Viva Questions & Answers

### Q1: What is DVC and why is it needed in MLOps?
**Answer:** DVC (Data Version Control) is an open-source tool designed to version control datasets, machine learning models, and pipelines. It is needed because Git cannot handle large data files efficiently.

### Q2: What is the difference between Git and DVC?
**Answer:** Git tracks source code files (`.py`, `.md`, `.yaml`), whereas DVC tracks large datasets and model binaries (`.csv`, `.pkl`), storing lightweight `.dvc` pointer files in Git.

### Q3: What is MLflow and what components does it have?
**Answer:** MLflow is an open-source platform to manage the ML lifecycle. Its key components include MLflow Tracking (logging parameters & metrics), MLflow Models (saving model artifacts), and MLflow Registry.

### Q4: What is an Experiment Run in MLflow?
**Answer:** An experiment run is a single execution of machine learning code where parameters (e.g. `n_estimators=100`), metrics (e.g. `accuracy=0.95`), and outputs (`model.pkl`) are recorded.

### Q5: What is `n_estimators` in Random Forest?
**Answer:** `n_estimators` is a hyperparameter that defines the total number of decision trees built in the Random Forest ensemble model.

### Q6: What does the command `dvc repro` do?
**Answer:** `dvc repro` executes the pipeline defined in `dvc.yaml`. It checks if inputs or code changed and re-runs only the necessary stages to ensure reproducible results.

### Q7: What is the role of `joblib` in this project?
**Answer:** `joblib` serializes the trained scikit-learn model object into a binary `.pkl` file on disk (`models/model.pkl`), allowing it to be saved and loaded later for prediction.

### Q8: How does MLflow UI help in selecting the best model?
**Answer:** The MLflow UI provides side-by-side metric comparison tables and visual plots (Parallel Coordinates) so users can compare multiple hyperparameter runs and immediately spot the model with highest accuracy.

### Q9: Why do we use `random_state = 42` in model training?
**Answer:** `random_state` sets a fixed random seed for dataset splitting and algorithm initialization, guaranteeing identical results across different runs for complete reproducibility.

### Q10: What is a `.dvc` file?
**Answer:** A `.dvc` file (e.g., `dataset.csv.dvc`) is a small text tracking file created by DVC that stores the MD5 hash and metadata of the dataset so Git can track dataset versions without storing the actual raw data.

---

## 27. Final Result Statement

> *"Successfully implemented a reproducible end-to-end Machine Learning pipeline using DVC for dataset tracking, MLflow for hyperparameter experiment logging, scikit-learn for Random Forest training, and Git/GitHub for code version control. All model hyperparameters, evaluation metrics, and model artifacts were recorded and verified across multiple experiment runs."*
