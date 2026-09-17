"""
src/train.py - Main Machine Learning Training Script
---------------------------------------------------
This script loads the dataset from `data/dataset.csv`, inspects its structure,
handles missing values, trains a Random Forest model, logs metrics & parameters 
to MLflow, and saves the trained model to `models/model.pkl`.
"""

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# ==============================================================================
# CONFIGURATION VARIABLES (Easy for Beginners to Change!)
# ==============================================================================
# Path to your input dataset CSV file
DATA_PATH = "data/dataset.csv"

# Target column name. 
# Set to None if you want the script to automatically pick the LAST column in your CSV.
# Or set it explicitly, e.g., TARGET_COLUMN = "target"
TARGET_COLUMN = None  

# Hyperparameter to experiment with (Try changing to 10, 50, 100, 200!)
N_ESTIMATORS = 100

# Random seed for reproducible splits and training
RANDOM_STATE = 42

# Path where trained model will be saved
MODEL_PATH = os.path.join("models", "model.pkl")
# ==============================================================================


def inspect_and_clean_data(df, target_col):
    """Prints dataset structure, summary info, and handles basic missing values."""
    print("=" * 60)
    print("STEP 1: INSPECTING DATASET STRUCTURE")
    print("=" * 60)
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Column Names: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    
    missing = df.isnull().sum()
    print("\nMissing Values Per Column:")
    print(missing)
    print("-" * 60)

    # Handle missing values if any exist
    if missing.sum() > 0:
        print("Handling missing values...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Fill numeric missing values with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"  Filled missing numeric values in '{col}' with median ({median_val})")
                else:
                    # Fill categorical missing values with mode
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    print(f"  Filled missing categorical values in '{col}' with mode ('{mode_val}')")

    return df


def detect_target_column(df, user_target_col):
    """Identifies the target column automatically if not explicitly provided."""
    if user_target_col and user_target_col in df.columns:
        return user_target_col
    
    # Fallback to the last column in the DataFrame
    auto_target = df.columns[-1]
    print(f"[INFO] Using automatically detected target column: '{auto_target}'")
    return auto_target


def preprocess_features(X):
    """Encodes non-numeric feature columns (one-hot encoding) if present."""
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"[INFO] Converting categorical columns to numeric: {list(non_numeric_cols)}")
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
    return X


def run_training():
    print("\nStarting ML Pipeline Training...")
    
    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Please put your CSV file in data/dataset.csv "
            "or run `python create_dataset.py` to create a sample dataset."
        )

    df = pd.read_csv(DATA_PATH)
    
    # 2. Target Column Identification
    target_col = detect_target_column(df, TARGET_COLUMN)
    
    # 3. Inspect and Clean Data
    df = inspect_and_clean_data(df, target_col)
    
    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert string targets to numeric if applicable
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        print(f"[INFO] Converting target column '{target_col}' categories to numbers.")
        y = pd.factorize(y)[0]

    # Preprocess non-numeric feature columns
    X = preprocess_features(X)

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # 6. Determine Problem Type (Classification vs Regression)
    is_classification = False
    unique_targets = len(np.unique(y))
    if unique_targets <= 20 or y.dtype == 'int64' or y.dtype == 'int32':
        is_classification = True
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    print("\n" + "=" * 60)
    print(f"STEP 2: TRAINING MODEL ({problem_type.upper()})")
    print("=" * 60)
    print(f"Model: Random Forest ({problem_type})")
    print(f"Hyperparameter - n_estimators: {N_ESTIMATORS}")

    # Set up MLflow Tracking Experiment
    mlflow.set_experiment("MLOps_Reproducibility_Experiment")

    with mlflow.start_run():
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS, 
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate Classification Metrics
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

            print("\nEvaluation Metrics:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")

            # Log to MLflow
            mlflow.log_param("problem_type", "Classification")
            mlflow.log_param("n_estimators", N_ESTIMATORS)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("target_column", target_col)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

        else:
            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS, 
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate Regression Metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            print("\nEvaluation Metrics:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  MSE:  {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")

            # Log to MLflow
            mlflow.log_param("problem_type", "Regression")
            mlflow.log_param("n_estimators", N_ESTIMATORS)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("target_column", target_col)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

        # 7. Save Model with joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"\nModel saved successfully at: '{MODEL_PATH}'")

        # Log Model artifact in MLflow (with trusted tree types for sklearn Random Forest)
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Logged parameters, metrics, and model artifact to MLflow!")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    run_training()
