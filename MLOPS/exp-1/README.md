# Iris MLOps Experiment

A small DVC-ready machine learning project for training an Iris classifier.

## Structure

- `data/iris.csv`: input dataset
- `src/train.py`: training script
- `models/`: generated model artifacts
- `mlruns/`: experiment tracking artifacts
- `dvc.yaml`: reproducible pipeline definition
- `params.yaml`: training parameters

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

The trained model is written to `models/iris_model.joblib`.
