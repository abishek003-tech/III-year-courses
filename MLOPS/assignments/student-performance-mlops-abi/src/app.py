"""
src/app.py
==========
FastAPI prediction API for the Student Performance system.

Endpoints
---------
  GET  /             Health check
  POST /predict      Predict G3 for a single student
  GET  /model-info   Show which model is loaded and metrics

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predict import predict_single, SAMPLE_STUDENT
from src.config import METRICS_JSON


# ── Pydantic request schema ──────────────────────────────────────────────────
class StudentInput(BaseModel):
    school: str     = Field(..., example="GP",     description="Student's school: GP or MS")
    sex: str        = Field(..., example="F",      description="Student's sex: F or M")
    age: int        = Field(..., example=17,       description="Student's age (15-22)")
    address: str    = Field(..., example="U",      description="Home address type: U (urban) or R (rural)")
    famsize: str    = Field(..., example="GT3",    description="Family size: GT3 or LE3")
    Pstatus: str    = Field(..., example="A",      description="Parent's cohabitation: T or A")
    Medu: int       = Field(..., example=4,        description="Mother's education (0-4)")
    Fedu: int       = Field(..., example=4,        description="Father's education (0-4)")
    Mjob: str       = Field(..., example="teacher",description="Mother's job")
    Fjob: str       = Field(..., example="health", description="Father's job")
    reason: str     = Field(..., example="course", description="Reason to choose school")
    guardian: str   = Field(..., example="mother", description="Student's guardian")
    traveltime: int = Field(..., example=2,        description="Travel time to school (1-4)")
    studytime: int  = Field(..., example=2,        description="Weekly study time (1-4)")
    failures: int   = Field(..., example=0,        description="Number of past class failures (0-4)")
    schoolsup: str  = Field(..., example="yes",    description="Extra educational support: yes/no")
    famsup: str     = Field(..., example="yes",    description="Family educational support: yes/no")
    paid: str       = Field(..., example="no",     description="Extra paid classes: yes/no")
    activities: str = Field(..., example="no",     description="Extra-curricular activities: yes/no")
    nursery: str    = Field(..., example="yes",    description="Attended nursery school: yes/no")
    higher: str     = Field(..., example="yes",    description="Wants higher education: yes/no")
    internet: str   = Field(..., example="yes",    description="Internet access at home: yes/no")
    romantic: str   = Field(..., example="no",     description="In a romantic relationship: yes/no")
    famrel: int     = Field(..., example=4,        description="Quality of family relationships (1-5)")
    freetime: int   = Field(..., example=3,        description="Free time after school (1-5)")
    goout: int      = Field(..., example=4,        description="Going out with friends (1-5)")
    Dalc: int       = Field(..., example=1,        description="Workday alcohol consumption (1-5)")
    Walc: int       = Field(..., example=1,        description="Weekend alcohol consumption (1-5)")
    health: int     = Field(..., example=3,        description="Current health status (1-5)")
    absences: int   = Field(..., example=4,        description="Number of school absences (0-93)")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "Predict a student's final grade (G3) based on demographic, "
        "social, and academic features.  Built by Abi (Abhishek S.) "
        "as a college MLOps assignment."
    ),
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def health_check():
    """API health check."""
    return {
        "status": "running",
        "message": "Student Performance Prediction API is live.",
        "author": "Abi (Abhishek S.)",
    }


@app.post("/predict", tags=["Prediction"])
def predict(student: StudentInput):
    """Predict the final grade G3 for a single student."""
    data   = student.model_dump()
    result = predict_single(data)

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["messages"])

    return {
        "predicted_G3": result["prediction"],
        "grade_label":  result["grade_label"],
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """Show the currently loaded best model and its metrics."""
    if not METRICS_JSON.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Run training first.")
    with open(METRICS_JSON) as fh:
        return json.load(fh)


@app.get("/sample", tags=["Info"])
def sample_input():
    """Return a valid sample student input for testing."""
    return {"sample_student": SAMPLE_STUDENT}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting prediction API at http://127.0.0.1:8000")
    print("[INFO] Interactive docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
