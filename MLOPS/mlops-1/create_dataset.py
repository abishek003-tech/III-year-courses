"""
create_dataset.py - Optional Dataset Generator Script
------------------------------------------------------
This script creates a sample dataset (data/dataset.csv) for demonstration.
If you bring your own CSV dataset, you can simply replace `data/dataset.csv` 
with your own file and skip this script!
"""

import os
import pandas as pd
import numpy as np

def generate_sample_dataset():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic features
    n_samples = 200
    age = np.random.randint(18, 65, size=n_samples)
    experience = age - 18 + np.random.randint(-2, 3, size=n_samples)
    experience = np.maximum(experience, 0)
    salary = 30000 + (experience * 2500) + np.random.randint(-5000, 5000, size=n_samples)
    score = np.random.uniform(50, 100, size=n_samples)
    
    # Target: High Performer / Promoted (1 or 0)
    target = ((salary > 55000) & (score > 70)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "age": age,
        "experience": experience,
        "salary": salary,
        "score": score.round(2),
        "target": target
    })
    
    # Introduce a couple of missing values to demonstrate automated missing value handling
    df.loc[5, "salary"] = np.nan
    df.loc[12, "score"] = np.nan
    
    # Save to CSV
    output_path = os.path.join("data", "dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created successfully at '{output_path}'!")
    print(f"Dataset Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    generate_sample_dataset()
