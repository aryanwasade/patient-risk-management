"""
run_pipeline.py -- Master execution script for Healthcare Patient Risk Mining.

Executes the entire end-to-end workflow:
1. Dataset loading and verification
2. Preprocessing pipeline build and serialization
3. Exploratory Data Analysis & visual plot generation
4. Model training across 4 architectures and selection
5. Comprehensive diagnostic evaluation & evaluation curves
"""

import sys
import os
import time

from src.data_loader import load_and_validate
from src.exploratory_analysis import run_full_eda
from src.train_model import run_full_training
from src.evaluate_model import run_evaluation_plots


def run_complete_pipeline():
    """Execute all pipeline stages in sequence."""
    start_time = time.time()

    print("\n" + "=" * 70)
    print(" HEALTHCARE PATIENT RISK PATTERN MINING: END-TO-END PIPELINE")
    print("=" * 70)

    # Stage 1: Load and Validate Data
    print("\n[STEP 1/4] Loading & Validating Dataset...")
    df, report, raw_path, is_valid = load_and_validate()
    if not is_valid:
        print("ERROR: Dataset validation failed!")
        sys.exit(1)

    # Stage 2: Exploratory Data Analysis & Pattern Mining
    print("\n[STEP 2/4] Mining Risk Patterns & Generating EDA Visualizations...")
    eda_results = run_full_eda(df)

    # Stage 3: Model Training & Selection (fits preprocessing pipeline, trains 4 models)
    print("\n[STEP 3/4] Training 4 Classifiers & 5-Fold Stratified Cross-Validation...")
    train_results = run_full_training()

    # Stage 4: Evaluation Plots & Diagnostic Curves
    print("\n[STEP 4/4] Generating Diagnostic Evaluation Visualizations...")
    eval_results = run_evaluation_plots()

    elapsed = time.time() - start_time
    best_model_name = train_results.get('best_model_name', 'Logistic Regression')
    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f" Best Selected Model: {best_model_name}")
    print("=" * 70)
    print("\nTo launch the interactive dashboard:")
    print("  python backend/app.py\n")


if __name__ == '__main__':
    run_complete_pipeline()
