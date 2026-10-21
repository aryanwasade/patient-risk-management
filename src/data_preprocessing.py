"""
data_preprocessing.py -- Preprocessing pipeline for the Heart Disease dataset.

This module handles:
- Separating features from the target variable
- Train / validation / test split (60/20/20, stratified)
- Missing value imputation (median for numeric, mode for categorical)
- One-hot encoding for nominal categorical features
- Standard scaling for numeric features
- All via scikit-learn Pipeline + ColumnTransformer (fit on training data ONLY)

Key design:
- Preprocessing is FIT only on training data to prevent data leakage.
- The fitted pipeline is saved as a Joblib artifact for reuse at prediction time.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.data_loader import (
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    get_project_root,
)


# ─── Feature Group Definitions ──────────────────────────────────────────────────
# Derived from FEATURE_INFO metadata in data_loader.py

# Numeric features that get imputed (median) + scaled
NUMERIC_FEATURES = [
    col for col, info in FEATURE_INFO.items()
    if info['type'] == 'numeric' and col != TARGET_COLUMN
]
# Result: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Binary features -- already 0/1, just impute (mode), no encoding needed
BINARY_FEATURES = [
    col for col, info in FEATURE_INFO.items()
    if info['type'] == 'binary' and col != TARGET_COLUMN
]
# Result: ['sex', 'fbs', 'exang']

# Categorical features -- impute (mode) + one-hot encode
CATEGORICAL_FEATURES = [
    col for col, info in FEATURE_INFO.items()
    if info['type'] == 'categorical' and col != TARGET_COLUMN
]
# Result: ['cp', 'restecg', 'slope', 'thal']

# Sanity check: all feature columns accounted for
ALL_PREPROCESSED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
assert set(ALL_PREPROCESSED_FEATURES) == set(FEATURE_COLUMNS), (
    f"Feature mismatch!\n"
    f"  Preprocessed: {sorted(ALL_PREPROCESSED_FEATURES)}\n"
    f"  Expected:     {sorted(FEATURE_COLUMNS)}"
)

# Random seed for reproducibility
RANDOM_STATE = 42


def split_features_target(df):
    """
    Separate features (X) from the target variable (y).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame of features, y is a Series of labels.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """
    Split data into training, validation, and test sets using stratified sampling.

    Split strategy:
    - First split: 80% train+val, 20% test
    - Second split: 75% train, 25% val (of the 80%) = 60% train, 20% val overall

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion for the test set (default 0.2).
    val_size : float
        Proportion for the validation set (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            'X_train': pd.DataFrame,
            'X_val': pd.DataFrame,
            'X_test': pd.DataFrame,
            'y_train': pd.Series,
            'y_val': pd.Series,
            'y_test': pd.Series
        }
    """
    # First split: separate out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: separate train and validation from the remaining data
    # val_size=0.2 of total -> 0.2/0.8 = 0.25 of remaining
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state
    )

    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    print(f"\nData Split Summary:")
    print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
    print(f"\n  Training target distribution:")
    for label, count in y_train.value_counts().sort_index().items():
        pct = count / len(y_train) * 100
        label_name = "No Disease" if label == 0 else "Disease"
        print(f"    {label} ({label_name}): {count} ({pct:.1f}%)")

    return splits


def build_preprocessing_pipeline():
    """
    Build the scikit-learn preprocessing pipeline using ColumnTransformer.

    Pipeline structure:
    - Numeric features: Impute missing (median) -> StandardScaler
    - Binary features: Impute missing (most frequent) -> pass-through (already 0/1)
    - Categorical features: Impute missing (most frequent) -> OneHotEncoder

    Returns
    -------
    sklearn.compose.ColumnTransformer
        The unfitted preprocessing pipeline.
    """

    # Sub-pipeline for numeric features
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Sub-pipeline for binary features (just impute, no scaling needed for 0/1)
    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # Sub-pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('bin', binary_pipeline, BINARY_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder='drop',    # Drop any unexpected columns
        verbose_feature_names_out=True
    )

    return preprocessor


def get_feature_names(preprocessor):
    """
    Extract human-readable feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        A fitted ColumnTransformer.

    Returns
    -------
    list
        List of feature names in the order they appear in the transformed output.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback: manually construct names
        names = []
        names.extend([f"num__{col}" for col in NUMERIC_FEATURES])
        names.extend([f"bin__{col}" for col in BINARY_FEATURES])
        # For one-hot encoded, we approximate
        for col in CATEGORICAL_FEATURES:
            names.append(f"cat__{col}_encoded")
        return names


def fit_and_transform(preprocessor, splits):
    """
    Fit the preprocessor on training data ONLY, then transform all splits.

    This prevents data leakage by ensuring validation and test data
    are transformed using statistics computed from the training set.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        The unfitted preprocessing pipeline.
    splits : dict
        Output from split_data().

    Returns
    -------
    dict
        {
            'X_train_processed': np.ndarray,
            'X_val_processed': np.ndarray,
            'X_test_processed': np.ndarray,
            'feature_names': list,
            'preprocessor': fitted ColumnTransformer
        }
    """
    print("\nPreprocessing Pipeline:")
    print(f"  Numeric features ({len(NUMERIC_FEATURES)}):     {NUMERIC_FEATURES}")
    print(f"    -> Impute (median) -> StandardScaler")
    print(f"  Binary features ({len(BINARY_FEATURES)}):      {BINARY_FEATURES}")
    print(f"    -> Impute (most frequent)")
    print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"    -> Impute (most frequent) -> OneHotEncoder (drop='first')")

    # FIT on training data only
    print("\n  Fitting on training data...")
    X_train_processed = preprocessor.fit_transform(splits['X_train'])

    # TRANSFORM validation and test data (no fitting!)
    print("  Transforming validation data...")
    X_val_processed = preprocessor.transform(splits['X_val'])

    print("  Transforming test data...")
    X_test_processed = preprocessor.transform(splits['X_test'])

    # Get feature names after transformation
    feature_names = get_feature_names(preprocessor)

    print(f"\n  Input features:  {splits['X_train'].shape[1]}")
    print(f"  Output features: {X_train_processed.shape[1]} (after one-hot encoding)")
    print(f"  Feature names: {feature_names}")

    return {
        'X_train_processed': X_train_processed,
        'X_val_processed': X_val_processed,
        'X_test_processed': X_test_processed,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }


def save_pipeline(preprocessor, filepath=None):
    """
    Save the fitted preprocessing pipeline to disk.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        The fitted preprocessing pipeline.
    filepath : str, optional
        Path to save to. Default: models/preprocessing_pipeline.joblib
    """
    if filepath is None:
        filepath = os.path.join(get_project_root(), 'models', 'preprocessing_pipeline.joblib')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"\n  [OK] Preprocessing pipeline saved to: {filepath}")


def load_pipeline(filepath=None):
    """
    Load a fitted preprocessing pipeline from disk.

    Parameters
    ----------
    filepath : str, optional
        Path to load from. Default: models/preprocessing_pipeline.joblib

    Returns
    -------
    sklearn.compose.ColumnTransformer
        The fitted preprocessing pipeline.
    """
    if filepath is None:
        filepath = os.path.join(get_project_root(), 'models', 'preprocessing_pipeline.joblib')

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at '{filepath}'. "
            f"Run the training pipeline first (python -m src.data_preprocessing)."
        )

    preprocessor = joblib.load(filepath)
    print(f"  [OK] Preprocessing pipeline loaded from: {filepath}")
    return preprocessor


def preprocess_single_input(patient_data, preprocessor):
    """
    Preprocess a single patient's data for prediction.

    Parameters
    ----------
    patient_data : dict
        Dictionary of feature_name -> value for one patient.
    preprocessor : sklearn.compose.ColumnTransformer
        Fitted preprocessing pipeline.

    Returns
    -------
    np.ndarray
        Preprocessed feature vector (1 x n_features).

    Raises
    ------
    ValueError
        If required features are missing.
    """
    # Validate required features exist
    missing = [f for f in FEATURE_COLUMNS if f not in patient_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Create a single-row DataFrame in the expected column order
    df = pd.DataFrame([patient_data])[FEATURE_COLUMNS]

    # Convert all values to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply the fitted preprocessing pipeline
    processed = preprocessor.transform(df)

    return processed


def run_full_preprocessing(df=None):
    """
    Execute the complete preprocessing pipeline end-to-end.

    This is the main entry point for preprocessing. It:
    1. Separates features from target
    2. Splits data into train/val/test
    3. Builds and fits the preprocessing pipeline (on train only)
    4. Saves the fitted pipeline

    Parameters
    ----------
    df : pd.DataFrame, optional
        Full dataset. If None, loads from the default path.

    Returns
    -------
    dict
        Complete preprocessing results including splits, processed arrays,
        feature names, and the fitted preprocessor.
    """
    if df is None:
        from src.data_loader import load_and_validate
        df, _, _, _ = load_and_validate()

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Separate features and target
    print("\n[1/4] Separating features and target...")
    X, y = split_features_target(df)
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape:   {y.shape}")

    # Step 2: Split data
    print("\n[2/4] Splitting data (60/20/20)...")
    splits = split_data(X, y)

    # Step 3: Build and fit preprocessing pipeline
    print("\n[3/4] Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()
    results = fit_and_transform(preprocessor, splits)

    # Step 4: Save pipeline
    print("\n[4/4] Saving preprocessing pipeline...")
    save_pipeline(results['preprocessor'])

    # Merge splits and processed data into one result
    results.update(splits)
    results['y_train'] = splits['y_train']
    results['y_val'] = splits['y_val']
    results['y_test'] = splits['y_test']

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)

    return results


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    """Run the full preprocessing pipeline as a standalone script."""
    results = run_full_preprocessing()

    # Print a summary of what was created
    print("\n\nPreprocessing Summary:")
    print("-" * 40)
    print(f"  Training samples:   {results['X_train_processed'].shape}")
    print(f"  Validation samples: {results['X_val_processed'].shape}")
    print(f"  Test samples:       {results['X_test_processed'].shape}")
    print(f"  Feature count:      {len(results['feature_names'])}")
    print(f"  Features: {results['feature_names']}")
