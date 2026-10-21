"""
data_loader.py — Load and validate the UCI Heart Disease dataset.

This module handles:
- Loading the CSV dataset from the data/ directory
- Validating the schema (expected columns, data types, value ranges)
- Reporting dataset statistics
- Providing clear error messages for missing or malformed data

Dataset: UCI Heart Disease (Cleveland subset)
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import os
import pandas as pd
import numpy as np


# ─── Expected Schema ────────────────────────────────────────────────────────────
# Defines the expected columns, their types, and valid ranges for validation.

EXPECTED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Feature metadata used for validation and documentation
FEATURE_INFO = {
    'age':      {'type': 'numeric',     'min': 1,   'max': 120,  'desc': 'Age in years'},
    'sex':      {'type': 'binary',      'values': [0, 1],        'desc': '0=Female, 1=Male'},
    'cp':       {'type': 'categorical', 'values': [1, 2, 3, 4],  'desc': 'Chest pain type (1-4)'},
    'trestbps': {'type': 'numeric',     'min': 50,  'max': 300,  'desc': 'Resting blood pressure (mm Hg)'},
    'chol':     {'type': 'numeric',     'min': 50,  'max': 700,  'desc': 'Serum cholesterol (mg/dL)'},
    'fbs':      {'type': 'binary',      'values': [0, 1],        'desc': 'Fasting blood sugar > 120 mg/dL'},
    'restecg':  {'type': 'categorical', 'values': [0, 1, 2],     'desc': 'Resting ECG results (0-2)'},
    'thalach':  {'type': 'numeric',     'min': 50,  'max': 250,  'desc': 'Max heart rate achieved'},
    'exang':    {'type': 'binary',      'values': [0, 1],        'desc': 'Exercise-induced angina'},
    'oldpeak':  {'type': 'numeric',     'min': -5,  'max': 10,   'desc': 'ST depression from exercise'},
    'slope':    {'type': 'categorical', 'values': [1, 2, 3],     'desc': 'Peak exercise ST slope (1-3)'},
    'ca':       {'type': 'numeric',     'min': 0,   'max': 3,    'desc': 'Major vessels colored (0-3)'},
    'thal':     {'type': 'categorical', 'values': [3, 6, 7],     'desc': 'Thalassemia (3=normal, 6=fixed, 7=reversible)'},
    'target':   {'type': 'binary',      'values': [0, 1],        'desc': '0=No disease, 1=Disease present'},
}

# Which columns are features vs target
TARGET_COLUMN = 'target'
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != TARGET_COLUMN]

# Dataset file path (relative to project root)
DEFAULT_DATA_PATH = os.path.join('data', 'heart.csv')


def get_project_root():
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_default_data_path():
    """Return the absolute path to the default dataset file."""
    return os.path.join(get_project_root(), DEFAULT_DATA_PATH)


def load_dataset(filepath=None):
    """
    Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, uses the default path (data/heart.csv).

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if filepath is None:
        filepath = get_default_data_path()

    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            f"Please ensure heart.csv is in the data/ directory. "
            f"You can download it from: "
            f"https://archive.ics.uci.edu/dataset/45/heart+disease"
        )

    # Check file is not empty
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Dataset file '{filepath}' is empty.")

    # Load with error handling
    try:
        df = pd.read_csv(filepath, na_values=['?', '', 'NA', 'N/A', 'null', 'None'])
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{filepath}' contains no parsable data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse '{filepath}': {e}")

    if df.empty:
        raise ValueError(f"Dataset loaded from '{filepath}' has zero rows.")

    return df


def validate_schema(df):
    """
    Validate that the DataFrame has the expected columns for the heart disease dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    dict
        Validation report with keys:
        - 'valid' (bool): Whether the schema is valid
        - 'missing_columns' (list): Columns expected but not found
        - 'extra_columns' (list): Columns found but not expected
        - 'warnings' (list): Non-fatal issues
    """
    report = {
        'valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'warnings': []
    }

    actual_columns = list(df.columns)

    # Check for missing required columns
    for col in EXPECTED_COLUMNS:
        if col not in actual_columns:
            report['missing_columns'].append(col)
            report['valid'] = False

    # Check for unexpected columns
    for col in actual_columns:
        if col not in EXPECTED_COLUMNS:
            report['extra_columns'].append(col)
            report['warnings'].append(f"Unexpected column '{col}' found — will be ignored.")

    # Check target column exists
    if TARGET_COLUMN not in actual_columns:
        report['valid'] = False
        report['warnings'].append(
            f"Target column '{TARGET_COLUMN}' is missing. Cannot proceed with training."
        )

    return report


def validate_data_quality(df):
    """
    Check data quality: missing values, duplicates, out-of-range values, type issues.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    dict
        Quality report with keys:
        - 'total_rows' (int)
        - 'total_columns' (int)
        - 'missing_values' (dict): Column -> count of missing values
        - 'total_missing' (int)
        - 'duplicate_rows' (int)
        - 'out_of_range' (dict): Column -> count of out-of-range values
        - 'type_issues' (list): Description of type conversion problems
        - 'warnings' (list)
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'total_missing': 0,
        'duplicate_rows': 0,
        'out_of_range': {},
        'type_issues': [],
        'warnings': []
    }

    # ── Missing values ──────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    report['missing_values'] = missing_cols.to_dict()
    report['total_missing'] = int(missing_cols.sum())

    if report['total_missing'] > 0:
        pct = (report['total_missing'] / (len(df) * len(df.columns))) * 100
        report['warnings'].append(
            f"Found {report['total_missing']} missing values ({pct:.2f}% of all cells)."
        )

    # ── Duplicate rows ──────────────────────────────────────────────────────
    report['duplicate_rows'] = int(df.duplicated().sum())
    if report['duplicate_rows'] > 0:
        report['warnings'].append(
            f"Found {report['duplicate_rows']} duplicate row(s)."
        )

    # ── Out-of-range and type checks ────────────────────────────────────────
    for col, info in FEATURE_INFO.items():
        if col not in df.columns:
            continue

        col_data = df[col].dropna()

        # Check if values can be converted to numeric
        if info['type'] in ('numeric', 'binary', 'categorical'):
            non_numeric = pd.to_numeric(col_data, errors='coerce')
            num_non_numeric = non_numeric.isnull().sum() - df[col].isnull().sum()
            if num_non_numeric > 0:
                report['type_issues'].append(
                    f"Column '{col}': {num_non_numeric} values cannot be converted to numeric."
                )

        # Check value ranges
        if info['type'] == 'numeric':
            out_of_range = ((col_data < info['min']) | (col_data > info['max'])).sum()
            if out_of_range > 0:
                report['out_of_range'][col] = int(out_of_range)
                report['warnings'].append(
                    f"Column '{col}': {out_of_range} values outside expected range "
                    f"[{info['min']}, {info['max']}]."
                )

        elif info['type'] in ('binary', 'categorical'):
            valid_values = set(info['values'])
            invalid = (~col_data.isin(valid_values)).sum()
            if invalid > 0:
                report['out_of_range'][col] = int(invalid)
                actual_unique = sorted(col_data.unique().tolist())
                report['warnings'].append(
                    f"Column '{col}': {invalid} values not in expected set {info['values']}. "
                    f"Actual unique values: {actual_unique}"
                )

    return report


def get_dataset_summary(df):
    """
    Generate a human-readable summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.

    Returns
    -------
    dict
        Summary statistics including shape, dtypes, class distribution,
        numeric stats, and feature descriptions.
    """
    summary = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {},
        'class_distribution': {},
        'feature_descriptions': {}
    }

    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats = df[col].describe()
        summary['numeric_stats'][col] = {
            'mean': round(float(stats['mean']), 2),
            'std': round(float(stats['std']), 2),
            'min': round(float(stats['min']), 2),
            'max': round(float(stats['max']), 2),
            'median': round(float(df[col].median()), 2),
            'missing': int(df[col].isnull().sum())
        }

    # Target class distribution
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        total = len(df)
        summary['class_distribution'] = {
            str(k): {
                'count': int(v),
                'percentage': round(v / total * 100, 1)
            }
            for k, v in counts.items()
        }

    # Feature descriptions from metadata
    for col in df.columns:
        if col in FEATURE_INFO:
            summary['feature_descriptions'][col] = FEATURE_INFO[col]['desc']

    return summary


def load_and_validate(filepath=None):
    """
    Complete data loading pipeline: load, validate schema, check quality, summarize.

    This is the main entry point for loading the dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, uses the default path.

    Returns
    -------
    tuple
        (df, schema_report, quality_report, summary)
        - df: pd.DataFrame — the loaded data
        - schema_report: dict — schema validation results
        - quality_report: dict — data quality check results
        - summary: dict — dataset summary statistics
    """
    print("=" * 60)
    print("HEALTHCARE PATIENT RISK PATTERN MINING")
    print("Data Loading & Validation")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/4] Loading dataset...")
    df = load_dataset(filepath)
    print(f"  [OK] Loaded {len(df)} records with {len(df.columns)} columns")

    # Step 2: Schema validation
    print("\n[2/4] Validating schema...")
    schema_report = validate_schema(df)
    if schema_report['valid']:
        print(f"  [OK] Schema is valid -- all {len(EXPECTED_COLUMNS)} expected columns present")
    else:
        print(f"  [FAIL] Schema validation failed!")
        if schema_report['missing_columns']:
            print(f"    Missing: {schema_report['missing_columns']}")
    if schema_report['extra_columns']:
        print(f"  [WARN] Extra columns (will be ignored): {schema_report['extra_columns']}")

    # Step 3: Data quality
    print("\n[3/4] Checking data quality...")
    quality_report = validate_data_quality(df)
    print(f"  Total rows: {quality_report['total_rows']}")
    print(f"  Missing values: {quality_report['total_missing']}")
    print(f"  Duplicate rows: {quality_report['duplicate_rows']}")
    if quality_report['out_of_range']:
        print(f"  Out-of-range: {quality_report['out_of_range']}")
    if quality_report['type_issues']:
        for issue in quality_report['type_issues']:
            print(f"  [WARN] {issue}")
    for w in quality_report['warnings']:
        print(f"  [WARN] {w}")

    # Step 4: Summary
    print("\n[4/4] Generating summary...")
    summary = get_dataset_summary(df)
    if TARGET_COLUMN in df.columns:
        print(f"  Target distribution:")
        for label, info in summary['class_distribution'].items():
            label_name = "No Disease" if label == "0" else "Disease Present"
            print(f"    {label} ({label_name}): {info['count']} ({info['percentage']}%)")

    print("\n" + "=" * 60)
    print("Data loading complete.")
    print("=" * 60)

    return df, schema_report, quality_report, summary


def load_user_csv(filepath):
    """
    Load and validate a user-uploaded CSV file for batch prediction.

    This is a lighter validation — we only need the feature columns,
    not the target column.

    Parameters
    ----------
    filepath : str
        Path to the uploaded CSV file.

    Returns
    -------
    tuple
        (df, errors)
        - df: pd.DataFrame or None if validation fails
        - errors: list of error strings (empty if valid)
    """
    errors = []

    try:
        df = pd.read_csv(filepath, na_values=['?', '', 'NA', 'N/A', 'null', 'None'])
    except Exception as e:
        return None, [f"Failed to read CSV: {str(e)}"]

    if df.empty:
        return None, ["Uploaded CSV file is empty."]

    # Check that at least the feature columns exist
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        errors.append(
            f"Missing required feature columns: {missing_features}. "
            f"Expected columns: {FEATURE_COLUMNS}"
        )
        return None, errors

    # Warn about extra columns (but don't fail)
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra_cols:
        errors.append(f"Warning: Extra columns will be ignored: {extra_cols}")

    return df, errors


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    """Run data loading and validation as a standalone script."""
    df, schema_report, quality_report, summary = load_and_validate()

    print("\n\nDetailed Numeric Statistics:")
    print("-" * 60)
    stats_df = pd.DataFrame(summary['numeric_stats']).T
    print(stats_df.to_string())
