"""
train_model.py -- Train and compare multiple ML classification models.

Models trained:
1. Logistic Regression (interpretable baseline)
2. Decision Tree (fully interpretable, visualizable)
3. Random Forest (ensemble, robust)
4. Gradient Boosting (state-of-the-art for tabular data)

Evaluation:
- 5-fold stratified cross-validation on training data
- Model selection based on mean CV F1-score
- Final evaluation on held-out test set (used ONCE)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.inspection import permutation_importance

from src.data_loader import get_project_root
from src.data_preprocessing import run_full_preprocessing

warnings.filterwarnings('ignore', category=FutureWarning)

# Random seed
RANDOM_STATE = 42

# Output directory
MODELS_DIR = os.path.join(get_project_root(), 'models')


def get_models():
    """
    Return a dictionary of models to train and compare.

    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            C=1.0
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    }
    return models


def cross_validate_models(models, X_train, y_train, cv_folds=5):
    """
    Perform stratified k-fold cross-validation on all models.

    Parameters
    ----------
    models : dict
        {model_name: model_instance}
    X_train : np.ndarray
        Training features (preprocessed).
    y_train : np.ndarray or pd.Series
        Training labels.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    dict
        {model_name: {metric: [fold_scores], ...}}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = {}

    print(f"\nRunning {cv_folds}-fold stratified cross-validation...")
    print("-" * 70)

    for name, model in models.items():
        print(f"\n  {name}...")
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        cv_results[name] = {
            'accuracy': scores['test_accuracy'].tolist(),
            'precision': scores['test_precision'].tolist(),
            'recall': scores['test_recall'].tolist(),
            'f1': scores['test_f1'].tolist(),
            'roc_auc': scores['test_roc_auc'].tolist(),
            'mean_accuracy': float(np.mean(scores['test_accuracy'])),
            'mean_precision': float(np.mean(scores['test_precision'])),
            'mean_recall': float(np.mean(scores['test_recall'])),
            'mean_f1': float(np.mean(scores['test_f1'])),
            'mean_roc_auc': float(np.mean(scores['test_roc_auc'])),
            'std_f1': float(np.std(scores['test_f1'])),
        }

        print(f"    Accuracy:  {cv_results[name]['mean_accuracy']:.4f}")
        print(f"    Precision: {cv_results[name]['mean_precision']:.4f}")
        print(f"    Recall:    {cv_results[name]['mean_recall']:.4f}")
        print(f"    F1-Score:  {cv_results[name]['mean_f1']:.4f} (+/- {cv_results[name]['std_f1']:.4f})")
        print(f"    ROC-AUC:   {cv_results[name]['mean_roc_auc']:.4f}")

    return cv_results


def select_best_model(cv_results):
    """
    Select the best model based on mean CV F1-score.

    Parameters
    ----------
    cv_results : dict
        Cross-validation results from cross_validate_models().

    Returns
    -------
    str
        Name of the best model.
    """
    best_name = max(cv_results, key=lambda k: cv_results[k]['mean_f1'])
    print(f"\n  Best model (by CV F1-score): {best_name}")
    print(f"    Mean F1: {cv_results[best_name]['mean_f1']:.4f}")
    return best_name


def train_final_models(models, X_train, y_train):
    """
    Train all models on the full training set.

    Parameters
    ----------
    models : dict
        {model_name: model_instance}
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray or pd.Series
        Training labels.

    Returns
    -------
    dict
        {model_name: fitted_model}
    """
    fitted_models = {}
    print("\nTraining final models on full training set...")

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models


def evaluate_on_test(fitted_models, X_test, y_test):
    """
    Evaluate all fitted models on the held-out test set.

    This should be called ONLY ONCE for the final report.

    Parameters
    ----------
    fitted_models : dict
        {model_name: fitted_model}
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray or pd.Series
        Test labels.

    Returns
    -------
    dict
        {model_name: {metric: value, confusion_matrix: [[...]], ...}}
    """
    test_results = {}
    print("\nEvaluating on held-out test set (FINAL - used once)...")
    print("-" * 70)

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))

            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }

            # Precision-Recall curve data
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            metrics['pr_curve'] = {
                'precision': prec.tolist(),
                'recall': rec.tolist()
            }

        test_results[name] = metrics

        print(f"\n  {name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    Confusion Matrix: {metrics['confusion_matrix']}")

    return test_results


def compute_feature_importance(fitted_models, X_train, y_train, feature_names):
    """
    Compute feature importance for all models.

    Uses built-in feature_importances_ for tree-based models,
    coefficient magnitudes for logistic regression,
    and permutation importance as a consistent fallback.

    Parameters
    ----------
    fitted_models : dict
        {model_name: fitted_model}
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray or pd.Series
        Training labels.
    feature_names : list
        Names of features after preprocessing.

    Returns
    -------
    dict
        {model_name: {feature_name: importance_score}}
    """
    importance_results = {}

    for name, model in fitted_models.items():
        importances = {}

        # Tree-based models have built-in feature_importances_
        if hasattr(model, 'feature_importances_'):
            raw_imp = model.feature_importances_
            for feat, imp in zip(feature_names, raw_imp):
                importances[feat] = float(imp)

        # Logistic Regression: use absolute coefficient values
        elif hasattr(model, 'coef_'):
            raw_imp = np.abs(model.coef_[0])
            for feat, imp in zip(feature_names, raw_imp):
                importances[feat] = float(imp)

        else:
            # Fallback: permutation importance
            perm_imp = permutation_importance(
                model, X_train, y_train,
                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            for feat, imp in zip(feature_names, perm_imp.importances_mean):
                importances[feat] = float(imp)

        # Sort by importance (descending)
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        importance_results[name] = importances

    return importance_results


def save_models_and_results(fitted_models, best_model_name, cv_results,
                            test_results, importance_results, feature_names):
    """
    Save all models, metrics, and results to disk.

    Saves:
    - best_model.joblib: The selected best model
    - all_models.joblib: All trained models
    - metrics.json: All evaluation metrics, feature importances, comparison data
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save best model
    best_model = fitted_models[best_model_name]
    best_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_model, best_path)
    print(f"\n  [OK] Best model saved: {best_path}")

    # Save all models
    all_path = os.path.join(MODELS_DIR, 'all_models.joblib')
    joblib.dump(fitted_models, all_path)
    print(f"  [OK] All models saved: {all_path}")

    # Compile all metrics
    metrics = {
        'best_model': best_model_name,
        'feature_names': feature_names,
        'cv_results': {},
        'test_results': test_results,
        'feature_importance': importance_results,
        'model_comparison': []
    }

    # Simplify CV results for JSON (remove per-fold arrays)
    for name in cv_results:
        metrics['cv_results'][name] = {
            k: v for k, v in cv_results[name].items()
            if k.startswith('mean_') or k.startswith('std_')
        }

    # Build comparison table
    for name in fitted_models:
        row = {
            'model': name,
            'cv_f1': round(cv_results[name]['mean_f1'], 4),
            'cv_f1_std': round(cv_results[name]['std_f1'], 4),
            'cv_accuracy': round(cv_results[name]['mean_accuracy'], 4),
            'cv_roc_auc': round(cv_results[name]['mean_roc_auc'], 4),
            'test_accuracy': round(test_results[name]['accuracy'], 4),
            'test_precision': round(test_results[name]['precision'], 4),
            'test_recall': round(test_results[name]['recall'], 4),
            'test_f1': round(test_results[name]['f1'], 4),
            'test_roc_auc': round(test_results[name].get('roc_auc', 0), 4),
            'is_best': name == best_model_name
        }
        metrics['model_comparison'].append(row)

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Metrics saved: {metrics_path}")

    return metrics


def run_full_training():
    """
    Execute the complete training pipeline end-to-end.

    Steps:
    1. Load and preprocess data
    2. Define models
    3. Cross-validate on training set
    4. Select best model by CV F1-score
    5. Train all models on full training set
    6. Evaluate on held-out test set (once)
    7. Compute feature importances
    8. Save everything

    Returns
    -------
    dict
        Complete training results.
    """
    # Step 1: Preprocess
    prep_results = run_full_preprocessing()

    X_train = prep_results['X_train_processed']
    X_val = prep_results['X_val_processed']
    X_test = prep_results['X_test_processed']
    y_train = prep_results['y_train']
    y_val = prep_results['y_val']
    y_test = prep_results['y_test']
    feature_names = prep_results['feature_names']

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Step 2: Define models
    print("\n[1/6] Defining models...")
    models = get_models()
    for name in models:
        print(f"  - {name}")

    # Step 3: Cross-validation
    print("\n[2/6] Cross-validation (5-fold stratified)...")
    cv_results = cross_validate_models(models, X_train, y_train)

    # Step 4: Select best model
    print("\n[3/6] Selecting best model...")
    best_model_name = select_best_model(cv_results)

    # Step 5: Train on full training set
    print("\n[4/6] Training all models on full training set...")
    models = get_models()  # Get fresh instances
    fitted_models = train_final_models(models, X_train, y_train)

    # Step 6: Evaluate on test set
    print("\n[5/6] Evaluating on test set...")
    test_results = evaluate_on_test(fitted_models, X_test, y_test)

    # Step 7: Feature importance
    print("\n[6/6] Computing feature importances...")
    importance_results = compute_feature_importance(
        fitted_models, X_train, y_train, feature_names
    )

    # Save everything
    print("\nSaving models and results...")
    metrics = save_models_and_results(
        fitted_models, best_model_name, cv_results,
        test_results, importance_results, feature_names
    )

    print("\n" + "=" * 60)
    print(f"Training complete. Best model: {best_model_name}")
    print("=" * 60)

    return {
        'fitted_models': fitted_models,
        'best_model_name': best_model_name,
        'cv_results': cv_results,
        'test_results': test_results,
        'importance_results': importance_results,
        'metrics': metrics,
        'feature_names': feature_names,
        'prep_results': prep_results
    }


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_full_training()

    print("\n\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(f"\n{'Model':<25} {'CV F1':>8} {'Test Acc':>10} {'Test F1':>8} {'Test AUC':>10} {'Best':>6}")
    print("-" * 70)
    for row in results['metrics']['model_comparison']:
        best_marker = ' <<' if row['is_best'] else ''
        print(f"{row['model']:<25} {row['cv_f1']:>8.4f} {row['test_accuracy']:>10.4f} "
              f"{row['test_f1']:>8.4f} {row['test_roc_auc']:>10.4f}{best_marker}")
