"""
evaluate_model.py -- Generate evaluation visualizations for trained models.

Creates:
- Confusion matrix plots for all models
- ROC curve comparison
- Precision-Recall curve comparison
- Feature importance bar chart for the best model
- Model comparison bar chart
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_project_root


PLOTS_DIR = os.path.join(get_project_root(), 'static', 'plots')
MODELS_DIR = os.path.join(get_project_root(), 'models')


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_metrics():
    """Load the saved metrics from training."""
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. "
            f"Run training first: python -m src.train_model"
        )
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrices(metrics):
    """Plot confusion matrices for all models in a grid."""
    test_results = metrics['test_results']
    n_models = len(test_results)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(test_results.items()):
        cm = np.array(result['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    ax=axes[idx])
        axes[idx].set_title(f'{name}', fontsize=11)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices (Test Set)', fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: confusion_matrices.png")
    return filepath


def plot_roc_curves(metrics):
    """Plot ROC curves for all models on one chart."""
    test_results = metrics['test_results']

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (name, result) in enumerate(test_results.items()):
        if 'roc_curve' in result:
            fpr = result['roc_curve']['fpr']
            tpr = result['roc_curve']['tpr']
            auc = result.get('roc_auc', 0)
            ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                    linewidth=2, label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'roc_curves.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: roc_curves.png")
    return filepath


def plot_pr_curves(metrics):
    """Plot Precision-Recall curves for all models."""
    test_results = metrics['test_results']

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, (name, result) in enumerate(test_results.items()):
        if 'pr_curve' in result:
            precision = result['pr_curve']['precision']
            recall = result['pr_curve']['recall']
            f1 = result.get('f1', 0)
            ax.plot(recall, precision, color=colors[idx % len(colors)],
                    linewidth=2, label=f'{name} (F1={f1:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Model Comparison')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'pr_curves.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: pr_curves.png")
    return filepath


def plot_feature_importance(metrics):
    """Plot feature importance for the best model."""
    best_model = metrics['best_model']
    importance = metrics['feature_importance'].get(best_model, {})

    if not importance:
        print("  [WARN] No feature importance data available.")
        return None

    # Take top 15 features
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), values, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Feature Importance - {best_model}')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'feature_importance.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: feature_importance.png")
    return filepath


def plot_model_comparison(metrics):
    """Create a grouped bar chart comparing all models across key metrics."""
    comparison = metrics['model_comparison']

    model_names = [m['model'] for m in comparison]
    metric_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    x = np.arange(len(model_names))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [m[key] for m in comparison]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Test Set)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'model_comparison.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: model_comparison.png")
    return filepath


def run_evaluation_plots():
    """Generate all evaluation visualizations."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION VISUALIZATIONS")
    print("=" * 60)

    ensure_dirs()
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.0)

    metrics = load_metrics()

    results = {}
    print("\n[1/5] Confusion matrices...")
    results['confusion_matrices'] = plot_confusion_matrices(metrics)

    print("[2/5] ROC curves...")
    results['roc_curves'] = plot_roc_curves(metrics)

    print("[3/5] Precision-Recall curves...")
    results['pr_curves'] = plot_pr_curves(metrics)

    print("[4/5] Feature importance...")
    results['feature_importance'] = plot_feature_importance(metrics)

    print("[5/5] Model comparison...")
    results['model_comparison'] = plot_model_comparison(metrics)

    print("\n" + "=" * 60)
    print("Evaluation plots complete.")
    print("=" * 60)

    return results


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_evaluation_plots()
