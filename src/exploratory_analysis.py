"""
exploratory_analysis.py -- EDA and risk pattern visualizations.

Generates static plots saved to static/plots/ for the dashboard,
and returns data for Plotly interactive charts on the frontend.

All plots are generated from the TRAINING data to avoid information leakage
in any statistical summaries shown on the dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    get_project_root,
    load_and_validate,
)


# Output directory for plots
PLOTS_DIR = os.path.join(get_project_root(), 'static', 'plots')


def ensure_plots_dir():
    """Create the plots output directory if it doesn't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def set_plot_style():
    """Set consistent, publication-quality plot styling."""
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def plot_age_distribution(df):
    """Plot the distribution of patient ages, colored by target class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    axes[0].hist(df['age'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_title('Patient Age Distribution')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['age'].mean(), color='red', linestyle='--', label=f"Mean: {df['age'].mean():.1f}")
    axes[0].legend()

    # By target class
    no_disease = df[df[TARGET_COLUMN] == 0]['age']
    disease = df[df[TARGET_COLUMN] == 1]['age']
    axes[1].hist(no_disease, bins=15, alpha=0.6, label='No Disease', color='#2ecc71', edgecolor='white')
    axes[1].hist(disease, bins=15, alpha=0.6, label='Disease Present', color='#e74c3c', edgecolor='white')
    axes[1].set_title('Age Distribution by Heart Disease Status')
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'age_distribution.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: age_distribution.png")
    return filepath


def plot_target_distribution(df):
    """Plot the target class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df[TARGET_COLUMN].value_counts()
    labels = ['No Disease (0)', 'Disease Present (1)']
    colors = ['#2ecc71', '#e74c3c']

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white')
    axes[0].set_title('Target Class Distribution')
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(count), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Class Proportion')

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'target_distribution.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: target_distribution.png")
    return filepath


def plot_clinical_distributions(df):
    """Plot distributions of key clinical measurements by target class."""
    clinical_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    feature_labels = {
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dL)',
        'thalach': 'Max Heart Rate Achieved',
        'oldpeak': 'ST Depression (Exercise)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(clinical_features):
        no_disease = df[df[TARGET_COLUMN] == 0][feat].dropna()
        disease = df[df[TARGET_COLUMN] == 1][feat].dropna()

        axes[idx].hist(no_disease, bins=20, alpha=0.6, label='No Disease',
                       color='#2ecc71', edgecolor='white')
        axes[idx].hist(disease, bins=20, alpha=0.6, label='Disease Present',
                       color='#e74c3c', edgecolor='white')
        axes[idx].set_title(feature_labels.get(feat, feat))
        axes[idx].set_xlabel(feat)
        axes[idx].set_ylabel('Count')
        axes[idx].legend()

    plt.suptitle('Clinical Measurements by Heart Disease Status', fontsize=16, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'clinical_distributions.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: clinical_distributions.png")
    return filepath


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features', fontsize=14)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: correlation_heatmap.png")
    return filepath


def plot_feature_comparisons(df):
    """Box plots comparing key features across target classes."""
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    fig, axes = plt.subplots(1, len(features), figsize=(20, 5))

    for idx, feat in enumerate(features):
        data_0 = df[df[TARGET_COLUMN] == 0][feat].dropna()
        data_1 = df[df[TARGET_COLUMN] == 1][feat].dropna()
        bp = axes[idx].boxplot([data_0, data_1],
                               labels=['No Disease', 'Disease'],
                               patch_artist=True,
                               widths=0.6)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        axes[idx].set_title(feat, fontsize=12)
        axes[idx].set_ylabel(feat)

    plt.suptitle('Feature Comparisons by Heart Disease Status', fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'feature_comparisons.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: feature_comparisons.png")
    return filepath


def plot_risk_factor_associations(df):
    """Plot associations between categorical risk factors and heart disease."""
    cat_features = {
        'cp': {1: 'Typical Angina', 2: 'Atypical Angina',
               3: 'Non-Anginal Pain', 4: 'Asymptomatic'},
        'sex': {0: 'Female', 1: 'Male'},
        'fbs': {0: 'FBS <= 120', 1: 'FBS > 120'},
        'exang': {0: 'No', 1: 'Yes'}
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (feat, labels) in enumerate(cat_features.items()):
        ct = pd.crosstab(df[feat].map(labels), df[TARGET_COLUMN])
        ct.columns = ['No Disease', 'Disease']
        ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'],
                edgecolor='white', alpha=0.8)
        axes[idx].set_title(f'{feat.upper()} vs Heart Disease')
        axes[idx].set_xlabel(feat)
        axes[idx].set_ylabel('Count')
        axes[idx].legend()
        axes[idx].tick_params(axis='x', rotation=30)

    plt.suptitle('Risk Factor Associations', fontsize=16, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'risk_factor_associations.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: risk_factor_associations.png")
    return filepath


def plot_clustering(df):
    """
    Perform K-Means clustering on numeric features for pattern discovery.

    NOTE: Clusters represent STATISTICAL GROUPINGS, not clinical risk categories.
    This visualization shows how patients naturally group based on their
    clinical measurements.
    """
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cluster_data = df[numeric_features].dropna()

    # Standardize for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Find optimal k using silhouette score
    from sklearn.metrics import silhouette_score
    k_range = range(2, 7)
    silhouette_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)

    best_k = list(k_range)[np.argmax(silhouette_scores)]

    # Final clustering with best k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(scaled_data)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Silhouette scores
    axes[0].plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    axes[0].set_title('Silhouette Score vs Number of Clusters')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].legend()

    # PCA with cluster labels
    scatter1 = axes[1].scatter(pca_data[:, 0], pca_data[:, 1],
                                c=cluster_labels, cmap='Set2', alpha=0.7, s=50)
    axes[1].set_title(f'Patient Clusters (k={best_k}, PCA Projection)')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.colorbar(scatter1, ax=axes[1], label='Cluster')

    # PCA with actual target labels
    target_subset = df.loc[cluster_data.index, TARGET_COLUMN]
    scatter2 = axes[2].scatter(pca_data[:, 0], pca_data[:, 1],
                                c=target_subset, cmap='RdYlGn_r', alpha=0.7, s=50)
    axes[2].set_title('Actual Heart Disease Status (PCA Projection)')
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.colorbar(scatter2, ax=axes[2], label='Target (0=No, 1=Yes)')

    plt.suptitle(
        'K-Means Clustering Analysis\n'
        '(Clusters are statistical groupings, not clinical risk categories)',
        fontsize=14, y=1.04
    )
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'clustering_analysis.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: clustering_analysis.png (best k={best_k})")
    return filepath, best_k, silhouette_scores


def plot_grouped_summary(df):
    """Generate grouped summary statistics by target class."""
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    grouped = df.groupby(TARGET_COLUMN)[numeric_features].agg(['mean', 'median', 'std'])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    # Build a simple table
    cell_text = []
    row_labels = []
    for feat in numeric_features:
        for cls in [0, 1]:
            label = f"{feat} ({'No Disease' if cls == 0 else 'Disease'})"
            mean_val = grouped.loc[cls, (feat, 'mean')]
            med_val = grouped.loc[cls, (feat, 'median')]
            std_val = grouped.loc[cls, (feat, 'std')]
            cell_text.append([f"{mean_val:.1f}", f"{med_val:.1f}", f"{std_val:.1f}"])
            row_labels.append(label)

    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=['Mean', 'Median', 'Std Dev'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Grouped Summary Statistics by Target Class', fontsize=14, pad=20)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'grouped_summary.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  [OK] Saved: grouped_summary.png")
    return filepath


def generate_eda_data(df):
    """
    Generate EDA summary data for the frontend API (Plotly charts).

    Returns JSON-serializable dictionaries for interactive charting.
    """
    eda_data = {}

    # Age distribution data
    eda_data['age_distribution'] = {
        'no_disease': df[df[TARGET_COLUMN] == 0]['age'].dropna().tolist(),
        'disease': df[df[TARGET_COLUMN] == 1]['age'].dropna().tolist(),
    }

    # Target distribution
    counts = df[TARGET_COLUMN].value_counts()
    eda_data['target_distribution'] = {
        'labels': ['No Disease', 'Disease Present'],
        'values': [int(counts.get(0, 0)), int(counts.get(1, 0))]
    }

    # Correlation matrix
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', TARGET_COLUMN]
    available_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[available_cols].corr()
    eda_data['correlation'] = {
        'columns': available_cols,
        'values': corr.values.tolist()
    }

    # Feature statistics by class
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    eda_data['feature_stats'] = {}
    for feat in features:
        if feat in df.columns:
            eda_data['feature_stats'][feat] = {
                'no_disease': {
                    'mean': round(float(df[df[TARGET_COLUMN] == 0][feat].mean()), 2),
                    'std': round(float(df[df[TARGET_COLUMN] == 0][feat].std()), 2),
                    'values': df[df[TARGET_COLUMN] == 0][feat].dropna().tolist()
                },
                'disease': {
                    'mean': round(float(df[df[TARGET_COLUMN] == 1][feat].mean()), 2),
                    'std': round(float(df[df[TARGET_COLUMN] == 1][feat].std()), 2),
                    'values': df[df[TARGET_COLUMN] == 1][feat].dropna().tolist()
                }
            }

    # Categorical feature distributions
    cat_features = ['cp', 'sex', 'fbs', 'exang']
    eda_data['categorical_stats'] = {}
    for feat in cat_features:
        if feat in df.columns:
            ct = pd.crosstab(df[feat], df[TARGET_COLUMN])
            eda_data['categorical_stats'][feat] = {
                'categories': [str(x) for x in ct.index.tolist()],
                'no_disease': ct[0].tolist() if 0 in ct.columns else [],
                'disease': ct[1].tolist() if 1 in ct.columns else []
            }

    return eda_data


def run_full_eda(df=None):
    """
    Execute the complete EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataset to analyze. If None, loads from default path.

    Returns
    -------
    dict
        EDA results including plot paths and interactive data.
    """
    if df is None:
        df, _, _, _ = load_and_validate()

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    ensure_plots_dir()
    set_plot_style()

    results = {'plot_paths': {}}

    # Generate all plots
    print("\n[1/8] Age distribution...")
    results['plot_paths']['age_distribution'] = plot_age_distribution(df)

    print("[2/8] Target distribution...")
    results['plot_paths']['target_distribution'] = plot_target_distribution(df)

    print("[3/8] Clinical distributions...")
    results['plot_paths']['clinical_distributions'] = plot_clinical_distributions(df)

    print("[4/8] Correlation heatmap...")
    results['plot_paths']['correlation_heatmap'] = plot_correlation_heatmap(df)

    print("[5/8] Feature comparisons...")
    results['plot_paths']['feature_comparisons'] = plot_feature_comparisons(df)

    print("[6/8] Risk factor associations...")
    results['plot_paths']['risk_factor_associations'] = plot_risk_factor_associations(df)

    print("[7/8] Clustering analysis...")
    cluster_path, best_k, sil_scores = plot_clustering(df)
    results['plot_paths']['clustering_analysis'] = cluster_path
    results['clustering'] = {'best_k': best_k, 'silhouette_scores': sil_scores}

    print("[8/8] Grouped summary...")
    results['plot_paths']['grouped_summary'] = plot_grouped_summary(df)

    # Generate interactive data for frontend
    print("\nGenerating interactive chart data...")
    results['eda_data'] = generate_eda_data(df)

    # Save EDA data as JSON for the API
    eda_json_path = os.path.join(get_project_root(), 'models', 'eda_data.json')
    os.makedirs(os.path.dirname(eda_json_path), exist_ok=True)
    with open(eda_json_path, 'w') as f:
        json.dump(results['eda_data'], f, indent=2)
    print(f"  [OK] EDA data saved to: {eda_json_path}")

    print("\n" + "=" * 60)
    print(f"EDA complete. {len(results['plot_paths'])} plots generated.")
    print("=" * 60)

    return results


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_full_eda()
    print(f"\nGenerated plots:")
    for name, path in results['plot_paths'].items():
        print(f"  {name}: {path}")
