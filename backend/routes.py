"""
routes.py -- API blueprints and routing for Healthcare Patient Risk Mining.
"""

import os
import json
import pandas as pd
from flask import Blueprint, request, jsonify, send_from_directory, current_app

from src.data_loader import (
    load_and_validate,
    FEATURE_COLUMNS,
    FEATURE_INFO,
    TARGET_COLUMN,
    get_project_root,
)
from src.predict import (
    predict_single,
    predict_batch,
    load_model_and_pipeline,
    DISCLAIMER,
    RISK_THRESHOLDS,
)

api_bp = Blueprint('api', __name__)

# Cache model and pipeline for fast request handling
_model = None
_preprocessor = None


def get_model_and_pipeline():
    """Lazy load and cache model and preprocessor."""
    global _model, _preprocessor
    if _model is None or _preprocessor is None:
        try:
            _model, _preprocessor = load_model_and_pipeline()
        except Exception as e:
            current_app.logger.warning(f"Could not load model at startup: {e}")
    return _model, _preprocessor


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model, preprocessor = get_model_and_pipeline()
    return jsonify({
        'status': 'healthy',
        'service': 'Healthcare Patient Risk Pattern Mining API',
        'model_loaded': model is not None and preprocessor is not None
    })


@api_bp.route('/dataset-info', methods=['GET'])
def dataset_info():
    """Return dataset metadata, feature dictionary, and summary statistics."""
    try:
        df, report, _, _ = load_and_validate()
        sample = df.head(10).to_dict(orient='records')
        
        # Numeric stats
        numeric_cols = [c for c, info in FEATURE_INFO.items() if info.get('type') == 'numeric']
        stats = df[numeric_cols].describe().round(2).to_dict()

        return jsonify({
            'total_records': len(df),
            'feature_count': len(FEATURE_COLUMNS),
            'target_distribution': df[TARGET_COLUMN].value_counts().to_dict(),
            'features': FEATURE_INFO,
            'summary_stats': stats,
            'sample_records': sample,
            'source': 'UCI Machine Learning Repository - Heart Disease (Cleveland Database)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/model-info', methods=['GET'])
def model_info():
    """Return model performance metrics, comparison, and cross-validation results."""
    metrics_path = os.path.join(get_project_root(), 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Model metrics file not found. Run training first.'}), 404

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/eda', methods=['GET'])
def eda_info():
    """Return EDA structured data and available visual plots list."""
    eda_path = os.path.join(get_project_root(), 'models', 'eda_data.json')
    plots_dir = os.path.join(get_project_root(), 'static', 'plots')

    available_plots = []
    if os.path.exists(plots_dir):
        available_plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]

    eda_data = {}
    if os.path.exists(eda_path):
        try:
            with open(eda_path, 'r') as f:
                eda_data = json.load(f)
        except Exception as e:
            current_app.logger.error(f"Error loading eda_data.json: {e}")

    return jsonify({
        'plots': available_plots,
        'eda_data': eda_data,
        'plot_base_url': '/static/plots/'
    })


@api_bp.route('/feature-importance', methods=['GET'])
def feature_importance():
    """Return feature importances or model coefficients."""
    metrics_path = os.path.join(get_project_root(), 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Metrics not found. Train models first.'}), 404

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        best_model_name = metrics_data.get('best_model', 'Logistic Regression')
        model, preprocessor = get_model_and_pipeline()

        feature_names = metrics_data.get('feature_names', [])
        importance_list = []

        # If logistic regression, use abs coefficients
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            for name, val in zip(feature_names, coefs):
                importance_list.append({
                    'feature': name,
                    'coefficient': float(val),
                    'importance': float(abs(val))
                })
        elif hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            for name, val in zip(feature_names, imps):
                importance_list.append({
                    'feature': name,
                    'importance': float(val)
                })

        # Sort descending by importance
        importance_list.sort(key=lambda x: x['importance'], reverse=True)

        return jsonify({
            'best_model': best_model_name,
            'features': importance_list,
            'total_features': len(importance_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict heart disease risk for a single patient.
    Expects JSON body matching clinical features.
    """
    if not request.is_json:
        return jsonify({
            'error': 'Request payload must be JSON',
            'disclaimer': DISCLAIMER
        }), 400

    data = request.get_json()
    model, preprocessor = get_model_and_pipeline()

    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Machine learning model is not initialized or trained.',
            'disclaimer': DISCLAIMER
        }), 503

    result = predict_single(data, model=model, preprocessor=preprocessor)

    if result.get('errors'):
        return jsonify({
            'error': 'Input validation failed',
            'details': result['errors'],
            'disclaimer': DISCLAIMER
        }), 422

    return jsonify(result)


@api_bp.route('/predict-csv', methods=['POST'])
def predict_csv():
    """
    Batch prediction on uploaded CSV file.
    Returns list of patient predictions and aggregate risk distribution.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv file'}), 400

    try:
        df = pd.read_csv(file)
        # Check required columns
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            return jsonify({
                'error': f"CSV is missing required feature columns: {missing}",
                'required_columns': FEATURE_COLUMNS
            }), 422

        model, preprocessor = get_model_and_pipeline()
        predictions_df = predict_batch(df, model=model, preprocessor=preprocessor)

        # Risk distribution summary
        risk_dist = predictions_df['risk_category'].value_counts().to_dict()
        pred_dist = predictions_df['prediction_label'].value_counts().to_dict()

        records = predictions_df.to_dict(orient='records')
        # Limit returned records to avoid massive payloads if file is large
        max_records = 200

        return jsonify({
            'total_records': len(df),
            'risk_distribution': risk_dist,
            'prediction_distribution': pred_dist,
            'mean_probability': float(predictions_df['probability'].mean()) if 'probability' in predictions_df else None,
            'records': records[:max_records],
            'truncated': len(df) > max_records,
            'disclaimer': DISCLAIMER
        })

    except Exception as e:
        return jsonify({'error': f"Failed to process CSV: {str(e)}"}), 500
