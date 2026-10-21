"""
test_prediction.py -- Unit tests for model prediction, validation, and risk categorization.
"""

import unittest
import pandas as pd
from src.predict import (
    load_model_and_pipeline,
    validate_patient_input,
    categorize_risk,
    predict_single,
    predict_batch,
    RISK_THRESHOLDS,
)
from src.data_loader import load_and_validate


class TestPredictionModule(unittest.TestCase):
    """Test suite for prediction logic and risk categorization."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.preprocessor = load_model_and_pipeline()
        cls.valid_patient = {
            'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3
        }

    def test_risk_categorization_thresholds(self):
        """Test risk category boundaries."""
        low_res = categorize_risk(0.20)
        self.assertEqual(low_res['category'], 'Low Risk')

        mod_res = categorize_risk(0.45)
        self.assertEqual(mod_res['category'], 'Moderate Risk')

        high_res = categorize_risk(0.85)
        self.assertEqual(high_res['category'], 'High Risk')

    def test_single_prediction_valid(self):
        """Test prediction output structure on valid patient."""
        res = predict_single(self.valid_patient, self.model, self.preprocessor)
        self.assertIn(res['prediction'], [0, 1])
        self.assertIn('probability', res)
        self.assertGreaterEqual(res['probability'], 0.0)
        self.assertLessEqual(res['probability'], 1.0)
        self.assertIn(res['risk']['category'], ['Low Risk', 'Moderate Risk', 'High Risk'])
        self.assertEqual(len(res['errors']), 0)
        self.assertIn('DISCLAIMER', res['disclaimer'])

    def test_single_prediction_missing_features(self):
        """Test input validation when required features are missing."""
        invalid_patient = {'age': 55, 'sex': 1}
        res = predict_single(invalid_patient, self.model, self.preprocessor)
        self.assertIsNone(res['prediction'])
        self.assertGreater(len(res['errors']), 0)

    def test_single_prediction_out_of_bounds(self):
        """Test input validation when feature is outside plausible clinical range."""
        invalid_patient = self.valid_patient.copy()
        invalid_patient['age'] = 250  # Out of bounds age
        cleaned, errors = validate_patient_input(invalid_patient)
        self.assertTrue(any('age' in err for err in errors))

    def test_batch_prediction(self):
        """Test batch prediction DataFrame manipulation."""
        df, _, _, _ = load_and_validate()
        sample_df = df.head(10)
        results_df = predict_batch(sample_df, self.model, self.preprocessor)

        self.assertIn('prediction', results_df.columns)
        self.assertIn('probability', results_df.columns)
        self.assertIn('risk_category', results_df.columns)
        self.assertEqual(len(results_df), 10)


if __name__ == '__main__':
    unittest.main()
