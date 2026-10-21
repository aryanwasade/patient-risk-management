"""
test_api.py -- Integration tests for Flask API endpoints.
"""

import io
import json
import unittest
from backend.app import create_app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for backend Flask API endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        """GET /api/health returns 200 OK and status."""
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))

    def test_dataset_info_endpoint(self):
        """GET /api/dataset-info returns dataset metadata."""
        res = self.client.get('/api/dataset-info')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('total_records', data)
        self.assertIn('features', data)
        self.assertIn('sample_records', data)

    def test_model_info_endpoint(self):
        """GET /api/model-info returns cross-validation and test metrics."""
        res = self.client.get('/api/model-info')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('best_model', data)
        self.assertIn('cv_results', data)
        self.assertIn('test_results', data)

    def test_eda_endpoint(self):
        """GET /api/eda returns available plots and metadata."""
        res = self.client.get('/api/eda')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('plots', data)
        self.assertIsInstance(data['plots'], list)
        self.assertGreater(len(data['plots']), 0)

    def test_feature_importance_endpoint(self):
        """GET /api/feature-importance returns ranked features."""
        res = self.client.get('/api/feature-importance')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('features', data)
        self.assertGreater(len(data['features']), 0)

    def test_predict_endpoint_valid(self):
        """POST /api/predict with valid patient JSON."""
        patient = {
            'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3
        }
        res = self.client.post('/api/predict',
                                data=json.dumps(patient),
                                content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('prediction', data)
        self.assertIn('probability', data)
        self.assertIn('risk', data)
        self.assertIn('disclaimer', data)

    def test_predict_endpoint_invalid_input(self):
        """POST /api/predict with missing required fields returns 422."""
        invalid_patient = {'age': 55}
        res = self.client.post('/api/predict',
                                data=json.dumps(invalid_patient),
                                content_type='application/json')
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertIn('error', data)

    def test_predict_csv_endpoint(self):
        """POST /api/predict-csv with uploaded CSV file."""
        csv_content = (
            "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n"
            "52,1,1,125,212,0,1,168,0,1.0,2,2,3\n"
            "53,1,1,140,203,1,0,155,1,3.1,0,0,7\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'patients.csv')
        }
        res = self.client.post('/api/predict-csv',
                                data=data,
                                content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        res_data = res.get_json()
        self.assertEqual(res_data.get('total_records'), 2)
        self.assertIn('risk_distribution', res_data)
        self.assertEqual(len(res_data['records']), 2)


if __name__ == '__main__':
    unittest.main()
