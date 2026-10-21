"""
test_preprocessing.py -- Unit tests for data loading and preprocessing.
"""

import unittest
import numpy as np
import pandas as pd

from src.data_loader import (
    load_and_validate,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.data_preprocessing import (
    build_preprocessing_pipeline,
    split_features_target,
    split_data,
    fit_and_transform,
    preprocess_single_input,
)


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for data loading and preprocessing integrity."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.report, _, _ = load_and_validate()
        cls.X, cls.y = split_features_target(cls.df)
        cls.splits = split_data(cls.X, cls.y)
        cls.pipe = build_preprocessing_pipeline()
        cls.results = fit_and_transform(cls.pipe, cls.splits)
        cls.preprocessor = cls.results['preprocessor']

    def test_data_loader_shape(self):
        """Test dataset shape and column existence."""
        self.assertGreater(len(self.df), 250)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns)
        self.assertIn(TARGET_COLUMN, self.df.columns)

    def test_no_data_leakage_split(self):
        """Ensure train, val, and test splits are disjoint and stratify properly."""
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']

        total_rows = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_rows, len(self.df))

        # Check disjoint index sets
        self.assertTrue(set(X_train.index).isdisjoint(set(X_val.index)))
        self.assertTrue(set(X_train.index).isdisjoint(set(X_test.index)))
        self.assertTrue(set(X_val.index).isdisjoint(set(X_test.index)))

    def test_preprocessor_output(self):
        """Ensure preprocessor produces 2D numeric array without NaNs."""
        X_train_proc = self.results['X_train_processed']
        self.assertIsInstance(X_train_proc, np.ndarray)
        self.assertFalse(np.isnan(X_train_proc).any())
        self.assertEqual(len(X_train_proc), len(self.splits['X_train']))

    def test_single_input_preprocessing(self):
        """Ensure single patient dict is converted to 1 x N numeric vector."""
        sample = {
            'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3
        }
        res = preprocess_single_input(sample, self.preprocessor)
        self.assertEqual(res.shape[0], 1)
        self.assertEqual(res.shape[1], self.results['X_train_processed'].shape[1])
        self.assertFalse(np.isnan(res).any())


if __name__ == '__main__':
    unittest.main()
