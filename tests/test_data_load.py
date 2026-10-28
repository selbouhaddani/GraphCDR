"""Original hickle features load quietly without hiding unrelated problems."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import hickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'prog'))
from data_load import load_drug_graph


class FeatureLoadingTests(unittest.TestCase):
    def test_original_archive_loads_without_changing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'original.hkl'
            with zipfile.ZipFile(ROOT / 'data' / 'Drug' / 'drug_graph_feat.zip') as archive:
                member = next(name for name in archive.namelist() if name.endswith('.hkl'))
                path.write_bytes(archive.read(member))
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                actual = load_drug_graph(str(path))
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                expected = hickle.load(str(path))
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(len(actual[1]), len(expected[1]))
            for actual_neighbors, expected_neighbors in zip(actual[1], expected[1]):
                np.testing.assert_array_equal(actual_neighbors, expected_neighbors)
            np.testing.assert_array_equal(actual[2], expected[2])

    def test_other_warnings_and_errors_remain_visible(self):
        def noisy_load(path):
            warnings.warn('Unexpected feature-file problem', UserWarning)
            return 'loaded'

        with patch('data_load.hkl.load', side_effect=noisy_load):
            with self.assertWarnsRegex(UserWarning, 'Unexpected feature-file problem'):
                self.assertEqual(load_drug_graph('example.hkl'), 'loaded')
        with patch('data_load.hkl.load', side_effect=ValueError('Invalid feature file')):
            with self.assertRaisesRegex(ValueError, 'Invalid feature file'):
                load_drug_graph('example.hkl')


if __name__ == '__main__':
    unittest.main()
