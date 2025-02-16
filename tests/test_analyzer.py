import unittest
from src.analysis.analyzer import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = DataAnalyzer()
        self.test_data = [1, 2, 3, 4, 5]  # Sample data for testing

    def test_analyze(self):
        self.analyzer.analyze(self.test_data)
        results = self.analyzer.get_results()
        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)  # Assuming results are returned as a dictionary

    def test_analyze_empty_data(self):
        self.analyzer.analyze([])
        results = self.analyzer.get_results()
        self.assertEqual(results, {})  # Assuming empty data returns an empty result

if __name__ == '__main__':
    unittest.main()