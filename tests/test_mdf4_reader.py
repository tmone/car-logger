import unittest
from src.data_loader.mdf4_reader import MDF4Reader

class TestMDF4Reader(unittest.TestCase):

    def setUp(self):
        self.reader = MDF4Reader()

    def test_load_data(self):
        # Assuming a sample MDF4 file path for testing
        file_path = 'path/to/sample.mdf4'
        data = self.reader.load_data(file_path)
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)  # Assuming the data is returned as a dictionary

    def test_get_data(self):
        file_path = 'path/to/sample.mdf4'
        self.reader.load_data(file_path)
        data = self.reader.get_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

if __name__ == '__main__':
    unittest.main()