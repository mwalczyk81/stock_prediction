import unittest
import pandas as pd
from src.data.preprocessing import add_technical_indicators, create_features_targets

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataFrame to mimic stock data
        data = {
            "Open": [100, 102, 101, 103, 104, 105, 106, 107, 108, 109],
            "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "Low": [99, 100, 98, 101, 102, 103, 104, 105, 106, 107],
            "Close": [104, 105, 103, 107, 108, 109, 110, 111, 112, 113],
            "Volume": [1000, 1100, 1050, 1150, 1200, 1250, 1300, 1350, 1400, 1450]
        }
        self.df = pd.DataFrame(data)

    def test_add_technical_indicators(self):
        df_ind = add_technical_indicators(self.df)
        # Verify that indicator columns have been added
        self.assertIn('SMA_20', df_ind.columns)
        self.assertIn('EMA_20', df_ind.columns)
        self.assertIn('RSI', df_ind.columns)

    def test_create_features_targets(self):
        df_ind = add_technical_indicators(self.df)
        features, target = create_features_targets(df_ind)
        self.assertEqual(len(features), len(target))

if __name__ == '__main__':
    unittest.main()
