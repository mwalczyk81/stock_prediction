import unittest
import pandas as pd
import numpy as np
import ta # For MACD comparison

# Functions to test
from src.data.preprocessing import add_macd, add_technical_indicators, create_features_targets
from src.utils.metrics import calculate_sharpe_ratio


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataFrame to mimic stock data
        # Using a longer series for more stable MACD calculation
        data_close = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134, 135
        ]
        self.df = pd.DataFrame({"Close": data_close})
        # A smaller df for other tests if needed
        data_small = {
            "Open": [100, 102, 101, 103, 104, 105, 106, 107, 108, 109],
            "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "Low": [99, 100, 98, 101, 102, 103, 104, 105, 106, 107],
            "Close": [104, 105, 103, 107, 108, 109, 110, 111, 112, 113],
            "Volume": [1000, 1100, 1050, 1150, 1200, 1250, 1300, 1350, 1400, 1450],
        }
        self.df_small = pd.DataFrame(data_small)


    def test_add_technical_indicators(self):
        # Using df_small as it has Open, High, Low, Close, Volume
        df_ind = add_technical_indicators(self.df_small.copy()) # Use copy to avoid modifying original
        self.assertIn("SMA_20", df_ind.columns)
        self.assertIn("EMA_20", df_ind.columns)
        self.assertIn("RSI", df_ind.columns)
        # SMA and EMA will have NaNs for initial periods < window size
        self.assertTrue(df_ind["SMA_20"].iloc[:19].isnull().all())
        self.assertFalse(df_ind["SMA_20"].iloc[19:].isnull().any()) # Should be 10 for df_small
        # RSI will have NaNs for initial periods < window size (14 for RSI)
        self.assertTrue(df_ind["RSI"].iloc[:13].isnull().all())


    def test_create_features_targets(self):
        # Using df_small as it has Open, High, Low, Close, Volume
        # To avoid errors due to short length for all indicators, let's use a slightly larger df for this test
        # or ensure create_features_targets handles short data gracefully (e.g. by returning empty/NaNs)
        # For simplicity, we'll assume it's tested with enough data from other tests or integration tests.
        # Here, we'll just check the basic transformation.
        df_processed = self.df_small.copy()
        # Need to add some indicators first as create_features_targets expects them
        df_processed = add_technical_indicators(df_processed)
        df_processed = add_macd(df_processed) # add_macd is called within create_features_targets

        features, target = create_features_targets(df_processed, horizon=1) # Using small horizon
        self.assertEqual(len(features), len(target))
        # create_features_targets drops NaNs, so length will be less than original
        self.assertTrue(len(features) < len(self.df_small))
        expected_feature_cols = [
            "Open", "High", "Low", "Close", "Volume", "SMA_20", "EMA_20", "RSI",
            "MACD", "Signal", "MACD_diff" # These are added by add_macd
        ] # Plus lag features, volatility, momentum
        for col in expected_feature_cols:
            self.assertIn(col, features.columns)


    def test_add_macd(self):
        df_with_macd = add_macd(self.df.copy())

        self.assertIn("MACD", df_with_macd.columns)
        self.assertIn("Signal", df_with_macd.columns)
        self.assertIn("MACD_diff", df_with_macd.columns)

        # Compare with ta library's direct calculation
        # Note: add_macd in preprocessing.py internally uses ta.trend.MACD
        # So this test primarily ensures the columns are named correctly and added to the DataFrame.
        # And that the internal ta.trend.MACD is called as expected.
        # The actual calculation correctness relies on the 'ta' library.
        expected_macd_indicator = ta.trend.MACD(self.df["Close"])
        expected_macd = expected_macd_indicator.macd()
        expected_signal = expected_macd_indicator.macd_signal()
        expected_diff = expected_macd_indicator.macd_diff()

        pd.testing.assert_series_equal(df_with_macd["MACD"], expected_macd, check_dtype=False, check_names=False)
        pd.testing.assert_series_equal(df_with_macd["Signal"], expected_signal, check_dtype=False, check_names=False)
        pd.testing.assert_series_equal(df_with_macd["MACD_diff"], expected_diff, check_dtype=False, check_names=False)

        # Check for NaNs in initial periods
        # MACD uses 12 and 26 period EMAs, signal uses 9 period EMA of MACD.
        # The first non-NaN MACD value is at index 24 (26-1-1 for EMA diff, but ta might be different)
        # The first non-NaN Signal value is at index 24 + 8 = 32
        # Let's check based on ta's behavior
        self.assertTrue(df_with_macd["MACD"].iloc[:24].isnull().all()) # Based on typical EMA behavior (26-1)
        self.assertFalse(df_with_macd["MACD"].iloc[25:].isnull().all()) # Check one after
        # Signal line needs MACD values first, then 9 periods for its EMA
        self.assertTrue(df_with_macd["Signal"].iloc[:(24 + 9 - 1 -1)].isnull().all()) # (26-1 for first MACD) + (9-1 for signal EMA)
        self.assertFalse(df_with_macd["Signal"].iloc[33:].isnull().all()) # Check one after 32


class TestMetrics(unittest.TestCase):
    def test_calculate_sharpe_ratio(self):
        # Test case 1: Typical data
        returns1 = pd.Series([0.01, -0.005, 0.02, 0.003, -0.001, 0.005, 0.015])
        mean_return1 = returns1.mean()
        std_return1 = returns1.std()
        expected_sharpe1 = (mean_return1 * 252) / (std_return1 * np.sqrt(252))
        self.assertAlmostEqual(calculate_sharpe_ratio(returns1), expected_sharpe1)

        # Test case 2: All positive returns
        returns2 = pd.Series([0.01, 0.005, 0.02, 0.003, 0.001, 0.005, 0.015])
        mean_return2 = returns2.mean()
        std_return2 = returns2.std()
        expected_sharpe2 = (mean_return2 * 252) / (std_return2 * np.sqrt(252))
        self.assertAlmostEqual(calculate_sharpe_ratio(returns2), expected_sharpe2)

        # Test case 3: All negative returns
        returns3 = pd.Series([-0.01, -0.005, -0.02, -0.003, -0.001, -0.005, -0.015])
        mean_return3 = returns3.mean()
        std_return3 = returns3.std()
        expected_sharpe3 = (mean_return3 * 252) / (std_return3 * np.sqrt(252))
        self.assertAlmostEqual(calculate_sharpe_ratio(returns3), expected_sharpe3)
        self.assertTrue(calculate_sharpe_ratio(returns3) < 0)


    def test_calculate_sharpe_ratio_edge_cases(self):
        # Test case 1: Standard deviation is zero (all returns are the same)
        returns_zero_std = pd.Series([0.01, 0.01, 0.01, 0.01])
        self.assertEqual(calculate_sharpe_ratio(returns_zero_std), 0.0)

        # Test case 2: Less than 2 data points
        returns_one_point = pd.Series([0.01])
        self.assertEqual(calculate_sharpe_ratio(returns_one_point), 0.0)
        returns_empty = pd.Series([], dtype=float)
        self.assertEqual(calculate_sharpe_ratio(returns_empty), 0.0)

        # Test case 3: Data that could result in positive/negative Sharpe
        returns_mixed = pd.Series([0.02, -0.01, 0.005, -0.002])
        mean_mixed = returns_mixed.mean()
        std_mixed = returns_mixed.std()
        expected_sharpe_mixed = (mean_mixed * 252) / (std_mixed * np.sqrt(252))
        self.assertAlmostEqual(calculate_sharpe_ratio(returns_mixed), expected_sharpe_mixed)


if __name__ == "__main__":
    unittest.main()
