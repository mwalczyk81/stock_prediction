import pandas as pd
import numpy as np

def calculate_sharpe_ratio(predictions_series: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio from a series of predicted returns.

    Args:
        predictions_series (pd.Series): A pandas Series of predicted returns.
        trading_days_per_year (int): Number of trading days in a year for annualization.

    Returns:
        float: The annualized Sharpe Ratio. Returns 0.0 if standard deviation is zero
               or if there are fewer than 2 data points.
    """
    if len(predictions_series) < 2:
        return 0.0

    mean_daily_return = predictions_series.mean()
    std_daily_return = predictions_series.std()

    if std_daily_return == 0:
        return 0.0  # Avoid division by zero; Sharpe ratio is undefined or infinite

    # Annualize Sharpe Ratio
    sharpe_ratio = (mean_daily_return * trading_days_per_year) / (std_daily_return * np.sqrt(trading_days_per_year))
    return sharpe_ratio
