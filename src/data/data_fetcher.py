import yfinance as yf
import logging
import pandas as pd
from typing import Optional

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data for a given ticker symbol between start_date and end_date.
    Flattens multi-index columns and drops duplicates.
    Ensures the datetime index has a timezone.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logging.warning(f"No data fetched for ticker: {ticker}")
            return None

        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Remove duplicate columns (e.g., duplicate 'Close')
        data = data.loc[:, ~data.columns.duplicated()]

        # Ensure the datetime index has timezone information
        if data.index.tz is None:
            data.index = data.index.tz_localize("US/Eastern")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None
