# src/data/data_fetcher.py
import yfinance as yf
import logging
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol between start_date and end_date.
    Flattens multi-index columns (if any) and drops duplicates.
    Ensures the datetime index has a timezone.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logging.warning(f"No data fetched for ticker: {ticker}")
        
        # Flatten columns if they are a MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Drop duplicate columns if any exist (e.g., duplicate 'Close')
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Ensure the datetime index has timezone information
        if data.index.tz is None:
            data.index = data.index.tz_localize("US/Eastern")
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        data = None
    return data
