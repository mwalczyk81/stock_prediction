import pandas as pd
import yfinance as yf

from typing import Optional
from src.logger import logger

def fetch_stock_data(ticker: str, start_date: str, end_date: str, save_to_csv: bool = False, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch historical stock data for a given ticker symbol.

    This function retrieves historical stock data from Yahoo Finance between `start_date` and `end_date`.
    It processes the data by:
    - Flattening multi-index columns (if present).
    - Removing duplicate columns.
    - Ensuring the datetime index is timezone-aware.
    - Optionally saving the data to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple Inc.).
        start_date (str): The start date for data retrieval in "YYYY-MM-DD" format.
        end_date (str): The end date for data retrieval in "YYYY-MM-DD" format.
        save_to_csv (bool, optional): If True, saves the data to a CSV file. Defaults to False.
        filename (Optional[str], optional): The custom filename for saving the CSV. Defaults to None.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing historical stock prices, or None if fetching fails.
    """
    try:
        logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Check if data is empty
        if data.empty:
            logger.warning(f"No data fetched for ticker: {ticker}")
            return None

        # Flatten multi-index columns if present (e.g., "Adj Close" appearing twice)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Remove duplicate columns (if any exist)
        data = data.loc[:, ~data.columns.duplicated()]

        # Ensure the datetime index has timezone information
        if data.index.tz is None:
            data.index = data.index.tz_localize("US/Eastern")

        # Save data to CSV if required
        if save_to_csv:
            filename = f"data/raw/{filename}" if filename else f"data/raw/{ticker}_stock_data.csv"
            data.to_csv(filename)
            logger.info(f"Stock data saved to {filename}")

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None
