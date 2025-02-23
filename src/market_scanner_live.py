import logging
import pandas as pd
import concurrent.futures
from rich.progress import Progress
from datetime import datetime, timedelta
import colorlog
from typing import List, Optional, Dict, Any

from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost

# For LSTM, we import once here.
from src.models.lstm import prepare_lstm_data, build_lstm_model, train_lstm_model
from sklearn.metrics import mean_squared_error

# Configure colored logging with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def analyze_stock(ticker: str, start_date: str, end_date: str, 
                  model_type: str = "rf", horizon: int = 5) -> Optional[Dict[str, Any]]:
    """
    Processes one ticker:
      - Fetches and preprocesses stock data.
      - Trains a regression model (RandomForest, XGBoost, or LSTM).
      - Predicts multi-day returns and computes risk-adjusted return.
      - Returns key metrics in a dictionary.
    """
    try:
        logger.info(f"Processing {ticker}... (start)")
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.error(f"No data for {ticker}. Skipping...")
            return None
        logger.info(f"{ticker} - Data fetched, {len(data)} rows.")

        features, target = create_features_targets(data, horizon=horizon)
        logger.info(f"{ticker} - Features and target created.")

        if model_type.lower() == "rf":
            model, X_test, y_test, predictions, mse = train_random_forest(features, target)
            # X_test is a DataFrame
            test_df = X_test.copy()
            avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

        elif model_type.lower() == "xgb":
            model, X_test, y_test, predictions, mse = train_xgboost(features, target)
            test_df = X_test.copy()
            avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

        elif model_type.lower() == "lstm":
            # For LSTM, we use the 'Close' column as a univariate series.
            sequence_length = 60  # This can be parameterized further.
            close_series = features['Close']
            X_seq, y_seq, scaler = prepare_lstm_data(close_series, sequence_length=sequence_length)
            train_size = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]
            model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32)
            predictions = model.predict(X_test).flatten()
            mse = mean_squared_error(y_test, predictions)
            # For LSTM, X_test is a numpy array; extract last element of each sequence as current 'Close'
            close_values = X_test[:, -1, 0]
            test_df = pd.DataFrame({'Close': close_values})
            # As y_test might be 2D, flatten it.
            y_test = y_test.flatten()
            test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
            test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
            # Without additional volatility features, set default volatility
            avg_volatility = 1.0
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

        # For RF and XGB, flatten predictions to ensure 1D array.
        if model_type.lower() in ["rf", "xgb"]:
            predictions = predictions.flatten()
            test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
            test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)

        logger.info(f"{ticker} - Model MSE: {mse}")

        avg_predicted_return = predictions.mean()
        last_predicted_return = predictions[-1]
        risk_adjusted_return = last_predicted_return / avg_volatility if avg_volatility != 0 else 0

        logger.info(f"{ticker} - Analysis complete.")
        return {
            "ticker": ticker,
            "mse": mse,
            "avg_predicted_return": avg_predicted_return,
            "last_predicted_return": last_predicted_return,
            "risk_adjusted_return": risk_adjusted_return,
            "dates": data.iloc[-len(test_df):].index,  # Use last n rows from original data
            "predicted_prices": test_df['Predicted_Price'],
            "actual_prices": test_df['Actual_Price']
        }
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None


def load_ticker_list(filepath: str = "data/raw/market_tickers.csv") -> List[str]:
    """
    Loads a list of tickers from a CSV file with a column named 'Ticker'.
    """
    df = pd.read_csv(filepath)
    return df['Ticker'].tolist()


def main() -> None:
    # Define dynamic date range: last 10 years until today
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    
    tickers = load_ticker_list("data/raw/market_tickers.csv")
    results: List[Dict[str, Any]] = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_stock, ticker, start_date, end_date, model_type="lstm", horizon=20): ticker
            for ticker in tickers
        }
        with Progress() as progress:
            task = progress.add_task("[blue]Scanning Market...", total=len(futures))
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=180)
                except Exception as e:
                    logger.error(f"Ticker {futures[future]} timed out or errored: {e}")
                    continue
                if result is not None:
                    results.append(result)
                progress.advance(task)
    
    if not results:
        logger.error("No valid results. Exiting...")
        return

    summary = pd.DataFrame({
        "Ticker": [res["ticker"] for res in results],
        "MSE": [res["mse"] for res in results],
        "Avg_Pred_Return": [res["avg_predicted_return"] for res in results],
        "Last_Pred_Return": [res["last_predicted_return"] for res in results],
        "Risk_Adjusted_Return": [res["risk_adjusted_return"] for res in results]
    })

    # Rank stocks by risk-adjusted return (higher is better)
    filtered = summary.sort_values(by="Risk_Adjusted_Return", ascending=False)
    
    print("Summary of model performance and predicted returns:")
    print(summary)
    print("\nTop stocks based on risk-adjusted predicted return:")
    print(filtered.head(5))


if __name__ == "__main__":
    main()
