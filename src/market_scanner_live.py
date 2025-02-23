import argparse
import concurrent.futures
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from rich.console import Console
from rich.progress import Progress

from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost
from src.models.lstm import prepare_lstm_data, LSTMPredictor, train_lstm_model
from sklearn.metrics import mean_squared_error
from src.logger import logger

# Use a Rich Console for progress output
progress_console = Console(file=sys.stdout)


def load_ticker_list(filepath: str = "data/raw/market_tickers.csv") -> List[str]:
    """
    Loads a list of tickers from a CSV file with a column named 'Ticker'.
    """
    df = pd.read_csv(filepath)
    return df['Ticker'].tolist()


def analyze_stock(ticker: str, start_date: str, end_date: str, 
                  model_type: str, horizon: int, progress: Progress) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single ticker by:
      - Fetching and preprocessing its stock data.
      - Training a regression model (RandomForest, XGBoost, or LSTM).
      - Computing model error and predicted returns.
      - Returning key metrics as a dictionary.
    """
    try:
        logger.info(f"[{ticker}] Starting analysis.")
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.error(f"[{ticker}] No data available.")
            return None

        logger.info(f"[{ticker}] Data fetched ({len(data)} rows).")
        features, target = create_features_targets(data, horizon=horizon)
        logger.info(f"[{ticker}] Features and target created.")

        if model_type.lower() == "rf":
            model, X_test, y_test, predictions, mse = train_random_forest(features, target)
            test_df = X_test.copy()
            avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

        elif model_type.lower() == "xgb":
            model, X_test, y_test, predictions, mse = train_xgboost(features, target)
            test_df = X_test.copy()
            avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

        elif model_type.lower() == "lstm":
            sequence_length = 60
            close_series = features['Close']
            X_seq, y_seq, scaler = prepare_lstm_data(close_series, sequence_length=sequence_length)
            train_size = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]
            # Create LSTM model instance
            model = LSTMPredictor(input_size=1, hidden_size=100, num_layers=2, dropout=0.2)
            # Train the model; use a shared progress bar if desired
            train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, progress=progress, ticker=ticker)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            with torch.no_grad():
                predictions = model(X_test.to(device)).cpu().numpy().flatten()
            mse = mean_squared_error(y_test.cpu().numpy(), predictions)
            # For LSTM, assume the last value of each input sequence represents the current price
            close_values = X_test[:, -1, 0].cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test[:, -1, 0]
            test_df = pd.DataFrame({'Close': close_values})
            test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
            test_df['Actual_Price'] = test_df['Close'] * (1 + (y_test.cpu().numpy().flatten() if isinstance(y_test, torch.Tensor) else y_test.flatten()))
            avg_volatility = 1.0

        else:
            logger.error(f"[{ticker}] Unknown model type: {model_type}")
            return None

        # For RF and XGB, ensure predictions are 1D and compute price columns.
        if model_type.lower() in ["rf", "xgb"]:
            predictions = predictions.flatten()
            test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
            test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)

        logger.info(f"[{ticker}] Model MSE: {mse:.6f}")
        avg_predicted_return = predictions.mean()
        last_predicted_return = predictions[-1]
        risk_adjusted_return = last_predicted_return / avg_volatility if avg_volatility != 0 else 0

        logger.info(f"[{ticker}] Analysis complete.")
        return {
            "ticker": ticker,
            "mse": mse,
            "avg_predicted_return": avg_predicted_return,
            "last_predicted_return": last_predicted_return,
            "risk_adjusted_return": risk_adjusted_return,
            "dates": data.iloc[-len(test_df):].index,
            "predicted_prices": test_df['Predicted_Price'],
            "actual_prices": test_df['Actual_Price']
        }
    except Exception as e:
        logger.error(f"[{ticker}] Error during analysis: {e}")
        return None


def main(args: Any) -> None:
    """
    Main function to run the live market scanner.
    It loads a list of tickers, sets a dynamic date range,
    and processes each ticker concurrently.
    """
    # Set date range: last 10 years until today.
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    
    tickers = load_ticker_list("data/raw/market_tickers.csv")
    logger.info(f"Scanning {len(tickers)} tickers from {start_date} to {end_date} using model: {args.model_type.upper()}.")

    results: List[Dict[str, Any]] = []
    with Progress(console=progress_console, transient=True) as progress:    
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
            executor.submit(analyze_stock, ticker, start_date, end_date, args.model_type, args.horizon, progress): ticker
            for ticker in tickers
        }
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
        logger.error("No valid results obtained. Exiting...")
        return

    summary = pd.DataFrame({
        "Ticker": [res["ticker"] for res in results],
        "MSE": [res["mse"] for res in results],
        "Avg_Pred_Return": [res["avg_predicted_return"] for res in results],
        "Last_Pred_Return": [res["last_predicted_return"] for res in results],
        "Risk_Adjusted_Return": [res["risk_adjusted_return"] for res in results]
    })
    filtered = summary.sort_values(by="Risk_Adjusted_Return", ascending=False)

    logger.info("Market scanning complete. Summary of results:")
    print(summary)
    print("\nTop stocks based on risk-adjusted predicted return:")
    print(filtered.head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Market Scanner")
    parser.add_argument("--model_type", type=str, default="lstm", help="Model type: rf, xgb, or lstm")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon (days)")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of concurrent workers")
    args = parser.parse_args()
    main(args)
