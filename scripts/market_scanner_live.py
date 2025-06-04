import argparse
import concurrent.futures
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from rich.progress import Progress
from sklearn.metrics import mean_squared_error

from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.lstm import prepare_lstm_data, train_lstm_model
from src.models.lstm_tuner import tune_lstm_hyperparameters
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost
from src.utils.logger import logger


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] Elapsed time: {end - start:.2f} seconds")


def load_ticker_list(filepath: str = "data/raw/market_tickers.csv") -> List[str]:
    """Loads a list of stock tickers from a CSV file.

    Args:
        filepath (str, optional): The file path of the CSV containing tickers. Defaults to "data/raw/market_tickers.csv".

    Returns:
        List[str]: A list of stock ticker symbols.
    """
    df = pd.read_csv(filepath)
    return df["Ticker"].tolist()


def analyze_stock(
    ticker: str,
    start_date: str,
    end_date: str,
    model_type: str,
    horizon: int,
    progress: Progress,
) -> Optional[Dict[str, Any]]:
    """
    Analyzes stock data for a given ticker using the same logic as main.py:
    - Fetch/load data
    - Create features/target
    - If LSTM, do hyperparameter tuning, train with best HP, and optionally do multiple runs
    - Compute and return metrics
    """
    try:
        progress.console.log(f"[{ticker}] Starting analysis.")
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.error(f"[{ticker}] No data available.")
            return None

        progress.console.log(f"[{ticker}] Data fetched ({len(data)} rows).")

        features, target = create_features_targets(data, horizon=horizon)
        progress.console.log(
            f"[{ticker}] Created features/target for horizon={horizon}."
        )

        # Decide how to handle models
        if model_type.lower() in ["rf", "xgb"]:
            # For example, re-use your existing train_random_forest() or train_xgboost() from main.py
            if model_type.lower() == "rf":
                (
                    model,
                    X_test,
                    y_test,
                    predictions,
                    mse_val,
                    sharpe_ratio,
                ) = train_random_forest(features, target)
            else:
                (
                    model,
                    X_test,
                    y_test,
                    predictions,
                    mse_val,
                    sharpe_ratio,
                ) = train_xgboost(features, target)

            # Suppose we just do last predictions from X_test
            # (You could do a “forward forecast” if you want.)
            last_return = predictions[-1]
            risk_adjusted_return = last_return / (X_test["Volatility"].mean() or 1.0)

        elif model_type.lower() == "lstm":
            # Recreate the logic from main.py
            sequence_length = 60
            close_series = features["Close"]
            X_seq, y_seq, _ = prepare_lstm_data(close_series, sequence_length)

            train_size = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]

            # 1) Hyperparameter Tuning
            best_model, best_hps = tune_lstm_hyperparameters(
                X_train.to("cuda"), y_train.to("cuda"), data, progress=progress
            )
            progress.console.log(f"[{ticker}] Best LSTM Hyperparams: {best_hps}")

            # 2) Train LSTM with best hyperparams
            best_model = train_lstm_model(
                best_model,
                X_train.to("cuda"),
                y_train.to("cuda"),
                epochs=100,
                batch_size=best_hps["batch_size"],
                progress=progress,
                ticker=ticker,
            )

            # 3) Evaluate model MSE on test set
            best_model.eval()
            with torch.no_grad():
                preds = best_model(X_test.to("cuda")).squeeze().cpu().numpy()
            mse_val = mean_squared_error(y_test.cpu().numpy(), preds)

            # For "risk" or predicted return, you can do single-step or multiple runs
            # For demonstration, let's do a single-step forecast (like main.py)
            last_return = float(preds[-1])  # last test prediction
            risk_adjusted_return = last_return  # or scale by volatility, if you like

        else:
            logger.error(f"[{ticker}] Unknown model type: {model_type}")
            return None

        progress.console.log(
            f"[{ticker}] MSE: {mse_val:.6f}, Risk-Adjusted Return: {risk_adjusted_return:.4f}"
        )
        return {
            "Ticker": ticker,
            "MSE": mse_val,
            "AvgPredReturn": float(last_return),
            "Risk_Adjusted_Return": float(risk_adjusted_return),
        }

    except Exception as e:
        logger.error(f"[{ticker}] Error during analysis: {e}")
        return None


def main(args: Any) -> None:
    """Runs the live market scanner with the 'main.py' style logic (tuning, training, etc.)."""
    with timer("Market Scanner"):
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

        tickers = load_ticker_list("data/raw/market_tickers.csv")
        logger.info(
            f"Scanning {len(tickers)} tickers from {start_date} to {end_date} using model: {args.model_type.upper()}."
        )

        results: List[Dict[str, Any]] = []
        with Progress() as progress:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        analyze_stock,
                        ticker,
                        start_date,
                        end_date,
                        args.model_type,
                        args.horizon,
                        progress,
                    ): ticker
                    for ticker in tickers
                }
                task = progress.add_task("[blue]Scanning Market...", total=len(futures))
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=180)
                    except Exception as e:
                        logger.error(
                            f"Ticker {futures[future]} timed out or errored: {e}"
                        )
                        continue
                    if result is not None:
                        results.append(result)
                    progress.advance(task)

        if not results:
            logger.error("No valid results obtained. Exiting...")
            return

        summary = pd.DataFrame(results).sort_values(
            by="Risk_Adjusted_Return", ascending=False
        )
        logger.info("Market scanning complete. Summary of results:")
        logger.info(summary.head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Market Scanner")
    parser.add_argument(
        "--model_type", type=str, default="lstm", help="Model type: rf, xgb, or lstm"
    )
    parser.add_argument(
        "--horizon", type=int, default=20, help="Forecast horizon (days)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of concurrent workers",
    )
    args = parser.parse_args()
    main(args)
