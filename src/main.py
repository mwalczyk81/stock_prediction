import argparse
import logging
import os
import pandas as pd
import torch
import sys

from rich.console import Console
from rich.logging import RichHandler
from datetime import datetime, timedelta
from rich.progress import Progress
from typing import Any, Tuple
from src.logger import logger
from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost
from src.models.lstm import prepare_lstm_data, train_lstm_model
from src.models.lstm_tuner import tune_lstm_hyperparameters
from src.forecasting import forecast_one_step, forecast_multi_step, forecast_one_step_lstm
from sklearn.metrics import mean_squared_error


progress_console = Console(file=sys.stdout)

# **✅ Function to Load or Fetch Stock Data**
def load_or_fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Loads stock data from CSV if available, otherwise fetches it."""
    csv_file = f"data/raw/{ticker}_stock_data.csv"

    if os.path.exists(csv_file):
        logger.info(f"Loading stock data from CSV: {csv_file}")
        return pd.read_csv(csv_file, index_col=0, parse_dates=True)

    logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    return fetch_stock_data(ticker, start_date, end_date, save_to_csv=True)


# **✅ Function to Train the Model with Progress Bar**
def train_model(model_type: str, features: pd.DataFrame, target: pd.Series, data, progress: Progress) -> Tuple[Any, pd.DataFrame, pd.Series, Any, float]:
    """Trains the selected model (Random Forest, XGBoost, or LSTM) with a progress bar."""

    if model_type.lower() == "rf":
        logger.info("Training Random Forest model...")
        model, X_test, y_test, predictions, mse = train_random_forest(features, target)

    elif model_type.lower() == "xgb":
        logger.info("Training XGBoost model...")
        model, X_test, y_test, predictions, mse = train_xgboost(features, target)

    elif model_type.lower() == "lstm":
        logger.info("Preparing LSTM data...")
        sequence_length = 60
        close_series = features['Close']
        X_seq, y_seq, scaler = prepare_lstm_data(close_series, sequence_length=sequence_length)
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        logger.info("Tuning LSTM hyperparameters...")
        
        best_model, best_hps = tune_lstm_hyperparameters(
            X_train.to('cuda'), y_train.to('cuda'), data, progress=progress
        )
        
        logger.info(f"Best LSTM Hyperparameters: {best_hps}")

        logger.info("Training LSTM model with best hyperparameters...")
        
        best_model = train_lstm_model(
            best_model, X_train.to('cuda'), y_train.to('cuda'),
            epochs=100, batch_size=best_hps['batch_size'], progress=progress
        )

        best_model.eval()
        with torch.no_grad():
            predictions = best_model(X_test.to('cuda')).squeeze().cpu().numpy()

        mse = mean_squared_error(y_test.cpu(), predictions)
        return best_model, X_test, y_test, predictions, mse

    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, X_test, y_test, predictions, mse


# **✅ Main Function with Improved Progress Bar Handling**
def main(args: Any) -> None:
    """Main function to run stock prediction and forecasting with proper progress bar handling."""
    ticker, start_date, end_date, horizon, model_type, forecast_steps = args.ticker, args.start_date, args.end_date, args.horizon, args.model_type, args.forecast_steps

    logger.info(f"Starting stock prediction for {ticker}, model: {model_type.upper()}, horizon: {horizon} days.")

    with Progress(console=progress_console, transient=True) as progress:
        # **Task 1: Fetching Stock Data**
        data = load_or_fetch_data(ticker, start_date, end_date)
        if data.empty:
            logger.error("No data fetched or loaded. Aborting.")
            return

        # **Task 2: Creating Features and Target**
        features, target = create_features_targets(data, horizon=horizon)

        # **Task 3: Training Model**
        model, X_test, y_test, predictions, mse = train_model(model_type, features, target, data, progress)

    logger.info(f"{ticker} - {model_type.upper()} Model Training Completed. MSE: {mse:.6f}")
    current_price = data['Close'].iloc[-1]

    # **Forecasting Logic**
    if forecast_steps > 0:
        logger.info(f"Performing {forecast_steps}-step forward forecast...")
        if model_type.lower() == "lstm":
            pred_return, pred_price = forecast_one_step_lstm(model, data, features['Close'], sequence_length=60, horizon=horizon)
        else:
            if forecast_steps == 1:
                pred_return, pred_price = forecast_one_step(model, data, horizon)
            else:
                forecasts = forecast_multi_step(model, data, horizon, forecast_steps)
                for forecast in forecasts:
                    logger.info(f"Step {forecast['step']}: Predicted return: {forecast['predicted_return']:.4f}, Predicted price: {forecast['predicted_price']:.2f}")
                return
    else:
        pred_return, pred_price = forecast_one_step_lstm(model, data, features['Close'], sequence_length=60, horizon=horizon) if model_type.lower() == "lstm" else forecast_one_step(model, data, horizon)

    logger.info(f"Forward Forecast for {ticker} ({model_type.upper()}): Current price = {current_price:.2f}, Predicted return over {horizon} days = {pred_return:.4f}, Predicted price = {pred_price:.2f}")


# **✅ Entry Point**
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction with optional forecasting.")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start_date", type=str, default=(datetime.today() - timedelta(days=365*10)).strftime("%Y-%m-%d"))
    parser.add_argument("--end_date", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="rf")
    parser.add_argument("--forecast_steps", type=int, default=0)

    args = parser.parse_args()
    main(args)
