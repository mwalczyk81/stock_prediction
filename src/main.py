import argparse
import logging
import os
import pandas as pd
import concurrent.futures
from rich.progress import Progress
from datetime import datetime, timedelta
import colorlog
from typing import Any

from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost
from src.utils.visualization import plot_predictions
from src.forecasting import forecast_one_step, forecast_multi_step

from src.models.lstm import prepare_lstm_data, build_lstm_model, train_lstm_model
from sklearn.metrics import mean_squared_error

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def main(args: Any) -> None:
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date
    horizon = args.horizon
    model_type = args.model_type
    forecast_steps = args.forecast_steps

    logger.info(f"Processing {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None or data.empty:
        logger.error("No data fetched. Exiting...")
        return

    logger.info("Creating features and target...")
    features, target = create_features_targets(data, horizon=horizon)
    
    logger.info(f"Training {model_type} model...")
    if model_type.lower() == "rf":
        model, X_test, y_test, predictions, mse = train_random_forest(features, target)
        predictions = predictions.flatten()
        test_df = X_test.copy()
        test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
        test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
        avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

    elif model_type.lower() == "xgb":
        model, X_test, y_test, predictions, mse = train_xgboost(features, target)
        predictions = predictions.flatten()
        test_df = X_test.copy()
        test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
        test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
        avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0

    elif model_type.lower() == "lstm":
        # For LSTM, use the 'Close' column to prepare a sequence.
        sequence_length = 60  # Must match the model input shape
        # Instead of using create_features_targets, use the raw price data.
        # Note: You might still use your engineered features for training,
        # but for forecasting, you need a sequence.
        # Here, we assume 'data' already contains the historical prices.
        
        # Build and train the LSTM model as before:
        close_series = features['Close']
        X_seq, y_seq, scaler = prepare_lstm_data(close_series, sequence_length=sequence_length)
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32)
        predictions = model.predict(X_test).flatten()  # Evaluate on historical test set for MSE
        mse = mean_squared_error(y_test, predictions)
        
        # For forward forecasting, call our new function:
        from src.forecasting import forecast_one_step_lstm
        predicted_return, predicted_price = forecast_one_step_lstm(model, data, sequence_length=sequence_length, horizon=horizon)
        
        logger.info(f"Forward Forecast for {ticker}: Predicted return over {horizon} days = {predicted_return:.4f}, Predicted price = {predicted_price:.2f}")
        
        # For visualization purposes, you can still reconstruct a test_df for historical evaluation:
        close_values = X_test[:, -1, 0]
        test_df = pd.DataFrame({'Close': close_values})
        y_test = y_test.flatten()
        test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
        test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
        avg_volatility = 1.0  # Default or computed from another method
    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    logger.info(f"{ticker} - Model MSE: {mse}")
    avg_predicted_return = predictions.mean()
    last_predicted_return = predictions[-1]
    risk_adjusted_return = last_predicted_return / avg_volatility if avg_volatility != 0 else 0

    # If forecast_steps is specified, perform forward forecasting.
    if forecast_steps > 0:
        from src.forecasting import forecast_one_step_lstm, forecast_multi_step
        if model_type.lower() == "lstm":
            # Call the dedicated LSTM forecasting function.
            pred_return, pred_price = forecast_one_step_lstm(model, data, sequence_length=60, horizon=horizon)
            logger.info(f"Forward Forecast for {ticker}: Predicted return over {horizon} days = {pred_return:.4f}, Predicted price = {pred_price:.2f}")
        else:
            if forecast_steps == 1:
                pred_return, pred_price = forecast_one_step(model, data, horizon)
                logger.info(f"One-step forecast: Predicted {horizon}-day return: {pred_return:.4f}, Predicted price: {pred_price:.2f}")
            else:
                forecasts = forecast_multi_step(model, data, horizon, forecast_steps)
                logger.info("Multi-step forecasts:")
                for forecast in forecasts:
                    logger.info(f"Step {forecast['step']}: Predicted return: {forecast['predicted_return']:.4f}, Predicted price: {forecast['predicted_price']:.2f}")

    else:
        # Otherwise, plot historical test predictions.
        test_dates = data.iloc[-len(test_df):].index
        logger.info("Plotting predictions...")
        plot_predictions(test_dates, test_df['Actual_Price'], test_df['Predicted_Price'],
                         title=f"{ticker} Stock Price Prediction")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock prediction for a single stock with optional forward forecasting.")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Ticker symbol of the stock (default: AAPL)")
    parser.add_argument("--start_date", type=str, default=(datetime.today() - timedelta(days=365*10)).strftime("%Y-%m-%d"),
                        help="Start date for historical data (default: 10 years ago)")
    parser.add_argument("--end_date", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="End date for historical data (default: today's date)")
    parser.add_argument("--horizon", type=int, default=20,
                        help="Prediction horizon in days (default: 20)")
    parser.add_argument("--model_type", type=str, default="rf",
                        help="Model type to use (rf, xgb, or lstm; default: rf)")
    parser.add_argument("--forecast_steps", type=int, default=0,
                        help="Number of future steps to forecast (0 means no forward forecast)")
    args = parser.parse_args()
    main(args)
