import logging
import colorlog
from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.utils.visualization import plot_predictions

# Configure colored logging
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

def main():
    # Set parameters for a single stock prediction
    ticker = "AAPL"  # Change to any ticker you want to analyze
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    horizon = 5  # Predict a 5-day return

    logger.info(f"Processing {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None or data.empty:
        logger.error("No data fetched. Exiting...")
        return

    logger.info("Creating features and target...")
    # Use the updated create_features_targets that accepts a horizon parameter
    features, target = create_features_targets(data, horizon=horizon)
    
    logger.info("Training RandomForest model...")
    model, X_test, y_test, predictions, mse = train_random_forest(features, target)
    logger.info(f"Model MSE: {mse}")

    # Reconstruct predicted prices using today's close and predicted return
    test_df = X_test.copy()
    test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
    test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
    test_dates = data.iloc[-len(X_test):].index

    # Compute risk-adjusted return using average volatility from the test set (if available)
    if 'Volatility' in X_test.columns:
        avg_volatility = X_test['Volatility'].mean()
    else:
        avg_volatility = 1.0  # Fallback if volatility is not present
    risk_adjusted_return = predictions[-1] / avg_volatility if avg_volatility != 0 else 0
    logger.info(f"Risk-Adjusted Return (Last Predicted Return / Avg Volatility): {risk_adjusted_return}")

    # Plot predicted vs. actual prices
    plot_predictions(test_dates, test_df['Actual_Price'], test_df['Predicted_Price'],
                     title=f"{ticker} Stock Price Prediction")

if __name__ == "__main__":
    main()
