import logging
import pandas as pd
import concurrent.futures
from rich.progress import Progress
from datetime import datetime, timedelta
import colorlog

from src.data.data_fetcher import fetch_stock_data
from src.data.preprocessing import create_features_targets
from src.models.random_forest import train_random_forest
from src.models.xgboost_model import train_xgboost

# Configure logging with colorlog
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

def analyze_stock(ticker, start_date, end_date, model_type="rf", horizon=5):
    """
    Processes one ticker:
    - Fetches and preprocesses the stock data.
    - Trains a model (RandomForest by default, or XGBoost if model_type="xgb").
    - Computes predicted returns and a risk-adjusted return metric.
    - Returns a dictionary with key metrics.
    """
    try:
        logger.info(f"Processing {ticker}... (start)")
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is None or data.empty:
            logger.error(f"No data for {ticker}. Skipping...")
            return None
        logger.info(f"{ticker} - Data fetched, {len(data)} rows.")

        # Create features and target using the specified horizon
        features, target = create_features_targets(data, horizon=horizon)
        logger.info(f"{ticker} - Features and target created.")

        # Select the model type
        if model_type == "rf":
            model, X_test, y_test, predictions, mse = train_random_forest(features, target)
        elif model_type == "xgb":
            model, X_test, y_test, predictions, mse = train_xgboost(features, target)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        logger.info(f"{ticker} - Model MSE: {mse}")

        # Reconstruct predicted prices using today's close and predicted return
        test_df = X_test.copy()
        test_df['Predicted_Price'] = test_df['Close'] * (1 + predictions)
        test_df['Actual_Price'] = test_df['Close'] * (1 + y_test)
        test_dates = data.iloc[-len(X_test):].index

        avg_predicted_return = predictions.mean()
        last_predicted_return = predictions[-1]
        avg_volatility = X_test['Volatility'].mean() if 'Volatility' in X_test.columns else 1.0
        risk_adjusted_return = last_predicted_return / avg_volatility if avg_volatility != 0 else 0

        logger.info(f"{ticker} - Analysis complete.")
        return {
            "ticker": ticker,
            "mse": mse,
            "avg_predicted_return": avg_predicted_return,
            "last_predicted_return": last_predicted_return,
            "risk_adjusted_return": risk_adjusted_return,
            "dates": test_dates,
            "predicted_prices": test_df['Predicted_Price'],
            "actual_prices": test_df['Actual_Price']
        }
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None

def load_ticker_list(filepath="data/raw/market_tickers.csv"):
    """
    Loads a list of tickers from a CSV file with a column named 'Ticker'.
    """
    df = pd.read_csv(filepath)
    return df['Ticker'].tolist()

def main():
    # Define dynamic date range: last 10 years until today
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    
    # Load tickers from a CSV file
    tickers = load_ticker_list("data/raw/market_tickers.csv")
    results = []
    
    # Use ThreadPoolExecutor for IO-bound processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_stock, ticker, start_date, end_date, model_type="xgb", horizon=5): ticker 
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

    # Build summary DataFrame with key metrics
    summary = pd.DataFrame({
        "Ticker": [res["ticker"] for res in results],
        "MSE": [res["mse"] for res in results],
        "Avg_Pred_Return": [res["avg_predicted_return"] for res in results],
        "Last_Pred_Return": [res["last_predicted_return"] for res in results],
        "Risk_Adjusted_Return": [res["risk_adjusted_return"] for res in results]
    })

    # Filter and rank stocks (e.g., only stocks with risk-adjusted return above a threshold)
    threshold = 0.01
    filtered = summary[summary["Risk_Adjusted_Return"] > threshold]
    filtered.sort_values(by="Risk_Adjusted_Return", ascending=False, inplace=True)

    print("Summary of model performance and predicted returns:")
    print(summary)
    print("\nFiltered stocks with risk-adjusted predicted return > 1%:")
    print(filtered)
    
    # Generate advice: suggest the top 5 stocks
    top_stocks = filtered.head(5)
    print("\nSuggested stocks to buy (top 5):")
    print(top_stocks)

if __name__ == "__main__":
    main()
