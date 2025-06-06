# Stock Prediction Project

This project uses AI and machine learning to predict stock prices and provide actionable buy/sell recommendations. It leverages historical data from Yahoo Finance, technical indicators, and machine learning models—including RandomForest and XGBoost—to analyze stocks over a multi-day horizon. Additional features such as volatility and momentum are incorporated to refine predictions and compute risk-adjusted returns.

This is just for fun and my own learning.....it is probably wrong so do not take advice from it!

## Directory Structure

- **data/**: Contains raw and processed stock data.
  - **market_tickers.csv**: A CSV file with a list of stock tickers (with a column named "Ticker").
- **notebooks/**: Jupyter notebooks for data exploration and prototyping.
- **scripts/**: Main scripts to run the project
    - `main.py`: Main script for running an end-to-end pipeline for a single stock.
    - `market_scanner_live.py`: Scans the market, ranks stocks by predicted and risk-adjusted returns, and provides recommendations.
- **src/**: Source code organized into:
  - **data/**: Data fetching and preprocessing modules.
    - `data_fetcher.py`: Downloads stock data using yfinance.
    - `preprocessing.py`: Cleans data and generates features (technical indicators, lag features, volatility, and momentum).
  - **forecasting/**: Forecasting scripts
    - `forecasting.py` Forecasts one or more steps ahead 
  - **models/**: Model training modules.
    - `random_forest.py`: Trains a RandomForest model.
    - `xgboost_model.py`: Trains an XGBoost model.
    - `lstm.py`: Builds and trains an LSTM network (optional).
  - **utils/**: Utility functions.
    - `visualization.py`: Contains functions for plotting predictions.
    - `logger.py`: Logger setup
- **tests/**: Unit and integration tests.
- **requirements.txt**: Project dependencies.
- **README.md**: Project documentation.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd stock_prediction_project
   ```

2. **Create and Activate a Virtual Environment**

    On Windows:
    ```bash
    py -3.11 -m venv env
    .\env\Scripts\activate
    ```

    On macOS/Linux:
    ```bash
    python3.11 -m venv env
    source env/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare Data**

    Place your market_tickers.csv file (which should include a column named "Ticker") in the project root (or in a designated data folder).

5. **Run the Project**
    **Single Stock Prediction:**  
    Run the main script for a single stock:

    ```bash
    python -m src.main
    ```

    **Market Scanning and Buy Advice:**  
    Run the market scanner to analyze and rank multiple stocks:

    ```bash
    python -m src.market_scanner_live
    ```

## Usage and Customization

**Model Selection:**  
Switch between RandomForest and XGBoost models by setting the `model_type` parameter in `market_scanner_live.py`.

**Target Horizon:**  
The project currently predicts a multi-day return (default is 5 days). You can adjust the horizon by modifying the `horizon` parameter in the `create_features_targets` function in `src/data/preprocessing.py`.

**Feature Engineering:**  
The preprocessing pipeline includes:
- Technical indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence): Includes the MACD line, its signal line, and the MACD histogram (difference). These are useful for identifying trend direction and momentum.
- Lag features (e.g., previous day's close price, previous returns)
- Volatility (rolling standard deviation of returns)
- Momentum (percentage change over a set period)

Feel free to add or modify features to better suit your strategy.

**Risk-Adjusted Returns:**  
The market scanner computes a risk-adjusted return (predicted return divided by historical volatility) to help rank stocks on a risk-adjusted basis.

**Visualization:**  
Plotting functions are available in `src/utils/visualization.py`. You can customize how results are displayed or save plots to files if preferred.

## Evaluation Metrics

To better assess model performance beyond standard metrics like Mean Squared Error (MSE), the following has been incorporated:

- **Sharpe Ratio**:
    - Calculated for the Random Forest and XGBoost models based on their predictions on the test set.
    - The Sharpe Ratio measures the risk-adjusted return of the model's predictions. It is calculated by taking the average predicted (excess) return and dividing it by the standard deviation of those predicted returns, then annualizing it.
    - A higher Sharpe Ratio generally indicates a better risk-adjusted performance. This helps in understanding if the model's predicted returns are a result of excessive risk or sound predictions.
    - This metric is logged during model training and can be used to compare different models or hyperparameter settings from a risk-adjusted perspective.

## Testing

To run unit tests:

```bash
python -m unittest discover -s tests
```
## Disclaimer

This project is intended for educational and research purposes only. The stock predictions and recommendations generated by this project are not financial advice. Trading in the stock market involves risk, and you should consult with a financial advisor before making any investment decisions.
