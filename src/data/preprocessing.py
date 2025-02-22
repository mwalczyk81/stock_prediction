# src/data/preprocessing.py
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame:
    - 20-day Simple Moving Average (SMA)
    - 20-day Exponential Moving Average (EMA)
    - Relative Strength Index (RSI)
    """
    # Ensure 'Close' is a Series
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]
        
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_lag_features(df, lags=[1, 2, 3]):
    """
    Adds lagged features:
    - Previous day's close and corresponding percentage return.
    """
    for lag in lags:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag{lag}'] = df['Close'].pct_change(lag)
    return df

def add_volatility(df, window=20):
    """
    Adds historical volatility as the rolling standard deviation of daily returns.
    """
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=window).std()
    df.drop(columns=['Daily_Return'], inplace=True)
    return df

def add_momentum(df, window=10):
    """
    Adds a momentum feature as the percentage change from 'window' days ago.
    """
    df['Momentum'] = df['Close'].pct_change(periods=window)
    return df

def add_extra_features(df):
    """
    Combines additional features (volatility and momentum).
    """
    df = add_volatility(df, window=20)
    df = add_momentum(df, window=10)
    return df

def create_features_targets(df, horizon=5):
    """
    Create features and target:
    - Features: Open, High, Low, Close, Volume, technical indicators, lag features, volatility, and momentum.
    - Target: The percentage return over a given horizon (default is 5 days).
    """
    df = add_technical_indicators(df)
    df = add_lag_features(df, lags=[1, 2, 3])
    df = add_extra_features(df)
    # Calculate horizon return: percentage change from today to 'horizon' days in the future
    df['Target'] = (df['Close'].shift(-horizon) / df['Close']) - 1
    df = df.dropna()
    
    features = df[['Open', 'High', 'Low', 'Close', 'Volume',
                   'SMA_20', 'EMA_20', 'RSI',
                   'Close_lag1', 'Return_lag1',
                   'Close_lag2', 'Return_lag2',
                   'Close_lag3', 'Return_lag3',
                   'Volatility', 'Momentum']]
    target = df['Target']
    return features, target

