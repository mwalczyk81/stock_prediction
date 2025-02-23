import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators: 20-day SMA, 20-day EMA, and RSI.
    Ensures 'Close' is a Series.
    """
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_lag_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    """
    Adds lagged features: previous close and percentage return for specified lags.
    """
    for lag in lags:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
        df[f'Return_lag{lag}'] = df['Close'].pct_change(lag)
    return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds historical volatility as the rolling standard deviation of daily returns.
    """
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=window).std()
    df.drop(columns=['Daily_Return'], inplace=True)
    return df

def add_momentum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds a momentum feature as the percentage change from 'window' days ago.
    """
    df['Momentum'] = df['Close'].pct_change(periods=window)
    return df

def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies additional feature functions (volatility and momentum).
    """
    df = add_volatility(df, window=20)
    df = add_momentum(df, window=10)
    return df

def create_features_targets(df: pd.DataFrame, horizon: int = 5) -> (pd.DataFrame, pd.Series):
    """
    Creates a feature matrix and target vector.
    Features include raw prices, technical indicators, lag features, volatility, and momentum.
    The target is the percentage return over a given horizon.
    """
    df = add_technical_indicators(df)
    df = add_macd(df)
    df = add_lag_features(df, lags=[1, 2, 3])
    df = add_extra_features(df)
    
    # Calculate horizon return: percentage change from current close to close after 'horizon' days
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

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD indicator and add to DataFrame."""
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df.drop(columns=['EMA12', 'EMA26'], inplace=True)
    return df
