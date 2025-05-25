import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_xgboost(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.Series, pd.Series, float, float]:
    """Trains an XGBoost model for stock return prediction.

    This function:
    - Splits the input dataset into training and testing sets.
    - Trains an XGBoost regression model to predict future stock returns.
    - Evaluates model performance using Mean Squared Error (MSE).

    Args:
        X (pd.DataFrame): The feature dataset containing historical stock indicators.
        y (pd.Series): The target variable representing stock returns.
        test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): The seed for randomization to ensure reproducibility. Defaults to 42.

    Returns:
        Tuple[xgb.XGBRegressor, pd.DataFrame, pd.Series, pd.Series, float, float]:
            - The trained XGBoost model.
            - The test feature set (X_test).
            - The test target values (y_test).
            - The model's predictions.
            - The mean squared error (MSE) of the predictions.
            - sharpe_ratio: Calculated Sharpe Ratio from predictions.
    """
    # Split dataset into training and testing sets (no shuffle to preserve time series order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    # Ensure predictions is a pd.Series for Sharpe Ratio calculation
    if isinstance(predictions, np.ndarray):
        predictions_series = pd.Series(predictions, index=y_test.index)
    else:
        predictions_series = predictions # Assuming it's already a Series if not ndarray

    # Calculate Sharpe Ratio using the utility function
    from src.utils.metrics import calculate_sharpe_ratio # Local import
    sharpe_ratio = calculate_sharpe_ratio(predictions_series)

    return model, X_test, y_test, predictions_series, mse, sharpe_ratio
