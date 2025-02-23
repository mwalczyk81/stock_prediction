from typing import Tuple
import time
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV

def train_random_forest(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Trains a RandomForestRegressor using TimeSeriesSplit for hyperparameter tuning.
    
    Returns:
      - best_model: The best estimator from grid search.
      - X_test: Test set features.
      - y_test: Test set target values.
      - predictions: Model predictions for the test set.
      - mse: Mean Squared Error on the test set.
    """
    # Split data without shuffling to preserve time order.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Define a hyperparameter grid.
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    # Use TimeSeriesSplit for proper time series cross-validation.
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    logging.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds.")
    
    best_model = grid_search.best_estimator_
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    
    # Log feature importances at DEBUG level for diagnostic purposes.
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        importances = best_model.feature_importances_
        for feat, imp in zip(X.columns, importances):
            logging.debug(f"Feature: {feat}, Importance: {imp:.4f}")
    
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return best_model, X_test, y_test, predictions, mse
