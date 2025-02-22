# src/models/random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import time
import logging

def train_random_forest(X, y, test_size=0.2, random_state=42):
    """
    Trains a RandomForestRegressor on the given features X and target y.
    Uses TimeSeriesSplit for hyperparameter tuning.
    Returns the trained model, test set, predictions, and the Mean Squared Error.
    """
    # Split data without shuffling to preserve time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    logging.info(f"Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds.")
    
    best_model = grid_search.best_estimator_
    
    # Print feature importances for diagnostics
    importances = best_model.feature_importances_
    for feat, imp in zip(X.columns, importances):
        logging.info(f"Feature: {feat}, Importance: {imp:.4f}")
    
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return best_model, X_test, y_test, predictions, mse
