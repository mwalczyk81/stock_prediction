# src/models/xgboost_model.py
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_xgboost(X, y, test_size=0.2, random_state=42):
    """
    Trains an XGBoost model to predict the next-day return.
    Returns the trained model, test set, predictions, and Mean Squared Error.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, X_test, y_test, predictions, mse
