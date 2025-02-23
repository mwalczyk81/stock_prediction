from sklearn.discriminant_analysis import StandardScaler
from src.data.preprocessing import create_features_targets
import pandas as pd
import numpy as np
import torch

def forecast_one_step_lstm(model, data: pd.DataFrame, scaler, sequence_length: int = 120, horizon: int = 5) -> (float, float):
    """
    Forecasts one step ahead using an LSTM model by taking the last `sequence_length`
    closing prices, reshaping them appropriately, and predicting the future return.
    
    Args:
      model: The trained LSTM model.
      data: Historical price data as a DataFrame.
      scaler: The MinMaxScaler used during training (for inverse transformation).
      sequence_length: Number of time steps in the sequence.
      horizon: The prediction horizon (days).
               
    Returns:
      predicted_return: The predicted percentage return.
      predicted_price: The forecasted price computed from the last 'Close' value.
    """
    # Extract the most recent sequence_length closing prices
    recent_sequence = data['Close'].tail(sequence_length).values.reshape(1, sequence_length, 1)
    
    # Predict the return (model expects shape (1, sequence_length, 1))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predicted_return = model(torch.tensor(recent_sequence, dtype=torch.float32).to(next(model.parameters()).device))
    predicted_return = predicted_return.squeeze().cpu().numpy()

  
    # Clip unrealistic returns (prevents extreme predictions)
    predicted_return = np.clip(predicted_return, -0.2, 0.2)  # Limit returns to -20% to +20%
    
    # Use the last actual close as the base price
    last_close = data['Close'].iloc[-1]
    predicted_price = last_close * (1 + predicted_return)
    
    return predicted_return, predicted_price


def forecast_one_step(model, data: pd.DataFrame, horizon: int = 5) -> (float, float):
    """
    Forecasts the next horizon return and predicted price using the latest data.
    
    Args:
        model: Trained regression model.
        data: Historical price data as a DataFrame.
        horizon: Number of days ahead to forecast.
        
    Returns:
        predicted_return: The predicted percentage return.
        predicted_price: The forecasted price, calculated as last_close * (1 + predicted_return).
    """
    # Create features using the entire data set (or a recent window)
    features, _ = create_features_targets(data, horizon=horizon)
    latest_features = features.tail(1)
    predicted_return = model.predict(latest_features)[0]
    last_close = data['Close'].iloc[-1]
    predicted_price = last_close * (1 + predicted_return)
    return predicted_return, predicted_price

def forecast_multi_step(model, data: pd.DataFrame, horizon: int = 5, steps: int = 3) -> list:
    """
    Forecasts multiple steps into the future iteratively.
    
    Args:
        model: Trained regression model.
        data: Historical price data as a DataFrame.
        horizon: Prediction horizon (e.g., 5-day return).
        steps: Number of iterative forecast steps.
        
    Returns:
        forecasts: A list of dictionaries with step number, predicted return, and predicted price.
    """
    forecasts = []
    current_data = data.copy()

    for step in range(steps):
        pred_return, pred_price = forecast_one_step(model, current_data, horizon=horizon)
        forecasts.append({
            "step": step + 1,
            "predicted_return": pred_return,
            "predicted_price": pred_price
        })
        # Append a new row with the predicted price to simulate future data.
        new_row = current_data.iloc[-1].copy()
        new_row['Close'] = pred_price
        current_data = current_data.append(new_row)
    
    return forecasts
