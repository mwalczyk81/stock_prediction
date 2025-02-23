import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Any

def prepare_lstm_data(series: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepares sequences for LSTM training from a univariate time series.
    
    Args:
        series (np.ndarray or pandas.Series): Time series data.
        sequence_length (int): The number of time steps in each sequence.
        
    Returns:
        X (np.ndarray): 3D array of shape (samples, sequence_length, 1) for LSTM input.
        y (np.ndarray): Target array.
        scaler (MinMaxScaler): Fitted scaler to transform predictions back.
        
    Raises:
        ValueError: If the series length is less than or equal to sequence_length.
    """
    # Convert series to a numpy array if it's not already
    series = np.asarray(series).reshape(-1, 1)
    if len(series) <= sequence_length:
        raise ValueError("The length of the series must be greater than the sequence_length.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)
    
    # Build sequences and targets using list comprehensions
    X = np.array([scaled_series[i-sequence_length:i, 0] for i in range(sequence_length, len(scaled_series))])
    y = np.array([scaled_series[i, 0] for i in range(sequence_length, len(scaled_series))])
    
    # Reshape X to be [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Builds and compiles an LSTM model.
    
    Args:
        input_shape (Tuple[int, int]): The shape of each input sample (timesteps, features).
    
    Returns:
        model (Sequential): A compiled Keras Sequential model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Any:
    """
    Trains the LSTM model using EarlyStopping to avoid overfitting.
    
    Args:
        model (Sequential): The compiled LSTM model.
        X_train (np.ndarray): Training data input sequences.
        y_train (np.ndarray): Training targets.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.
    
    Returns:
        history: The training history object.
    """
    callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks
    )
    return history
