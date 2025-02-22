# src/models/lstm.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm  # progress bar

def prepare_lstm_data(series, sequence_length=60):
    """
    Prepares data for the LSTM model from a univariate time series.
    Returns input sequences (X), corresponding targets (y), and the scaler.
    A progress bar is displayed during sequence creation.
    """
    series = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)
    
    X, y = [], []
    # Using tqdm to show progress over the number of sequences created
    for i in tqdm(range(sequence_length, len(scaled_series)), desc="Preparing LSTM data"):
        X.append(scaled_series[i-sequence_length:i, 0])
        y.append(scaled_series[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    # Reshape for LSTM: (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model using TensorFlow Keras.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Trains the LSTM model using early stopping.
    Returns the training history.
    """
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        verbose=1, callbacks=callbacks)
    return history
