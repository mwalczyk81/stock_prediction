from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import logger


def prepare_lstm_data(
    series: np.ndarray, sequence_length: int = 120
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Prepares sequences for LSTM training from a univariate time series.

    This function scales the input series using MinMaxScaler, creates rolling
    sequences of a fixed length, and returns them in PyTorch tensor format.

    Args:
        series (np.ndarray): A 1D NumPy array representing the time series data.
        sequence_length (int, optional): The length of the input sequences. Defaults to 120.

    Raises:
        ValueError: If the length of the series is less than or equal to `sequence_length`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
            - X (torch.Tensor): The input features for the LSTM model, shaped as (samples, sequence_length, 1).
            - y (torch.Tensor): The target values, representing relative price changes.
            - scaler (MinMaxScaler): The scaler used for normalization.
    """

    series = np.asarray(series).reshape(-1, 1)
    if len(series) <= sequence_length:
        raise ValueError(
            "The length of the series must be greater than the sequence_length."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)

    X = np.array(
        [
            scaled_series[i - sequence_length : i, 0]
            for i in range(sequence_length, len(scaled_series))
        ]
    )
    y = np.array(
        [
            (series[i, 0] - series[i - 1, 0]) / series[i - 1, 0]
            for i in range(sequence_length, len(series))
        ]
    )

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    logger.debug(f"Scaled y range: min={y.min().item()}, max={y.max().item()}")
    return X, y, scaler


class LSTMPredictor(nn.Module):
    """LSTM model for time series forecasting.

    This model consists of an LSTM layer followed by a fully connected layer.
    It predicts the next step in a univariate time series.

    Args:
        input_size (int, optional): The number of input features. Defaults to 1.
        hidden_size (int, optional): The number of units in the LSTM hidden layer. Defaults to 100.
        num_layers (int, optional): The number of LSTM layers. Defaults to 2.
        dropout (float, optional): The dropout rate applied to the LSTM layers. Defaults to 0.2.
    """

    def __init__(self, input_size=1, hidden_size=100, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The model output, representing the predicted value for the next time step.
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output


def train_lstm_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    progress: Progress = None,
    ticker: str = None,
) -> nn.Module:
    """Trains the LSTM model using a shared progress bar if provided; otherwise, creates its own.

    This function trains the LSTM model on the provided dataset and implements early stopping if no
    improvement is observed over a few consecutive epochs. It uses Mean Squared Error (MSE) as the loss function.

    Args:
        model (nn.Module): The LSTM model to train.
        X_train (torch.Tensor): The input features for training, shaped as (samples, sequence_length, 1).
        y_train (torch.Tensor): The target values for training.
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        progress (Progress, optional): A Rich progress bar instance for tracking training progress. Defaults to None.
        ticker (str, optional): The stock ticker symbol being trained on. Defaults to None.

    Returns:
        nn.Module: The trained LSTM model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize AMP's GradScaler for mixed precision
    scaler = torch.GradScaler()

    best_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if device == "cuda":
                with torch.autocast(device_type="cuda"):
                    outputs = model(X_batch)
                    loss = criterion(outputs.view(-1, 1), y_batch.view(-1, 1))
            else:
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1, 1), y_batch.view(-1, 1))

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if progress is not None:
            progress.update(0, advance=1)  # Update your progress bar as needed

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if progress is not None:
                    progress.console.log("Early stopping triggered!")
                break

        if progress is not None and epoch % 10 == 0:
            progress.console.log(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    if progress is not None:
        progress.console.log("LSTM training complete.")
    return model
