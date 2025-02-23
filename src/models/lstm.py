import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rich.progress import Progress
from src.logger import logger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

# **1️⃣ Data Preparation Function**
def prepare_lstm_data(series: np.ndarray, sequence_length: int = 120) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Prepares sequences for LSTM training from a univariate time series."""
    series = np.asarray(series).reshape(-1, 1)
    if len(series) <= sequence_length:
        raise ValueError("The length of the series must be greater than the sequence_length.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)

    X = np.array([scaled_series[i-sequence_length:i, 0] for i in range(sequence_length, len(scaled_series))])
    y = np.array([(series[i, 0] - series[i-1, 0]) / series[i-1, 0] for i in range(sequence_length, len(series))])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    logger.debug(f"Scaled y range: min={y.min().item()}, max={y.max().item()}")
    return X, y, scaler


# **2️⃣ Define LSTM Model in PyTorch**
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, dropout=0.2):
        """Initializes the LSTM model."""
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output


# **3️⃣ Training Function with Progress Bar**
def train_lstm_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, 
                     epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001, 
                     progress: Progress = None, ticker: str = None) -> nn.Module:
    """Trains the LSTM model using a shared progress bar if provided; otherwise, creates its own.
    Implements early stopping if no improvement is seen over several epochs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping_patience = 5
    best_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting LSTM training for {epochs} epochs...")

    own_progress = False
    if progress is None:
        progress = Progress()
        progress.start()
        own_progress = True
    task = progress.add_task(f"[cyan]Training LSTM for {ticker}...", total=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 1), y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        progress.update(task, advance=1)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break

    # Ensure the task is marked as complete even if early stopping was triggered.
    progress.update(task, completed=epochs)

    if own_progress:
        progress.stop()

    logger.info("LSTM training complete.")
    return model
