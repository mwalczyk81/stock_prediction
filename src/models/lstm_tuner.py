import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from rich.progress import Progress
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from src.logger import logger

# Disable default Optuna logging to prevent excessive output
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()

class LSTMPredictor(nn.Module):
    """LSTM model for time series forecasting.

    This model consists of an LSTM layer followed by a fully connected layer.
    It predicts the next step in a univariate time series.

    Args:
        input_size (int, optional): The number of input features. Defaults to 1.
        hidden_size (int, optional): The number of units in the LSTM hidden layer. Defaults to 50.
        num_layers (int, optional): The number of LSTM layers. Defaults to 2.
        dropout (float, optional): The dropout rate applied to the LSTM layers. Defaults to 0.2.
    """

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, 
            dropout=dropout if num_layers > 1 else 0  # Avoid dropout for a single-layer LSTM
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The model output, representing the predicted value for the next time step.
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output

def objective(trial: optuna.Trial, X_train: torch.Tensor, y_train: torch.Tensor, data: np.ndarray, device: str) -> float:
    """Defines the hyperparameter search objective for Optuna.

    This function defines the range of hyperparameters to search, trains the LSTM model for a
    given set of hyperparameters, and evaluates the performance using Mean Squared Error (MSE).

    Args:
        trial (optuna.Trial): The trial instance provided by Optuna.
        X_train (torch.Tensor): The training dataset features.
        y_train (torch.Tensor): The training dataset target values.
        data (np.ndarray): The original time series data (used for reference, if needed).
        device (str): The device to run the model on ('cuda' or 'cpu').

    Raises:
        optuna.exceptions.TrialPruned: If the trial should be pruned early based on performance.

    Returns:
        float: The average validation loss (MSE) for the given hyperparameters.
    """

    latest_price = data['Close'].iloc[-1]
    logger.debug(f"Using latest price from data: {latest_price}")

    # Define hyperparameter search space
    hidden_size = trial.suggest_int("hidden_size", 50, 200, step=50)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Prepare dataset
    train_dataset = TensorDataset(X_train.clone().detach().float(), 
                                  y_train.clone().detach().float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMPredictor(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 50
    for epoch in range(num_epochs):
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
        if epoch % 10 == 0:
            logger.debug(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Report and check for pruning
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss

def tune_lstm_hyperparameters(X_train: torch.Tensor, y_train: torch.Tensor, data: np.ndarray, n_trials: int = 10, 
                              progress: Progress = None) -> Tuple[nn.Module, dict]:
    """Tunes LSTM hyperparameters using Optuna with a shared progress bar if provided.

    This function runs an Optuna study to find the best LSTM hyperparameters for time series forecasting.

    Args:
        X_train (torch.Tensor): The training dataset features.
        y_train (torch.Tensor): The training dataset target values.
        data (np.ndarray): The original time series data.
        n_trials (int, optional): The number of hyperparameter tuning trials. Defaults to 10.
        progress (Progress, optional): A Rich progress bar instance for tracking tuning progress. Defaults to None.

    Returns:
        Tuple[nn.Module, dict]: A tuple containing the best-trained LSTM model and its hyperparameters.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = optuna.create_study(direction="minimize")

    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")

    # Handle progress bar
    own_progress = False
    if progress is None:
        progress = Progress(transient=True)
        progress.start()
        own_progress = True
    task = progress.add_task("[cyan]Running Hyperparameter Tuning...", total=n_trials)

    def wrapped_objective(trial):
        """Wrapper function for Optuna's objective function that updates progress."""
        loss = objective(trial, X_train, y_train, data, device)
        progress.update(task, advance=1)
        return loss

    # Run hyperparameter optimization
    study.optimize(wrapped_objective, n_trials=n_trials)

    # Ensure the progress task shows 100% completion.
    progress.update(task, completed=n_trials)

    if own_progress:
        progress.stop()

    # Retrieve the best hyperparameters
    best_hps = study.best_params
    logger.info(f"Best Hyperparameters: {best_hps}")

    # Create the best model using optimal hyperparameters
    best_model = LSTMPredictor(input_size=1,
                               hidden_size=best_hps["hidden_size"],
                               num_layers=best_hps["num_layers"],
                               dropout=best_hps["dropout"]).to(device)

    return best_model, best_hps
