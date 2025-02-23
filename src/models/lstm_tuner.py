import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rich.progress import Progress
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from src.logger import logger

optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()

# **1️⃣ Define LSTM Model**
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        """Initializes the LSTM model."""
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, 
            dropout=dropout if num_layers > 1 else 0  # Avoid dropout for a single-layer LSTM
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last LSTM output

# **2️⃣ Objective Function for Optuna Tuning**
def objective(trial, X_train, y_train, data, device):
    """Defines the hyperparameter search objective for Optuna."""
    latest_price = data['Close'].iloc[-1]
    logger.debug(f"Using latest price from data: {latest_price}")

    hidden_size = trial.suggest_int("hidden_size", 50, 200, step=50)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_dataset = TensorDataset(X_train.clone().detach().float(), 
                                  y_train.clone().detach().float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss

# **3️⃣ Function to Tune LSTM Hyperparameters**
def tune_lstm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, data, n_trials=10, 
                              progress: Progress = None) -> Tuple[nn.Module, dict]:
    """Tunes LSTM hyperparameters using Optuna with a shared progress bar if provided."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = optuna.create_study(direction="minimize")

    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")

    own_progress = False
    if progress is None:
        progress = Progress(transient=True)
        progress.start()
        own_progress = True
    task = progress.add_task("[cyan]Running Hyperparameter Tuning...", total=n_trials)

    def wrapped_objective(trial):
        loss = objective(trial, X_train, y_train, data, device)
        progress.update(task, advance=1)
        return loss

    study.optimize(wrapped_objective, n_trials=n_trials)

    # Ensure the progress task shows 100% completion.
    progress.update(task, completed=n_trials)

    if own_progress:
        progress.stop()

    best_hps = study.best_params
    logger.info(f"Best Hyperparameters: {best_hps}")

    best_model = LSTMPredictor(input_size=1,
                               hidden_size=best_hps["hidden_size"],
                               num_layers=best_hps["num_layers"],
                               dropout=best_hps["dropout"]).to(device)

    return best_model, best_hps
