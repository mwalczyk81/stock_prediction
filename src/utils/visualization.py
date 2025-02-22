import matplotlib.pyplot as plt

def plot_predictions(dates, actual, predicted, title="Stock Price Prediction"):
    """
    Plots actual vs. predicted stock prices over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual Price", color='blue')
    plt.plot(dates, predicted, label="Predicted Price", color='red')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.show()
