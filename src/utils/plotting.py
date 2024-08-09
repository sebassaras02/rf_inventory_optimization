import matplotlib.pyplot as plt

def plot_results(inventory, forecast, ordered, stock_min, capacity):
    """
    Plot the results of the inventory optimization using Q-Learning

    Args:
        inventory (list): List of inventory levels for each month
        forecast (np.array): Numpy array of forecasted demand
        ordered (list): List of ordered amount for each month
        stock_min (int): Minimum stock level

    Returns:
        None
    """
    plt.plot(inventory, "g" , label="Inventory variation")
    plt.plot(forecast, "b", label="Forecasted consumption")
    plt.plot(ordered, "p", label="Quantity Ordered")
    plt.axhline(stock_min, color="red", label="Security Stock")
    plt.axhline(capacity, color="orange", label="Capacity")
    plt.legend()