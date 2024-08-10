import plotly.express as px
import numpy as np
import plotly.graph_objects as go


def plot_results(inventory, forecast, orders_placed, orders_received, stock_min, capacity, type="lines"):
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
    if type == "lines":
        space = np.arange(1, len(inventory) + 1, 1)
        # Crear la figura
        fig = go.Figure()

        # Añadir trazas con colores y formas de marcadores definidos
        fig.add_trace(go.Scatter(x=space, y=inventory,
                                mode='lines',
                                name='Inventory variation',
                                line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=space, y=forecast,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=space, y=orders_placed,
                                mode='markers',
                                name='Orders placed',
                                marker=dict(color='cyan', symbol='x', size=12)))
        fig.add_trace(go.Scatter(x=space, y=orders_received,
                                mode='markers',
                                name='Orders received',
                                marker=dict(color='green', symbol='square', size=12)))
        fig.add_trace(go.Scatter(x=space, y=[stock_min] * len(inventory),
                                mode='lines',
                                name='Minimum stock level',
                                line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=space, y=[capacity] * len(inventory),
                                mode='lines',
                                name='Maximum stock level',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(
                            title='Inventory Management',
                            width=800,
                            height=600
                        )

        # Mostrar la figura
        fig.show()