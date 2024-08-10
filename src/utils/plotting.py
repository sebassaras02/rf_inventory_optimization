import plotly.express as px
import numpy as np
import plotly.graph_objects as go


def create_traces(space, inventory, forecast, orders_placed, orders_received, stock_min, capacity, trace_type):
    traces = []
    
    if trace_type == "line":
        traces.append(go.Scatter(x=space, y=inventory,
                                 mode='lines',
                                 name='Inventory variation',
                                 line=dict(color='orange')))
        traces.append(go.Scatter(x=space, y=forecast,
                                 mode='lines',
                                 name='Forecast',
                                 line=dict(color='royalblue')))
    elif trace_type == "bar":
        traces.append(go.Bar(x=space, y=inventory,
                             name='Inventory variation',
                             marker=dict(color='orange')))
        traces.append(go.Bar(x=space, y=forecast,
                             name='Forecast',
                             marker=dict(color='royalblue')))

    traces.append(go.Scatter(x=space, y=orders_placed,
                             mode='markers',
                             name='Orders placed',
                             marker=dict(color='cyan', symbol='x', size=12)))
    traces.append(go.Scatter(x=space, y=orders_received,
                             mode='markers',
                             name='Orders received',
                             marker=dict(color='green', symbol='square', size=12)))
    traces.append(go.Scatter(x=space, y=[stock_min] * len(inventory),
                             mode='lines',
                             name='Minimum stock level',
                             line=dict(color='red', dash='dash')))
    traces.append(go.Scatter(x=space, y=[capacity] * len(inventory),
                             mode='lines',
                             name='Maximum stock level',
                             line=dict(color='red', dash='dash')))
    
    return traces

def plot_results(inventory, forecast, orders_placed, orders_received, stock_min, capacity, type="lines"):
    """
    Plot the results of the inventory optimization using Q-Learning.

    Args:
        inventory (list): List of inventory levels for each month.
        forecast (np.array): Numpy array of forecasted demand.
        orders_placed (list): List of ordered amount for each month.
        orders_received (list): List of received orders for each month.
        stock_min (int): Minimum stock level.
        capacity (int): Maximum stock capacity.
        type (str): Type of plot ('lines' or 'bar').

    Returns:
        fig (go.Figure): The plotly figure object.
    """
    space = np.arange(1, len(inventory) + 1, 1)
    
    fig = go.Figure()
    
    if type == "lines":
        traces = create_traces(space, inventory, forecast, orders_placed, orders_received, stock_min, capacity, trace_type="line")
    elif type == "bar":
        traces = create_traces(space, inventory, forecast, orders_placed, orders_received, stock_min, capacity, trace_type="bar")
    
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title='Inventory Management',
        xaxis_title='Month',
        yaxis_title='Stock Level',
        width=800,
        height=600,
        showlegend=True
    )

    # Mostrar la figura
    fig.show()
    
    return fig
