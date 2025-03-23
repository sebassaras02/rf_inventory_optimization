import numpy as np


def inference_values(q_table, choices, forecast, initial_state, min_order, lead_time):
    """
    Inference function to get the inventory levels, optimal actions, forecast and ordered amount based on the results of the Q-Learning algorithm.

    Args:
        q_table (np.array): Q-Table with the Q-values of each state-action pair
        choices (list): List of possible actions
        forecast (np.array): Numpy array of forecasted demand
        initial_state (int): Initial stock value of the inventory
        min_order (int): Minimum order value
        lead_time (int): Lead time (in time units) for the orders

    Returns:
        inventory_levels (list): List of inventory levels for each period
        optimal_actions (list): List of optimal actions for each period
        forecast (np.array): Numpy array of forecasted demand
        orders_placed (list): List of ordered amounts for each period
        orders_received (list): List of orders received for each period
    """
    inventory_levels = []
    optimal_actions = []
    ordered_amount = []
    orders_placed = [0] * len(forecast)
    orders_received = [0] * len(forecast)

    current_state = initial_state
    orders = []

    for iteration in range(len(forecast)):
        if iteration == 0:
            current_state = initial_state

        # Obtain the best action based on the Q-Table
        best_action_index = np.argmax(q_table[iteration])
        best_action = choices[best_action_index]
        optimal_actions.append(best_action)

        # Determine the amount to order based on the best action
        if best_action == "no":
            order_value = 0
        else:
            try:
                multiplier = int(best_action.rstrip("m"))
            except ValueError:
                multiplier = 0  # En caso de error, se asigna 0
            order_value = multiplier * min_order

        # Record the order placed in the current period
        orders_placed[iteration] = order_value

        if order_value > 0:
            orders.append(
                {
                    "quantity": order_value,
                    "unit_ordered": iteration,
                    "unit_arrival": iteration + lead_time,
                }
            )

        # Apply pending orders if their arrival time has come
        for order in orders[:]:
            if iteration == order["unit_arrival"]:
                current_state += order["quantity"]
                orders_received[iteration] = order["quantity"]
                orders.remove(order)

        # Update the inventory level based on the forecast and the ordered amount
        current_state -= forecast[iteration]
        inventory_levels.append(current_state)
        ordered_amount.append(order_value)

    return inventory_levels, optimal_actions, forecast, orders_placed, orders_received