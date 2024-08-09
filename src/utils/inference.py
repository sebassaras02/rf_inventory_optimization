import numpy as np

def inference_values(q_table, choices, forecast, initial_state):
    """
    Inference function to get the inventory levels, optimal actions, forecast and ordered amount based on the results of the Q-Learning algorithm.

    Args:
        q_table (np.array): Q-Table with the Q-values of each state-action pair
        choices (list): List of possible actions
        forecast (np.array): Numpy array of forecasted demand
        initial_state (dict): Dictionary with the initial state of the inventory
    
    Returns:
        inventory_levels (list): List of inventory levels for each month
        optimal_actions (list): List of optimal actions for each month
        forecast (np.array): Numpy array of forecasted demand
        ordered_amount (list): List of ordered amount for each month
    """
    inventory_levels = [initial_state["stock"]]
    optimal_actions = []
    ordered_amount = []

    current_state = initial_state
    order_value = 0

    for iteration in range(len(forecast)):

        if iteration == 0:
            current_state = initial_state["stock"]
        
        # Obtener la mejor acción utilizando la Q-Tabla
        best_action_index = np.argmax(q_table[iteration])
        best_action = choices[best_action_index]
        optimal_actions.append(best_action)

        # Determinar el valor del pedido basado en la acción
        if best_action == "nopedir":
            order_value = 0
        elif best_action == "minimo":
            order_value = initial_state["minimoPedido"]
        elif best_action == "2minimo":
            order_value = 2 * initial_state["minimoPedido"]
        elif best_action == "3minimo":
            order_value = 3 * initial_state["minimoPedido"]
        elif best_action == "4minimo":
            order_value = 4 * initial_state["minimoPedido"]

        ordered_amount.append(order_value)

        # Actualizar el inventario para el mes actual
        current_state += order_value - forecast[iteration]
        inventory_levels.append(current_state)
    
    return inventory_levels, optimal_actions, forecast, ordered_amount