import numpy as np
from utils.inference import inference_values
from utils.plotting import plot_results

class QLearningOptimizer:
    """
    Inventory Optimizer using Q-Learning for Reinforcement Learning

    Args:
        forecast (np.array): Numpy array of forecasted demand
        initial_stock (int): Initial stock of the inventory
        security_stock (int): Security stock of the inventory
        n_actions (list): List of possible actions
        min_order (int): Minimum order quantity
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate

    Returns:
        None
    """

    def __init__(self, forecast, initial_stock, security_stock, capacity, n_actions, min_order, lead_time, alpha=0.1, gamma=0.6, epsilon=0.1):
        """
        Constructor of the Q-Learning Optimizer
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.choices = np.array(n_actions) 
        self.raw_actions = n_actions
        self.min_order = min_order
        self.forecast = forecast
        self.initial_stock = initial_stock
        self.security_stock = security_stock
        self.capacity = capacity
        self.lead_time = lead_time
        self.orders_done = []
        self.orders_arrived = []
    
    def __transition(self, state, action, consuption):
        """
        This function transitions to the new state based on the action taken and the consuption of the month.

        Args:
            state (int): Current state of the inventory
            action (str): Action taken
            consuption (int): Consuption of the month
        
        Returns:
            new_state (int): New state of the inventory
        """
        new_state = state

        # Add the order to the state
        if action == "minimo":
            orden_asked = self.min_order
            new_state += orden_asked
        elif action == "2minimo":
            orden_asked = 2 * self.min_order
            new_state += orden_asked
        elif action == "3minimo":
            orden_asked = 3 * self.min_order
            new_state += orden_asked
        elif action == "4minimo":
            orden_asked = 4 * self.min_order
            new_state += 4 * self.min_order
        elif action == "nopedir":
            orden_asked = 0
            new_state = new_state
        
        if consuption > 0: 
            # Reduce the state by the consuption
            new_state -= consuption
            return new_state, orden_asked
        else:
            return new_state
    
    def __create_q_table(self):
        """
        This function creates the Q-table based on the forecasted demand and the possible actions.

        Args:
            None
        
        Returns:
            None
        """
        # Create the Q-table
        self.q_table = np.zeros((len(self.forecast), len(self.choices)))
    
    def __get_reward(self, current_state, security_stock, maximum_stock):
        """
        This function returns the reward based on the current state of the inventory.

        Args:
            current_state (int): Current state of the inventory
            security_stock (int): Security stock of the inventory
            maximum_stock (int): Maximum stock of the inventory
            
        
        Returns:
            reward (float): Reward obtained
        """
        if current_state <= security_stock:
            return -100
        elif current_state > security_stock and current_state <= 0.25*maximum_stock:
            return 10
        elif current_state > 0.25*maximum_stock and current_state <= 0.5*maximum_stock:
            return 20
        elif current_state > 0.5*maximum_stock and current_state <= 0.75*maximum_stock:
            return 2
        elif current_state > 0.75*maximum_stock and current_state <= maximum_stock:
            return -5
        elif current_state > maximum_stock:
            return -50
    
    def __choose_action(self, state):
        """"
        This function chooses an action based on the epsilon-greedy policy.

        Args:
            state (int): Current state of the inventory
        
        Returns:
            action (str): Action to take
        """
        if state > 0.8*self.capacity:
            return "nopedir"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.choices)  # Exploration       
        else:
            max_index = np.argmax(self.q_table[state])  # Explotation
            return self.choices[max_index]
    
    
    def __update_q_table(self, action, reward, unit):
        """
        This function updates the Q-table based on the reward and the new state based on the technique used.
        The technique can be TD-Learning and SARSA for Q-learning.

        Args:
            action (str): Action taken
            reward (float): Reward obtained
            unit (int): Current unit of time
        
        Returns:
            None
        """
        index = self.raw_actions.index(action)
        if unit < len(self.forecast) - 1:
            self.q_table[unit, index] = self.q_table[unit, index] + self.alpha * (reward + self.gamma * np.max(self.q_table[unit+1]) - self.q_table[unit, index])
        else:
            # For the last state, there is no future state to consider
            self.q_table[unit, index] = self.q_table[unit, index] + self.alpha * (reward - self.q_table[unit, index])

    
    def fit(self, epochs=1000):
        """
        This function trains the Q-Learning model to optimize the inventory levels.

        Args:
            epochs (int): Number of epochs to train the model
        
        Returns:
            None
        """
        self.__create_q_table()
        for epoch in range(epochs):
            state = self.initial_stock
            # create a list to save the order done
            orders = []
            for unit in range(len(self.forecast)):
                # Choose an action
                action = self.__choose_action(state=unit)
                # Track the action to consider the lead time
                if action != "nopedir":
                    orders.append((action, unit))
                
                # Process pending orders if any
                if len(orders) > 0:
                    for pending_action, order_unit in orders[:]:
                        if unit == order_unit + self.lead_time:
                            state = self.__transition(state=state, action=pending_action, consuption=0)
                            orders.remove((pending_action, order_unit))
                
                # Apply consumption for the current unit
                new_state = self.__transition(state=state, action="nopedir", consuption=self.forecast[unit])
            
                # Get the reward
                reward = self.__get_reward(current_state=new_state, security_stock=self.security_stock, maximum_stock=self.capacity)
                # Update the Q-table
                self.__update_q_table(action=action, reward=reward, unit=unit)
                # Update the state
                state = new_state

    def predict(self):
        """
        Predict the inventory levels, optimal actions, forecast and ordered amount based on the results of the Q-Learning algorithm.

        Args:
            None
        
        Returns:
            inventory_levels (list): List of inventory levels for each month
            optimal_actions (list): List of optimal actions for each month
            forecast (np.array): Numpy array of forecasted demand
            ordered_amount (list): List of ordered amount for each month
        """
        inventory_levels, optimal_actions, forecast, ordered_amount = inference_values(
            q_table=self.q_table, 
            choices=self.choices, 
            forecast=self.forecast, 
            initial_state=self.initial_stock, 
            min_order=self.min_order)
        return (inventory_levels, optimal_actions, forecast, ordered_amount)
    
    def plot(self):
        """
        Plot the results of the inventory optimization using Q-Learning.

        Args:
            None
        
        Returns:
            None   
        """
        inventory_levels, optimal_actions, forecast, ordered_amount = self.predict()
        plot_results(inventory=inventory_levels, forecast=forecast, ordered=ordered_amount, stock_min=self.security_stock, capacity=self.capacity)