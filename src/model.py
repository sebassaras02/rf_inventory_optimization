import numpy as np

class InventoryOptimizerQLearning:
    def __init__(self, forecast, initial_stock, security_stock, n_actions, min_order, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.choices = np.array(n_actions) 
        self.min_order = min_order
        self.forecast = forecast
        self.initial_stock = initial_stock
        self.security_stock = security_stock
    
    def __transition(self, state, action, consuption):
        new_state = state

        # Add the order to the state
        if action == "minimo":
            new_state += self.min_order
        elif action == "2minimo":
            new_state += 2 * self.min_order
        elif action == "3minimo":
            new_state += 3 * self.min_order
        elif action == "4minimo":
            new_state += 4 * self.min_order
        elif action == "nopedir":
            new_state = new_state
        
        # Reduce the state by the consuption
        new_state -= consuption
        return new_state
    
    def __create_q_table(self):
        # Create the Q-table
        self.q_table = np.zeros((len(self.forecast), len(self.choices)))
    
    def __get_reward(self, current_state, threshold):
        if current_state <= threshold:
            return -10
        elif current_state > threshold and current_state <= 1.5*threshold:
            return 100
        elif current_state > 1.5*threshold and current_state <= 2*threshold:
            return 0.5
        elif current_state > 2*threshold and current_state <= 3*threshold:
            return 0.25
        else:
            return -50
    
    def __choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.choices)  # Exploration
        else:
            max_index = np.argmax(self.q_table[state])  # Explotation
            return self.choices[max_index]
    
    def __update_q_table(self, action, reward, unit):
        self.q_table[unit, self.choices.index(action)] =  self.q_table[unit, self.choices.index(action)] + self.alpha * (reward + self.gamma * np.max(self.q_table[unit+1]) - self.q_table[unit, self.choices.index(action)])


    
    def fit(self, epochs=1000):
        for epoch in range(epochs):
            state = self.initial_stock
            for unit in range(len(self.forecast)):
                # Choose an action
                action = self.__choose_action(state=state)
                # Transition to the new state
                new_state = self.__transition(state=state, action=action, consuption=self.forecast[unit])
                # Get the reward
                reward = self.__get_reward(current_state=new_state, threshold=self.security_stock)
                # Update the Q-table
                self.__update_q_table(action=action, reward=reward, unit=unit)
                # Update the state
                state = new_state
