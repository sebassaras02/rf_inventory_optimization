# 📊 Inventory Optimization using Q-Learning

This repository contains a Python implementation of an inventory optimization model using Q-Learning, a Reinforcement Learning (RL) algorithm. The model is designed to help manage inventory levels by making optimal decisions on order quantities based on forecasted demand, initial stock levels, and inventory capacity.

## 🚀 Introduction

Efficient inventory management is crucial for reducing costs and avoiding stockouts or overstocking. This project implements a Q-Learning-based optimizer that learns to make optimal inventory decisions over time. It considers factors such as forecasted demand, security stock, and inventory capacity to minimize costs and maintain optimal stock levels.

## ✨ Features

- 🧠 **Q-Learning Algorithm**: Implements Q-Learning for decision-making based on temporal difference learning.
- 🔄 **Dynamic Inventory Management**: Adjusts inventory levels based on forecasted consumption dynamically without traditional rules.
- 🛠️ **Customizable Parameters**: Adjustable learning rate, discount factor, and exploration rate.
- 📈 **Visualizations**: Plots inventory levels, forecast, and order amounts to provide insights into the optimization process.

## 🔧 Usage
Initialize the Model

Create an instance of the QLearningOptimizer class.

To create any model for inventory optimization, you have to follow this:

1. Create a forecasting model and predict the future consuption.
2. You have to know the limitations of your system such as security stock, maximum level of stock, initial stock, lead time, and the minimal order quantity.
3. The actions are limited based from 1 to 6 times the minimal order quantity. 


```python
from QLearningOptimizer import QLearningOptimizer
import numpy as np

# Example forecasted demand for 6 months
forecast = np.array([400, 325, 356, 210, 150, 400])

# Initial conditions of the system
initial_state = {
 "stock": 800,
 "leadTime": 2,
 "minimumOrder": 100,
 "securityStock": 200,
 "maximalCapacity": 1000
}

# Define the actions
actions = ["no", "m", "2m", "3m", "4m", "5m", "6m"]

# Initialize the optimizer
model = QLearningOptimizer(
 forecast=forecast, 
 initial_stock=initial_state["stock"], 
 security_stock=initial_state["securityStock"],
 capacity=initial_state["maximalCapacity"],
 n_actions=actions, 
 min_order=initial_state["minimumOrder"], 
 lead_time=initial_state["leadTime"],
 alpha=0.1, 
 gamma=0.6, 
 epsilon=0.1
)

# Train the model
model.fit(epochs=1000)

# Predict inventory levels and actions
predictions = model.predict()

# Plot the results
model.plot("bar")
```

## ⚙️ Customization

You can customize the behavior of the agent modifying the parameters for training.

- alpha (float): Learning rate (default: 0.1)
- gamma (float): Discount factor (default: 0.6)
- epsilon (float): Exploration rate (default: 0.1)

## ☕ Support the Project

If you find this inventory optimization tool helpful and would like to support its continued development, consider buying me a coffee. Your support helps maintain and improve this project!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.paypal.com/paypalme/sebassarasti)

### Other Ways to Support
- ⭐ Star this repository
- 🍴 Fork it and contribute
- 📢 Share it with others who might find it useful
- 🐛 Report issues or suggest new features

Your support, in any form, is greatly appreciated! 🙏