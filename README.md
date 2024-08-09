# Inventory Optimization using Q-Learning

This repository contains a Python implementation of an inventory optimization model using Q-Learning, a type of Reinforcement Learning (RL) algorithm. The model is designed to help manage inventory levels by making optimal decisions on order quantities based on forecasted demand, initial stock levels, and inventory capacity.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Managing inventory efficiently is crucial for reducing costs and avoiding stockouts or overstocking. This project implements a Q-Learning-based optimizer that learns to make the best inventory decisions over time. It takes into account factors such as forecasted demand, security stock, and inventory capacity to minimize costs and maintain optimal stock levels.

## Features

- **Q-Learning Algorithm**: Implements the Q-Learning algorithm for decision-making.
- **Dynamic Inventory Management**: Adjusts inventory levels based on forecasted consumption dinamycally without tradional rules.
- **Customizable Parameters**: Users can adjust learning rate, discount factor, and exploration rate.
- **Visualizations**: Plots inventory levels, forecast, and order amounts to provide insights into the optimization process.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/inventory-qlearning.git
cd inventory-qlearning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Initialize the Model: Create an instance of the QLearningOptimizer class.
```python
from QLearningOptimizer import QLearningOptimizer

import numpy as np

# Example forecasted demand for 6 months
forecast = np.array([120, 100, 150, 80, 130, 110])

# Initialize the optimizer
optimizer = QLearningOptimizer(
    forecast=forecast,
    initial_stock=200,
    security_stock=50,
    capacity=500,
    n_actions=["minimo", "2minimo", "3minimo", "4minimo", "nopedir"],
    min_order=50,
    alpha=0.1,
    gamma=0.6,
    epsilon=0.1
)

# Train the model
optimizer.fit(epochs=1000)

# Predict inventory levels and actions
inventory_levels, optimal_actions, forecast, ordered_amount = optimizer.predict()

# Plot the results
optimizer.plot()
```

## Customization

You can customize the behavior of the optimizer by adjusting the parameters in the constructor:

- alpha (float): Learning rate (default: 0.1)
- gamma (float): Discount factor (default: 0.6)
- epsilon (float): Exploration rate (default: 0.1)

## Preventing Overstocking

The optimizer includes logic in the __choose_action method to prevent overstocking by avoiding orders when the inventory exceeds 80% of the total capacity.

