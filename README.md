# AI-Powered Data Center Energy Optimization

## Overview
This project demonstrates the use of Artificial Intelligence, specifically Deep Q-Learning, to optimize energy consumption in data centers by intelligently managing cooling systems. By leveraging reinforcement learning techniques, the system successfully reduces energy usage while maintaining optimal temperatures.

## Features
- Simulation of a dynamic data center environment with variables like atmospheric temperature, user data rates, and intrinsic temperature.
- Deep Q-Learning model for optimizing cooling systems.
- Streamlit app for interactive deployment and visualization of energy optimization results.
- Graphical comparison of AI-driven vs non-AI energy consumption and temperature regulation.
- Calculation and display of energy savings achieved with AI use.

## Technologies Used
- **Python**: Programming language for implementing the model and app.
- **Keras**: For building and training the neural network.
- **Streamlit**: Interactive platform for app deployment.
- **Matplotlib**: Visualization of simulation results (graphs and metrics).
- **NumPy**: Array manipulation for handling environment states.

## How It Works

### Model Training
The uploaded `AI&dSPROJECT.pynb` contains the implementation of the Deep Q-Learning model for optimizing energy consumption in cooling systems. Below is an overview of the key components:

1. **Environment Class**:
   - Simulates a data center environment with realistic parameters.
   - Variables include:
     - **Optimal Temperature Range**: Defined limits for efficient server operation (e.g., 18°C–24°C).
     - **Monthly Atmospheric Temperatures**: Mimics real-world seasonal fluctuations.
     - **Data Rates and User Counts**: Represents fluctuating workloads affecting cooling needs.

2. **Deep Q-Learning Process**:
   - **State Representation**: Uses normalized environment metrics like temperature and data rates.
   - **Action Space**: Includes cooling adjustments to minimize energy usage.
   - **Reward Mechanism**: Rewards energy-efficient actions while maintaining optimal temperature conditions.

3. **Training Workflow**:
   - Defines neural network architecture with multiple layers to process state-action pairs effectively.
   - Utilizes hyperparameters like learning rate, discount factor, and exploration rate for reinforcement learning.
   - Runs simulations across epochs and timesteps, adjusting the environment dynamically.

4. **Outputs**:
   - Temperature and energy consumption graphs compare AI-driven cooling against traditional methods.
   - The trained model (`energy_optimizer_dqn.h5`) is saved for deployment in the Streamlit app.

### Streamlit App
The `app.py` deploys the trained model in an interactive interface, allowing users to simulate and visualize the AI-driven optimization process. Here's how it works:

1. **Environment Setup**:
   - Initializes the same simulation parameters as in the training code.
   - Handles dynamic user data rates, atmospheric temperatures, and intrinsic temperature calculations.

2. **Model Loading**:
   - Provides an option to load the pre-trained AI model (`energy_optimizer_dqn.h5`) for simulation.
   - Uses random actions for demonstration purposes if the model file is unavailable.

3. **Simulation**:
   - Runs multiple epochs and timesteps to replicate cooling system performance under varying conditions.
   - Logs:
     - **Temperature Data**: AI-regulated and non-AI baseline values.
     - **Energy Data**: Cumulative consumption logs for AI-driven and traditional cooling systems.

4. **Visualization**:
   - Graphs and metrics dynamically display:
     - **Temperature Comparison**: AI-regulated vs non-AI cooling.
     - **Energy Savings**: Percentage of energy reduction achieved.
   - Calculates and displays energy savings in real-time.

5. **User Interaction**:
   - Provides an interactive experience through Streamlit features:
     - **Progress Visualization**: Displays progress during simulation.
     - **Dynamic Metrics**: Shows real-time updates of energy savings and cooling performance.

This system showcases the practical application of AI in solving real-world challenges, providing meaningful insights into energy optimization techniques for data centers.
## Results

The project demonstrates the potential of AI in optimizing energy consumption within data centers through the use of Deep Q-Learning. Below are the key results obtained from the simulation:

### Temperature Comparison
- **AI-Regulated Cooling**: The Deep Q-Learning model effectively maintains the temperature within the optimal range (18°C–24°C).
- **Non-AI Baseline**: Traditional cooling methods often overshoot or undershoot the optimal temperature, leading to inefficiencies.

### Energy Consumption Comparison
- **AI-Driven Consumption**: The AI system consistently exhibits lower energy usage due to intelligent decision-making.
- **Non-AI Baseline Consumption**: Traditional methods consume significantly more energy because of a lack of optimization.

### Energy Savings
- The AI system reduces cumulative energy consumption by approximately **17%** compared to the non-AI baseline, as calculated using the following formula:  
  $$ \text{Savings (\%)} = \frac{\text{Non-AI Energy} - \text{AI Energy}}{\text{Non-AI Energy}} \times 100 $$

  Example:
  - AI Energy Consumption: **225.50 units**  
  - Non-AI Energy Consumption: **273.00 units**  
  - **Savings**: ~17.39%

### Graphs
The simulation outputs the following visualizations:
1. **Temperature Comparison**: A line graph comparing AI-regulated temperatures against baseline (non-AI) values over simulation timesteps.
2. **Energy Consumption Comparison**: A cumulative energy usage graph comparing AI-driven and non-AI energy consumption.

### Key Insights
- AI regulates temperature effectively, ensuring operational efficiency.
- Significant energy savings demonstrate the capability of the Deep Q-Learning model to optimize energy-intensive systems.
- The results affirm the scalability and adaptability of this approach to real-world data center environments.

These results highlight the transformative potential of AI-driven solutions in addressing challenges associated with energy management.

