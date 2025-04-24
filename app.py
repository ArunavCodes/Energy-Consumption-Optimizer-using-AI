import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# Optimized and Extended Deep Q-Learning for Energy Consumption Reduction
class Environment:
    def __init__(self, optimal_temperature=(18.0, 24.0), initial_month=0, initial_number_users=10, initial_rate_data=60):
        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.optimal_temperature = optimal_temperature
        self.min_temperature, self.max_temperature = -20, 80
        self.min_number_users, self.max_number_users, self.max_update_users = 10, 100, 5
        self.min_rate_data, self.max_rate_data, self.max_update_data = 20, 300, 10
        self.initial_month = initial_month
        self.initial_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.reset(initial_month)

    def update_env(self, direction, energy_ai, month):
        energy_noai = 0
        if self.temperature_noai < self.optimal_temperature[0]:
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif self.temperature_noai > self.optimal_temperature[1]:
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]

        self.reward = 1e-3 * (energy_noai - energy_ai)
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]

        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        self.current_number_users = np.clip(self.current_number_users, self.min_number_users, self.max_number_users)

        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        self.current_rate_data = np.clip(self.current_rate_data, self.min_rate_data, self.max_rate_data)

        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature

        delta_temperature_ai = energy_ai if direction == 1 else -energy_ai
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        self.temperature_noai += delta_intrinsic_temperature

        if self.temperature_ai < self.min_temperature:
            self.temperature_ai = self.optimal_temperature[0]
            self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
        elif self.temperature_ai > self.max_temperature:
            self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
            self.temperature_ai = self.optimal_temperature[1]

        self.total_energy_ai += energy_ai
        self.total_energy_noai += energy_noai

        next_state = np.array([
            (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature),
            (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users),
            (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        ]).reshape(1, -1)

        return next_state, self.reward

    def reset(self, new_month):
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = sum(self.optimal_temperature) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0

    def observe(self):
        return np.array([
            (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature),
            (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users),
            (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        ]).reshape(1, -1)

def run_simulation(model_path=None):
    # Initialize environment
    env = Environment()
    
    # Load model if provided
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        st.warning("No model found. Using random actions for demonstration.")
        model = None
    
    # Initialize logs
    temp_logs_ai = []
    temp_logs_noai = []
    energy_logs_ai = []
    energy_logs_noai = []
    
    # Simulation parameters
    epochs = 4
    timesteps_per_epoch = 50
    epsilon = 0.3
    direction_boundary = 2
    temperature_step = 1.5
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        env.reset(np.random.randint(0, 12))
        state = env.observe()
        total_reward = 0
        
        for timestep in range(timesteps_per_epoch):
            # Update progress
            progress = (epoch * timesteps_per_epoch + timestep) / (epochs * timesteps_per_epoch)
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs}, Timestep {timestep+1}/{timesteps_per_epoch}")
            
            # Choose action
            if model is None or np.random.rand() < epsilon:
                action = np.random.randint(0, 5)
            else:
                q_values = model.predict(state, verbose=0)[0]
                action = np.argmax(q_values)
            
            # Update environment
            direction = -1 if action < direction_boundary else 1
            energy_ai = abs(action - direction_boundary) * temperature_step
            next_state, reward = env.update_env(direction, energy_ai, timestep % 12)
            
            # Log data
            temp_logs_ai.append(env.temperature_ai)
            temp_logs_noai.append(env.temperature_noai)
            energy_logs_ai.append(env.total_energy_ai)
            energy_logs_noai.append(env.total_energy_noai)
            
            state = next_state
            total_reward += reward
    
    # Display results
    st.success(f"Simulation complete! Final AI Energy: {env.total_energy_ai:.2f}, No AI Energy: {env.total_energy_noai:.2f}")
    
    # Plot results
    st.subheader("Temperature Comparison")
    fig1, ax1 = plt.subplots()
    ax1.plot(temp_logs_ai, label="AI Temperature")
    ax1.plot(temp_logs_noai, label="No AI Temperature")
    ax1.legend()
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Temperature")
    st.pyplot(fig1)
    
    st.subheader("Energy Consumption Comparison")
    fig2, ax2 = plt.subplots()
    ax2.plot(energy_logs_ai, label="AI Energy")
    ax2.plot(energy_logs_noai, label="No AI Energy")
    ax2.legend()
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Energy")
    st.pyplot(fig2)
    
    # Calculate and display savings
    if env.total_energy_noai > 0:
        savings = (env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100
        st.metric("Energy Savings", f"{savings:.2f}%")

def main():
    st.title("AI-Powered Data Center Energy Optimization")
    st.markdown("""
    This application demonstrates how Deep Q-Learning can be used to optimize energy consumption 
    in data centers by intelligently managing cooling systems.
    """)
    
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Path to trained model (optional)", "energy_optimizer_dqn.h5")
    
    if st.button("Run Simulation"):
        st.write("Running simulation...")
        run_simulation(model_path)

if __name__ == "__main__":
    main()
