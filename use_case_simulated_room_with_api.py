import os
import numpy as np
import json
import random
import time
import requests
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Configuration loading and environment variables
class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.__dict__.update(config)
        self.openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.openweathermap_api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY is not set in environment variables.")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Initialize the configuration
config = Config('config.json')

# Load Q-table
q_tables = np.load('q_tables_final.npy', allow_pickle=True)

# Simulated environment class
class SimulatedRoom:
    def __init__(self):
        self.temperature = 10.0  # Initial room temperature
        self.outside_temp = 20.0  # Initial outside temperature
        self.humidity = 50.0  # Initial humidity
        self.optimal_temp = 22.0  # Target room temperature
        self.num_states = q_tables[0].shape[0]  # Get number of states

    def get_state(self) -> int:
        # Compute the current state based on temperature and humidity
        temp_component = int((self.temperature - 12) / 20 * (self.num_states // 2))
        humidity_component = int(self.humidity / 100 * (self.num_states // 2))
        return min(temp_component + humidity_component, self.num_states - 1)

    def perform_action(self, action: int) -> Tuple[int, float]:
        old_temp = self.temperature

        # Simulating day/night temperature fluctuations
        hour = int(time.time() % 24)
        self.outside_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)

        # Apply action to modify temperature
        if action == 0:  # No change
            pass
        elif action == 1:  # Very slight increase
            self.temperature += 0.2
        elif action == 2:  # Slight increase
            self.temperature += 0.5
        elif action == 3:  # Moderate increase
            self.temperature += 1.0
        elif action == 4:  # Very slight decrease
            self.temperature -= 0.2
        elif action == 5:  # Slight decrease
            self.temperature -= 0.5
        elif action == 6:  # Moderate decrease
            self.temperature -= 1.0

        # Natural temperature adjustment based on outside temperature
        self.temperature += 0.1 * (self.outside_temp - self.temperature)

        # Update humidity (simplified model)
        self.humidity += random.uniform(-2, 2)
        self.humidity = max(30, min(70, self.humidity))

        # Keep temperature within reasonable bounds
        self.temperature = max(12, min(32, self.temperature))

        new_state = self.get_state()

        # Reward calculation
        temp_diff = abs(self.temperature - self.optimal_temp)
        humidity_diff = abs(self.humidity - 50) / 2  # Optimal humidity is 50%
        reward = 20 - temp_diff - humidity_diff

        return new_state, reward

# Select action using epsilon-greedy policy
def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[state])

# Log results and actions taken at each interval
def log_results(hour: int, temperature: float, outside_temp: float, humidity: float, action: str, reward: float):
    logging.info(f"Hour {hour}: Room Temp: {temperature:.2f}°C, Outside Temp: {outside_temp:.2f}°C, Humidity: {humidity:.2f}%")
    logging.info(f"Action taken: {action}, Reward: {reward:.2f}")

# Main live control loop with periodic check
def run_live_control(check_interval: int = 60):
    room = SimulatedRoom()
    epsilon = 0.1  # Small exploration rate for live use case
    q_table = q_tables[0]  # Select Q-table for the robot

    for hour in range(24):  # Simulating for 24 hours
        state = room.get_state()
        action = choose_action(q_table, state, epsilon)
        new_state, reward = room.perform_action(action)

        actions = ["No change", "Very slight increase", "Slight increase", "Moderate increase", 
                   "Very slight decrease", "Slight decrease", "Moderate decrease"]

        # Log action and state every hour
        log_results(hour, room.temperature, room.outside_temp, room.humidity, actions[action], reward)

        # Wait for the next check based on the interval set (in minutes)
        time.sleep(check_interval * 60)  # Sleep for x minutes before the next check

    logging.info("Live control completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_live_control(check_interval=5)  # Check every 5 minutes
