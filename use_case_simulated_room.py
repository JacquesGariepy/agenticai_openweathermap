import numpy as np
import json
import random
from typing import Tuple

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load trained Q-table
q_tables = np.load('q_tables_final.npy', allow_pickle=True)

class SimulatedRoom:
    def __init__(self):
        self.temperature = 22.0  # Starting room temperature
        self.outside_temp = 20.0  # Starting outside temperature
        self.num_states = q_tables[0].shape[0]  # Get the actual number of states from the Q-table

    def get_state(self) -> int:
        # Map the temperature to the available states
        return int((self.temperature - 12) / 20 * (self.num_states - 1))

    def perform_action(self, action: int) -> Tuple[int, float]:
        old_temp = self.temperature
        
        # Update outside temperature (simulating day/night cycle)
        self.outside_temp = 20 + 5 * np.sin(2 * np.pi * (self.get_state() % 24) / 24)
        
        # Action effects
        if action == 0:  # No change
            pass
        elif action == 1:  # Slight increase
            self.temperature += 0.5
        elif action == 2:  # Moderate increase
            self.temperature += 1.0
        elif action == 3:  # Slight decrease
            self.temperature -= 0.5
        elif action == 4:  # Moderate decrease
            self.temperature -= 1.0
        
        # Natural temperature change based on outside temperature
        self.temperature += 0.1 * (self.outside_temp - self.temperature)
        
        # Ensure temperature stays within a reasonable range
        self.temperature = max(12, min(32, self.temperature))
        
        new_state = self.get_state()
        
        # Reward calculation
        optimal_temp = 22.0
        reward = 10 - abs(self.temperature - optimal_temp)
        
        return new_state, reward

    def get_temperature(self) -> float:
        return self.temperature

def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[state])

def main():
    room = SimulatedRoom()
    robot_idx = 0  # Use the first trained robot
    q_table = q_tables[robot_idx]
    epsilon = 0.1  # Small epsilon for some exploration

    print("Simulating room temperature control over 24 hours:")
    print("Hour | Room Temp | Outside Temp | Action | Reward")
    print("-------------------------------------------------")

    for hour in range(24):
        state = room.get_state()
        action = choose_action(q_table, state, epsilon)
        new_state, reward = room.perform_action(action)
        
        actions = ["No change", "Slight increase", "Moderate increase", "Slight decrease", "Moderate decrease"]
        
        print(f"{hour:2d}   | {room.get_temperature():9.2f} | {room.outside_temp:12.2f} | {actions[action]:15s} | {reward:.2f}")

    print("\nFinal room temperature:", room.get_temperature())

if __name__ == "__main__":
    main()