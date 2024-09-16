import numpy as np
import json
import random
from typing import Tuple, List
import matplotlib.pyplot as plt

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load trained Q-table
q_tables = np.load('q_tables_final.npy', allow_pickle=True)

class SimulatedRoom:
    def __init__(self):
        self.temperature = 15.0  # Starting room temperature
        self.outside_temp = 20.0  # Starting outside temperature
        self.num_states = q_tables[0].shape[0]  # Get the actual number of states from the Q-table
        self.optimal_temp = 22.0
        self.humidity = 50.0  # Starting humidity

    def get_state(self) -> int:
        # Map the temperature and humidity to the available states
        temp_component = int((self.temperature - 12) / 20 * (self.num_states // 2))
        humidity_component = int(self.humidity / 100 * (self.num_states // 2))
        return min(temp_component + humidity_component, self.num_states - 1)

    def perform_action(self, action: int) -> Tuple[int, float]:
        old_temp = self.temperature
        
        # Update outside temperature (simulating day/night cycle)
        hour = (self.get_state() % 24)
        self.outside_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)
        
        # Action effects
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
        
        # Natural temperature change based on outside temperature
        self.temperature += 0.1 * (self.outside_temp - self.temperature)
        
        # Update humidity (simplified model)
        self.humidity += random.uniform(-2, 2)
        self.humidity = max(30, min(70, self.humidity))
        
        # Ensure temperature stays within a reasonable range
        self.temperature = max(12, min(32, self.temperature))
        
        new_state = self.get_state()
        
        # Reward calculation
        temp_diff = abs(self.temperature - self.optimal_temp)
        humidity_diff = abs(self.humidity - 50) / 2  # Optimal humidity is 50%
        reward = 20 - temp_diff - humidity_diff
        
        return new_state, reward

    def get_temperature(self) -> float:
        return self.temperature

    def get_humidity(self) -> float:
        return self.humidity

def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[state])

def plot_results(hours: List[int], room_temps: List[float], outside_temps: List[float], humidities: List[float], rewards: List[float]):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    ax1.plot(hours, room_temps, label='Room Temperature')
    ax1.plot(hours, outside_temps, label='Outside Temperature')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.set_title('Temperature Over Time')
    
    ax2.plot(hours, humidities)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Humidity (%)')
    ax2.set_title('Humidity Over Time')
    
    ax3.plot(hours, rewards)
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Over Time')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()

def main():
    room = SimulatedRoom()
    robot_idx = 0  # Use the first trained robot
    q_table = q_tables[robot_idx]
    epsilon = 0.1  # Small epsilon for some exploration

    print("Simulating room temperature control over 24 hours:")
    print("Hour | Room Temp | Outside Temp | Humidity | Action | Reward")
    print("----------------------------------------------------------")

    hours = []
    room_temps = []
    outside_temps = []
    humidities = []
    rewards_list = []

    for hour in range(24):
        state = room.get_state()
        action = choose_action(q_table, state, epsilon)
        new_state, reward = room.perform_action(action)
        
        actions = ["No change", "Very slight increase", "Slight increase", "Moderate increase", 
                   "Very slight decrease", "Slight decrease", "Moderate decrease"]
        
        print(f"{hour:2d}   | {room.get_temperature():9.2f} | {room.outside_temp:12.2f} | {room.get_humidity():8.2f} | {actions[action]:17s} | {reward:.2f}")

        hours.append(hour)
        room_temps.append(room.get_temperature())
        outside_temps.append(room.outside_temp)
        humidities.append(room.get_humidity())
        rewards_list.append(reward)

    print("\nFinal room temperature:", room.get_temperature())
    print("Final humidity:", room.get_humidity())

    plot_results(hours, room_temps, outside_temps, humidities, rewards_list)
    print("\nResults have been plotted and saved as 'simulation_results.png'")

if __name__ == "__main__":
    main()