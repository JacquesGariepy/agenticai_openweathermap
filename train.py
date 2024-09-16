import os
import numpy as np
from autogen import AssistantAgent
import logging
import json
from typing import List, Tuple, Optional
from pydantic import BaseModel, field_validator
from concurrent.futures import ProcessPoolExecutor
import retry
import time
import requests
from dotenv import load_dotenv
from threading import Lock
import random

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config(BaseModel):
    model: str
    num_robots: int
    alpha: float
    gamma: float
    initial_epsilon: float
    epsilon_decay: float
    min_epsilon: float
    num_states: int
    num_actions: int
    episodes: int
    max_steps_per_episode: int
    save_interval: int
    log_interval: int
    load_path: Optional[str] = None
    latitude: float
    longitude: float
    openweathermap_url: str
    max_calls_per_minute: int
    monthly_quota: int
    optimal_state: int
    reward_scale: float
    penalty_scale: float
    log_level: str
    use_simulated_env: bool
    simulation_noise: float

    @field_validator('alpha', 'gamma', 'initial_epsilon', 'epsilon_decay', 'min_epsilon', 'simulation_noise')
    def check_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Rates must be between 0 and 1')
        return v

    @field_validator('openweathermap_url')
    def check_url(cls, v):
        if not v.startswith("http"):
            raise ValueError("Invalid URL format")
        return v

    @field_validator('log_level')
    def check_log_level(cls, v):
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
        return v

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        return Config(**json.load(f))

config = load_config('config.json')

openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not openweathermap_api_key:
    raise ValueError("OPENWEATHERMAP_API_KEY is not set in environment variables.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

llm_config = {"model": config.model, "api_key": api_key}

robots = [AssistantAgent(f"robot_{i}", llm_config=llm_config) for i in range(config.num_robots)]

def init_or_load_q_tables(num_robots: int, num_states: int, num_actions: int, load_path: Optional[str] = None) -> List[np.ndarray]:
    if load_path and os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            return np.load(f, allow_pickle=True)
    return [np.zeros((num_states, num_actions)) for _ in range(num_robots)]

q_tables = init_or_load_q_tables(config.num_robots, config.num_states, config.num_actions, config.load_path)

class SimulatedEnvironment:
    def __init__(self):
        self.state = random.randint(0, config.num_states - 1)

    def get_state(self) -> int:
        # Simulate small changes in the environment
        self.state = max(0, min(config.num_states - 1, self.state + random.randint(-1, 1)))
        return self.state

    def perform_action(self, action: int) -> Tuple[int, float]:
        current_state = self.get_state()
        
        if action == 0:  # No change
            new_state = current_state
        elif action == 1:  # Slight increase
            new_state = min(current_state + 1, config.num_states - 1)
        elif action == 2:  # Moderate increase
            new_state = min(current_state + 2, config.num_states - 1)
        elif action == 3:  # Slight decrease
            new_state = max(current_state - 1, 0)
        elif action == 4:  # Moderate decrease
            new_state = max(current_state - 2, 0)
        
        optimal_state = config.num_states // 2
        distance_to_optimal = abs(new_state - optimal_state)
        previous_distance = abs(current_state - optimal_state)
        
        if distance_to_optimal < previous_distance:
            reward = 10 - distance_to_optimal  # Reward for moving closer to optimal
        elif distance_to_optimal > previous_distance:
            reward = -distance_to_optimal  # Penalty for moving away from optimal
        else:
            reward = 0  # No change in distance
        
        return new_state, reward

class RealEnvironment:
    _lock = Lock()
    _calls_made = 0
    _last_reset = time.time()

    def __init__(self):
        self.api_key = openweathermap_api_key
        self.lat = config.latitude
        self.lon = config.longitude
        self.base_url = config.openweathermap_url

    @classmethod
    def throttle(cls):
        with cls._lock:
            current_time = time.time()
            if current_time - cls._last_reset >= 60:
                cls._calls_made = 0
                cls._last_reset = current_time
            
            if cls._calls_made >= config.max_calls_per_minute:
                sleep_time = 60 - (current_time - cls._last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cls._calls_made = 0
                cls._last_reset = time.time()
            
            cls._calls_made += 1

            if cls._calls_made > config.monthly_quota:
                raise ValueError("Monthly API call quota exceeded.")

    def get_state(self) -> int:
        self.throttle()
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            pressure = data['main']['pressure']

            temp_state = int((temperature + 20) / 4)
            humidity_state = int(humidity / 20)
            wind_state = int(wind_speed / 2)
            pressure_state = int((pressure - 950) / 10)

            combined_state = temp_state * 125 + humidity_state * 25 + wind_state * 5 + pressure_state
            return min(max(combined_state, 0), config.num_states - 1)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching weather data: {str(e)}")
            logging.error(f"Response content: {response.content if 'response' in locals() else 'N/A'}")
            raise ValueError("Failed to fetch weather data")

    def perform_action(self, action: int) -> Tuple[int, float]:
        current_state = self.get_state()
        
        if action == 0:  # No change
            new_state = current_state
        elif action == 1:  # Slight increase
            new_state = min(current_state + 1, config.num_states - 1)
        elif action == 2:  # Moderate increase
            new_state = min(current_state + 2, config.num_states - 1)
        elif action == 3:  # Slight decrease
            new_state = max(current_state - 1, 0)
        elif action == 4:  # Moderate decrease
            new_state = max(current_state - 2, 0)
        
        optimal_state = config.num_states // 2
        reward = max(0, 20 - abs(new_state - optimal_state))
        
        return new_state, reward

def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(0, config.num_actions)
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int) -> None:
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] = q_table[state, action] + config.alpha * (
        reward + config.gamma * q_table[next_state, best_next_action] - q_table[state, action]
    )

def train_robot(robot_idx: int, episodes: int, q_table: np.ndarray) -> np.ndarray:
    env = SimulatedEnvironment() if config.use_simulated_env else RealEnvironment()
    epsilon = config.initial_epsilon
    epsilon_decay = config.epsilon_decay
    min_epsilon = config.min_epsilon
    for episode in range(episodes):
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        state = env.get_state()
        total_reward = 0
        steps = 0
        
        while steps < config.max_steps_per_episode:
            action = choose_action(q_table, state, epsilon)
            next_state, reward = env.perform_action(action)
            update_q_table(q_table, state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
        
        if episode % config.log_interval == 0:
            logging.info(f"Robot {robot_idx}, Episode {episode}: Total reward = {total_reward}, Epsilon = {epsilon:.4f}")
    
    return q_table

def train() -> List[np.ndarray]:
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_robot, range(config.num_robots), [config.episodes]*config.num_robots, q_tables))
    return results

def explain_action(robot_idx: int, state: int) -> str:
    q_values = q_tables[robot_idx][state]
    action = np.argmax(q_values)
    actions = ["No change", "Slight increase", "Moderate increase", "Slight decrease", "Moderate decrease"]
    explanation = f"In state {state}, robot {robot_idx} chose action {action} ({actions[action]}).\n"
    explanation += "Q-values for each action:\n"
    for a, q in enumerate(q_values):
        explanation += f"{actions[a]}: {q:.2f}\n"
    return explanation

def save_q_tables(q_tables: List[np.ndarray], filename: str) -> None:
    timestamped_filename = f'{filename}_{int(time.time())}.npy'
    np.save(timestamped_filename, q_tables)
    logging.info(f"Q-tables saved in {timestamped_filename}")

@retry.retry(exceptions=Exception, tries=5, delay=1, backoff=2)
def generate_explanation_with_retry(robot: AssistantAgent, explanation: str) -> str:
    logging.info(f"Attempting to generate explanation for robot: {robot.name}")
    logging.info(f"Explanation being sent: {explanation}")
    try:
        response = robot.generate_init_message(explanation)
        logging.info(f"Explanation generated successfully for robot: {robot.name}")
        return response
    except Exception as e:
        logging.error(f"Error generating explanation for robot {robot.name}: {str(e)}")
        raise

def main():
    global q_tables
    try:
        start_time = time.time()
        q_tables = train()
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Total training time: {training_time:.2f} seconds")

        save_q_tables(q_tables, 'q_tables_final')
        
        for robot_idx in range(config.num_robots):
            for state in range(config.num_states):
                explanation = explain_action(robot_idx, state)
                try:
                    response = generate_explanation_with_retry(robots[robot_idx], explanation)
                    logging.info(f"Robot {robot_idx} explanation for state {state}:\n{response}")
                except Exception as e:
                    logging.error(f"Failed to generate explanation for robot {robot_idx}, state {state}: {str(e)}")

        logging.info("Training completed and Q-tables saved.")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
