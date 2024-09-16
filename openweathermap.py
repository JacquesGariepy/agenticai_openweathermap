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
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration model with validation
class Config(BaseModel):
    model: str
    num_robots: int
    alpha: float
    gamma: float
    epsilon: float
    num_states: int
    num_actions: int
    episodes: int
    save_interval: int
    load_path: Optional[str] = None
    latitude: float
    longitude: float
    openweathermap_url: str = "http://api.openweathermap.org/data/2.5/weather"

    @field_validator('alpha', 'gamma', 'epsilon')
    def check_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Rates must be between 0 and 1')
        return v

    @field_validator('openweathermap_url')
    def check_url(cls, v):
        if not v.startswith("http"):
            raise ValueError("Invalid URL format")
        return v

# Load configuration with validation
def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        return Config(**json.load(f))

config = load_config('config.json')

# Fetch the API key from the environment variables
openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
if not openweathermap_api_key:
    raise ValueError("OPENWEATHERMAP_API_KEY is not set in environment variables.")

# Configure the model LLM and AutoGen agents
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

llm_config = {"model": config.model, "api_key": api_key}

robots = [AssistantAgent(f"robot_{i}", llm_config=llm_config) for i in range(config.num_robots)]

# Initialize or load Q-tables
def init_or_load_q_tables(num_robots: int, num_states: int, num_actions: int, load_path: Optional[str] = None) -> List[np.ndarray]:
    if load_path and os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            return np.load(f, allow_pickle=True)
    return [np.zeros((num_states, num_actions)) for _ in range(num_robots)]

q_tables = init_or_load_q_tables(config.num_robots, config.num_states, config.num_actions, config.load_path)

class Environment:
    def __init__(self):
        self.api_key = openweathermap_api_key
        self.lat = config.latitude
        self.lon = config.longitude
        self.base_url = config.openweathermap_url
        self.setup_connection()

    def setup_connection(self):
        # Construct the full URL with parameters
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        full_url = f"{self.base_url}?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
        logging.info(f"Requesting weather data from: {full_url}")
        print(f"Requesting weather data from: {full_url}")
        
        response = requests.get(self.base_url, params=params)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")
        
        if response.status_code == 200:
            logging.info("Connected to OpenWeatherMap API successfully")
        elif response.status_code == 401:
            raise ConnectionError("Connection failed: Invalid API key.")
        else:
            raise ConnectionError(f"Connection error to OpenWeatherMap API: {response.status_code}")

    def get_state(self) -> int:
        # Fetch current weather data
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        response = requests.get(self.base_url, params=params)
        print(f"Fetching weather data with params: {params}")
        print(f"Response content: {response.content}")
        
        data = response.json()

        # Handle potential errors
        if response.status_code != 200:
            logging.error(f"Error fetching data: {data}")
            raise ValueError("Failed to fetch weather data")

        # Convert temperature to a discrete state
        temperature = data['main']['temp']
        state = int((temperature + 20) / 4)  # Convert temperature range -20°C to 40°C into 0-15 states
        return min(max(state, 0), config.num_states - 1)

    def perform_action(self, action: int) -> Tuple[int, float]:
        # Simulate an action (e.g., adjust a climate control system)
        current_state = self.get_state()
        
        # Simulate new state after action
        new_state = (current_state + action - 2) % config.num_states
        
        # Calculate reward
        reward = -abs(new_state - 7)  # Max reward when temperature is around 15°C (state 7)
        
        return new_state, reward

# Epsilon-greedy policy to choose an action
def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(0, config.num_actions)
    else:
        return np.argmax(q_table[state])

# Update the Q-table
def update_q_table(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int) -> None:
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] = q_table[state, action] + config.alpha * (
        reward + config.gamma * q_table[next_state, best_next_action] - q_table[state, action]
    )

# Training function for a single robot
def train_robot(robot_idx: int, episodes: int, q_table: np.ndarray) -> np.ndarray:
    env = Environment()
    local_epsilon = config.epsilon
    for episode in range(episodes):
        local_epsilon *= 0.99  # Decay epsilon
        state = env.get_state()
        total_reward = 0
        steps = 0
        
        while steps < 24:  # Simulate 24 hours
            action = choose_action(q_table, state, local_epsilon)
            next_state, reward = env.perform_action(action)
            update_q_table(q_table, state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
        
        if episode % 10 == 0:
            logging.info(f"Robot {robot_idx}, Episode {episode}: Total reward = {total_reward}")
    
    return q_table

# Main training function with parallelization
def train() -> List[np.ndarray]:
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_robot, range(config.num_robots), [config.episodes]*config.num_robots, q_tables))
    return results

# Explain an action
def explain_action(robot_idx: int, state: int) -> str:
    q_values = q_tables[robot_idx][state]
    action = np.argmax(q_values)
    explanation = f"In state {state}, robot {robot_idx} chose action {action}.\n"
    explanation += "Q-values for each action:\n"
    for a, q in enumerate(q_values):
        explanation += f"Action {a}: {q:.2f}\n"
    return explanation

# Save Q-tables
def save_q_tables(q_tables: List[np.ndarray], filename: str) -> None:
    np.save(filename, q_tables)
    logging.info(f"Q-tables saved in {filename}")

# Retry decorator with exponential backoff
# Retry decorator with exponential backoff for generating explanations
@retry.retry(exceptions=Exception, tries=5, delay=1, backoff=2)
def generate_explanation_with_retry(robot: AssistantAgent, explanation: str) -> str:
    """
    Generate an explanation with retry mechanism in case of failure.

    Parameters:
    - robot: The AssistantAgent responsible for generating the explanation.
    - explanation: A string representing the current state and Q-table information.

    Returns:
    - The generated explanation from the LLM.
    """
    logging.info(f"Attempting to generate explanation for robot: {robot.name}")
    try:
        # Generate the explanation
        response = robot.generate_init_message(explanation)
        logging.info(f"Explanation generated successfully for robot: {robot.name}")
        return response
    except Exception as e:
        logging.error(f"Error generating explanation for robot {robot.name}: {str(e)}")
        raise  # This will trigger the retry logic

# Main function with error handling and saving
def main():
    global q_tables
    try:
        start_time = time.time()
        q_tables = train()
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Total training time: {training_time:.2f} seconds")

        save_q_tables(q_tables, 'q_tables_final.npy')
        
        # Generate explanations
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
