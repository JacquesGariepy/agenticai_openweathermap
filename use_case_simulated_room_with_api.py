import os
import numpy as np
from autogen import AssistantAgent  # Import AutoGen for agent handling
import logging
import json
from typing import List, Tuple, Optional
from flask import Flask, jsonify, request, render_template
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from threading import Lock
import random
import time

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app initialization
app = Flask(__name__)

# Configuration class (similar to the original code)
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

# Initialize configuration
config = Config('config.json')

# Initialize Q-tables (same as before)
q_tables = np.load('q_tables_final.npy', allow_pickle=True)

# Initialize agents with AutoGen
llm_config = {"model": config.model, "api_key": config.api_key}
agents = [AssistantAgent(f"agent_{i}", llm_config=llm_config) for i in range(config.num_robots)]

class SimulatedEnvironment:
    def __init__(self):
        self.state = random.randint(0, config.num_states - 1)
        self.reward_history = []  # Track rewards for plotting
        self.last_action = None

    def get_state(self) -> int:
        # Simulate small changes in the environment
        self.state = max(0, min(config.num_states - 1, self.state + random.randint(-1, 1)))
        return self.state

    def perform_action(self, action: int) -> Tuple[int, float]:
        current_state = self.get_state()
        
        # Simulate state transition based on action
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

        # Reward based on proximity to optimal state
        optimal_state = config.num_states // 2
        reward = max(0, 20 - abs(new_state - optimal_state))
        self.reward_history.append(reward)
        self.last_action = action
        return new_state, reward

# Initialize a simulated environment
env = SimulatedEnvironment()

# Choose action using epsilon-greedy policy
def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(0, config.num_actions)
    else:
        return np.argmax(q_table[state])

# Update Q-table based on actions
def update_q_table(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int) -> None:
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] = q_table[state, action] + config.alpha * (
        reward + config.gamma * q_table[next_state, best_next_action] - q_table[state, action]
    )

# Save performance graph
def save_performance_graph():
    plt.plot(env.reward_history)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Performance over time')
    plt.savefig('static/performance.png')
    plt.close()

# Use AutoGen to generate explanations and coordinate agents
def generate_explanation(agent_idx: int, state: int):
    agent = agents[agent_idx]
    q_values = q_tables[agent_idx][state]
    action = np.argmax(q_values)
    actions = ["No change", "Slight increase", "Moderate increase", "Slight decrease", "Moderate decrease"]
    
    # Use AutoGen to explain the decision
    explanation_prompt = (
        f"The agent chose {actions[action]} in state {state}. "
        f"Explain the rationale behind this decision and suggest improvements."
    )
    response = agent.generate_init_message(explanation_prompt)
    logging.info(f"Agent {agent_idx} explanation: {response}")
    return response

# Dashboard route
@app.route('/')
def index():
    return render_template('index.html', epsilon=config.initial_epsilon, num_episodes=config.episodes)

# Route to update parameters dynamically
@app.route('/update_params', methods=['POST'])
def update_params():
    config.initial_epsilon = float(request.form['epsilon'])
    config.episodes = int(request.form['episodes'])
    return jsonify({"status": "Parameters updated successfully"})

# Route to start training and use AutoGen for explanation
@app.route('/train', methods=['POST'])
def start_training():
    global q_tables
    epsilon = config.initial_epsilon
    state = env.get_state()
    
    for _ in range(config.episodes):
        action = choose_action(q_tables[0], state, epsilon)
        next_state, reward = env.perform_action(action)
        update_q_table(q_tables[0], state, action, reward, next_state)
        state = next_state
    
    save_performance_graph()
    
    # Generate explanations for the last actions
    explanation = generate_explanation(0, state)
    
    return jsonify({"status": "Training completed", "explanation": explanation})

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
