# Weather-Based Q-Learning Simulation Application

## Overview

This project simulates decision-making robots that interact with real-time weather data retrieved from the OpenWeatherMap API. The robots use **Q-learning** to make decisions based on current weather conditions (e.g., temperature, humidity) and continuously improve their actions over time.

The application integrates the **Current Weather API** from OpenWeatherMap and provides a dynamic environment where robots learn through reinforcement learning techniques.

## Table of Contents
- [Purpose](#purpose)
- [How It Works](#how-it-works)
- [Expected Behavior](#expected-behavior)
- [How to Set Up and Run the Application](#how-to-set-up-and-run-the-application)
- [Retrieving City Data from OpenWeatherMap](#retrieving-city-data-from-openweathermap)
- [Dependencies](#dependencies)
- [Configuration File (`config.json`)](#configuration-file-configjson)
- [Known Issues](#known-issues)

## Purpose

The purpose of this project is to simulate the behavior of robots that learn how to make optimal decisions in response to changes in weather conditions. Using **Q-learning**, robots adjust their actions and receive rewards, which helps them improve over multiple episodes.

### Key Features:
- **Real-time weather data**: Retrieves live weather data (temperature, humidity, etc.) using the OpenWeatherMap API.
- **Q-learning**: Reinforcement learning algorithm that improves the decision-making process of robots over time.
- **Scalability**: The application allows you to adjust the number of robots and learning episodes.

## How It Works

1. **Environment Setup**:
   - The application connects to OpenWeatherMap's **Current Weather API** and retrieves the latest weather data for a specific geographic location.
   - Weather data is then transformed into a set of "states" (e.g., temperature buckets) that represent the environment for the Q-learning process.

2. **Q-Learning**:
   - **States**: The state of the environment is defined by the weather conditions, such as temperature, wind speed, or cloudiness.
   - **Actions**: The robots perform actions (such as adjusting temperature) based on the current state.
   - **Rewards**: A reward function evaluates each action's success, and robots update their Q-tables to optimize future actions.
   - Over time, robots learn to make better decisions based on the weather conditions.

3. **Q-Table Output**: At the end of the simulation, the learned Q-tables are saved to a file for further use.

## Expected Behavior

### On Application Start:
- **Weather Data Fetch**: The app will fetch real-time weather data from the OpenWeatherMap API using your API key and location (latitude and longitude).
- **Q-Learning Simulation**: Multiple robots will start interacting with the environment, learning from the weather data.
- **Logging**: You will see logs displaying the progress of the simulation, including total rewards for each robot and updates every few episodes.
- **Q-Tables**: At the end of the simulation, Q-tables will be saved, representing the learning process of each robot.


### How to Register for an OpenWeatherMap API Key

To access weather data using the OpenWeatherMap API, you need to **register for an account** and obtain an API key. Here's how to do it:

1. **Sign up**: Visit the [OpenWeatherMap Sign-Up page](https://home.openweathermap.org/users/sign_up) and fill in your details (email, username, and password).
   
2. **Confirm your email**: After signing up, you will receive a confirmation email. Please confirm your email to activate your account.

3. **Obtain your API key**: Once your account is activated, log in and navigate to the [API keys section](https://home.openweathermap.org/api_keys) in your account. Your default API key will be displayed there. You can also create additional keys for different projects if necessary.

4. **Wait for activation**: It may take some time for your API key to be fully activated and ready for use. Be patient, and ensure your email is confirmed to avoid delays.

### Free Tier and Usage Limits
- **Free License**: OpenWeatherMap provides a **free plan** that allows up to **1,000 API calls per day**. This is ideal for small projects or individual developers.
- For larger applications, you can explore paid plans with higher call limits and additional features by visiting the [OpenWeatherMap Pricing page](https://openweathermap.org/price).

### How to Register for an OpenWeatherMap API Key

To access weather data using the OpenWeatherMap API, you need to **register for an account** and obtain an API key. Here's how to do it:

1. **Sign up**: Visit the [OpenWeatherMap Sign-Up page](https://home.openweathermap.org/users/sign_up) and fill in your details (email, username, and password).
   
2. **Confirm your email**: After signing up, you will receive a confirmation email. Please confirm your email to activate your account.

3. **Obtain your API key**: Once your account is activated, log in and navigate to the [API keys section](https://home.openweathermap.org/api_keys) in your account. Your default API key will be displayed there. You can also create additional keys for different projects if necessary.

4. **Wait for activation**: It may take some time for your API key to be fully activated and ready for use. Be patient, and ensure your email is confirmed to avoid delays.

### Free Tier and Usage Limits
- **Free License**: OpenWeatherMap provides a **free plan** that allows up to **1,000 API calls per day**. This is ideal for small projects or individual developers.
- For larger applications, you can explore paid plans with higher call limits and additional features by visiting the [OpenWeatherMap Pricing page](https://openweathermap.org/price).


## How to Set Up and Run the Application

### Using Conda:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JacquesGariepy/agenticai_openweathermap.git
   cd agenticai_openweathermap
   ```

2. **Create a new Conda environment**:
   Ensure you have `conda` installed. Run the following commands to create and activate a new environment:
   ```bash
   conda create --name weather_qlearning python=3.9
   conda activate weather_qlearning
   ```

3. **Install dependencies**:
   Install the necessary dependencies using `conda` and `pip`:
   ```bash
   conda install pydantic retry autogen numpy requests
   ```

4. **Set up OpenWeatherMap API key**:
   - Register for an account on [OpenWeatherMap](https://home.openweathermap.org/users/sign_up) and get an API key.
   - Place the API key in the `config.json` file under the `openweathermap_api_key` field.

5. **Modify Configuration**:
   Edit the `config.json` file to set the latitude, longitude, and your OpenWeatherMap API key:
   ```json
   {
    "model": "o1-mini",
    "num_robots": 5,
    "alpha": 0.1,
    "gamma": 0.9,
    "initial_epsilon": 1.0,
    "epsilon_decay": 0.9995,
    "min_epsilon": 0.01,
    "num_states": 625,
    "num_actions": 5,
    "episodes": 10000,
    "max_steps_per_episode": 24,
    "save_interval": 1000,
    "log_interval": 100,
    "load_path": null,
    "latitude": 48.8566,
    "longitude": 2.3522,
    "openweathermap_url": "http://api.openweathermap.org/data/2.5/weather",
    "max_calls_per_minute": 60,
    "monthly_quota": 1000000,
    "optimal_state": 312,
    "reward_scale": 10,
    "penalty_scale": 1,
    "log_level": "INFO",
    "use_simulated_env": true,
    "simulation_noise": 0.1
   }
   ```
6. **Creating an `.env` File for API Keys**:

To securely manage your API keys (e.g., OpenAI and OpenWeatherMap keys), you can store them in an `.env` file and load them into your Python environment using the `python-dotenv` package. Here's how you can set it up:

#### Steps to Create and Use the `.env` File

6.1. **Install the `python-dotenv` package**:
   First, you need to install the `python-dotenv` package, which allows you to load environment variables from a `.env` file into Python.
   ```bash
   pip install python-dotenv
   ```

6.2. **Create a `.env` file**:
   In your project directory, create a file called `.env` and add your API keys like this:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
   ```

6.3. **Add `.env` to `.gitignore`**:
   To ensure your API keys are not accidentally committed to version control, add the `.env` file to your `.gitignore`:
   ```bash
   echo ".env" >> .gitignore
   ```

This approach allows you to securely manage and load your API keys without hardcoding them into your code.

7. **Run the application**:
   After setting up the configuration file, run the application:
   ```bash
   python train.py
   ```

### Using Pip (Alternative):
If you prefer not to use `conda`, you can install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

## Retrieving City Data from OpenWeatherMap

You can retrieve weather data for any city using the following steps:

1. **Get API Key**: Create an account on [OpenWeatherMap](https://home.openweathermap.org/users/sign_up) and generate an API key.
2. **Find Latitude and Longitude**: Use the [OpenWeatherMap city search](https://openweathermap.org/find) to find the latitude and longitude of your city.
3. **Update `config.json`**: Replace the latitude and longitude values in your `config.json` file.

Example API Request:
```bash
https://api.openweathermap.org/data/2.5/weather?lat=46.8139&lon=-71.2082&appid=YOUR_API_KEY
```

## Dependencies

Here are the dependencies required for the project:
- **numpy**: Numerical library for managing arrays and performing Q-learning calculations.
- **requests**: To fetch data from the OpenWeatherMap API.
- **pydantic**: For configuration validation.
- **retry**: To retry failed API calls automatically.
- **autogen**: Required for generating and managing robots in the simulation.

## Configuration File (`config.json`)

The configuration file allows you to customize the behavior of the simulation. Here's a breakdown of the fields in `config.json`:
- `model`: The robot model (e.g., "o1-mini").
- `num_robots`: The number of robots in the simulation.
- `alpha`: The learning rate for Q-learning.
- `gamma`: The discount factor for future rewards.
- `epsilon`: The exploration rate (probability of exploring random actions).
- `num_states`: The number of possible weather states (discretized).
- `num_actions`: The number of actions a robot can take in each state.
- `episodes`: The number of learning episodes.
- `save_interval`: How frequently the Q-tables are saved.
- `load_path`: The file path to load a pre-trained Q-table.
- `openweathermap_api_key`: Your API key for OpenWeatherMap.
- `openweathermap_url`: The OpenWeatherMap API endpoint.
- `latitude`, `longitude`: Geographic coordinates of the city.

## Known Issues

- **Invalid API Key**: Make sure your OpenWeatherMap API key is correct, or you’ll encounter a connection error.
- **JSON Parsing Errors**: Ensure the `config.json` file is correctly formatted; otherwise, a `json.decoder.JSONDecodeError` will be raised.
