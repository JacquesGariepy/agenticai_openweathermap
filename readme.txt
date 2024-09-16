# Weather-Based Q-Learning Simulation Application

## Overview

This project is a **Q-learning simulation** that integrates real-time weather data from OpenWeatherMap to simulate decision-making robots. Each robot in the simulation makes decisions based on weather conditions like temperature, cloudiness, and humidity, and uses **Q-learning** to adapt and improve its decisions over time.

The application retrieves weather data from OpenWeatherMap's **Current Weather API** for a given city (based on geographic coordinates) and uses that data to define the environment for the robots.

## Table of Contents
- [Purpose](#purpose)
- [How It Works](#how-it-works)
- [Expected Behavior](#expected-behavior)
- [How to Run the Application](#how-to-run-the-application)
- [Retrieving City Data from OpenWeatherMap](#retrieving-city-data-from-openweathermap)
- [Dependencies](#dependencies)
- [Configuration File (`config.json`)](#configuration-file-configjson)
- [Known Issues](#known-issues)

## Purpose

The main goal of this project is to simulate the behavior of robots that can learn how to make optimal decisions in an environment that changes based on real-world weather conditions. Using **Q-learning**, the robots improve their actions through repeated episodes, and the weather conditions influence the states the robots interact with.

### Key Features
- **Real-time weather data**: Fetches up-to-date weather information (temperature, wind, cloudiness, etc.) from OpenWeatherMap.
- **Q-learning simulation**: Robots use reinforcement learning to make decisions based on current weather conditions and improve their actions over time.
- **Scalability**: The system can handle multiple robots and episodes, and save learned policies in a Q-table.

## How It Works

1. **Environment Setup**: The simulation connects to the OpenWeatherMap API and retrieves current weather data for a specific geographic location (latitude/longitude). This data is used to define the current state of the environment (e.g., temperature range).

2. **Q-Learning Simulation**:
   - **State Representation**: Weather conditions (e.g., temperature) are converted into discrete states.
   - **Actions**: The robots perform actions based on the current state, which could be interpreted as some interaction with the environment (e.g., adjusting air conditioning based on temperature).
   - **Rewards**: Rewards are calculated based on how favorable the action was, for example, minimizing temperature deviations from an ideal value.
   - **Learning**: The robots update their Q-tables to optimize future actions using the rewards they receive.

3. **Training**: The simulation runs for a predefined number of episodes, during which the robots improve their decision-making process by interacting with the environment.

4. **Q-Table Output**: After the simulation, the learned Q-tables are saved for future use.

## Expected Behavior

### On Launch:
- **Weather Connection**: The app will attempt to connect to the OpenWeatherMap API and retrieve weather data for the specified location (using latitude and longitude).
- **Simulation Start**: Multiple robots will start interacting with the environment, learning from the weather data.
- **Learning Output**: Periodic logs will indicate the progress of the simulation, including total rewards for each episode.
- **Saving Q-Tables**: After training, the Q-tables (which store the learned decision-making policies) will be saved to a file.

### Expected Logs:
- Success or failure messages for API connections.
- Episode-by-episode rewards for each robot.
- Final saved Q-tables.

### Error Scenarios:
- If the API key is invalid or the city data is incorrect, you will see a connection error.
- If the `config.json` file is incorrectly formatted, the application will raise a `json.decoder.JSONDecodeError`.

## How to Run the Application

1. **Clone the repository**:
   ```
   git clone https://github.com/JacquesGariepy/agenticai_openweathermap
   cd agenticai_openweathermap
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ and install required dependencies using:
   ```
   pip install --upgrade -r requirements.txt
   ```

3. **Set up OpenWeatherMap API key**:
   You need to create an account on [OpenWeatherMap](https://home.openweathermap.org/users/sign_up) and get an API key. This key must be placed in the `config.json` file under the `openweathermap_api_key` field.

4. **Modify Configuration**:
   Edit the `config.json` file to set the latitude, longitude, and API key:
   ```json
   {
       "model": "o1-mini",
       "num_robots": 3,
       "alpha": 0.1,
       "gamma": 0.9,
       "epsilon": 0.2,
       "num_states": 10,
       "num_actions": 5,
       "episodes": 1000,
       "save_interval": 100,
       "load_path": "",
       "openweathermap_api_key": "YOUR_API_KEY",
       "openweathermap_url": "https://api.openweathermap.org/data/2.5/weather",
       "latitude": 46.8139,
       "longitude": -71.2082
   }
   ```

5. **Run the application**:
   ```
   python openweathermap.py
   ```

## Retrieving City Data from OpenWeatherMap

To get weather data for a specific city, you can use the following steps:
1. **Register for an API key**: [Sign up](https://home.openweathermap.org/users/sign_up) for an account on OpenWeatherMap to get your API key.
2. **Find latitude and longitude**: Visit [OpenWeatherMap's city search](https://openweathermap.org/find) and search for your city. You will find the geographic coordinates (latitude and longitude) of the city.
3. **Update config file**: Place these coordinates into the `latitude` and `longitude` fields in the `config.json`.

### Example Request:
You can retrieve current weather data with this format:
```
https://api.openweathermap.org/data/2.5/weather?lat=46.8139&lon=-71.2082&appid=YOUR_API_KEY
```

## Dependencies

- Python 3.8+
- Required libraries (listed in `requirements.txt`):
  - `numpy`
  - `requests`
  - `pydantic`
  - `retry`

## Configuration File (`config.json`)

Here’s a breakdown of the fields in the `config.json` file:
- `model`: The model to be used by the robots (e.g., "o1-mini").
- `num_robots`: The number of robots participating in the simulation.
- `alpha`: The learning rate for Q-learning.
- `gamma`: The discount factor for future rewards.
- `epsilon`: The exploration rate for Q-learning.
- `num_states`: The number of discrete states in the environment.
- `num_actions`: The number of possible actions for the robots.
- `episodes`: The number of episodes to train the robots.
- `save_interval`: The interval at which the Q-tables will be saved.
- `load_path`: Path to load an existing Q-table (if any).
- `openweathermap_api_key`: Your API key for OpenWeatherMap.
- `openweathermap_url`: The OpenWeatherMap API endpoint.
- `latitude`, `longitude`: Geographic coordinates for the city whose weather data is fetched.

## Known Issues

- **Invalid API Key**: If you use an invalid API key, the app will fail to connect to OpenWeatherMap, and you'll see a "ConnectionError."
- **JSON Format Errors**: Any format errors in the `config.json` file (e.g., missing commas) will raise a `JSONDecodeError`.
