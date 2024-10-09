Here’s a `README.md` file based on your project, which appears to simulate active inference in a T-maze environment using Python:

```markdown
# Active Inference in a T-Maze Environment

This project simulates an agent's decision-making process in a T-maze environment using **Active Inference**. The environment is implemented as a grid world, and the agent makes decisions based on prior beliefs, observations, and rewards. The simulation is run through multiple rounds with varying parameters to analyze the agent's behavior.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Simulation Overview](#simulation-overview)
- [Key Classes and Functions](#key-classes-and-functions)
- [Running the Simulation](#running-the-simulation)
- [Outputs](#outputs)
- [License](#license)

## Installation

### Prerequisites
Ensure you have Python 3.x installed along with the following libraries:
- `numpy`
- `matplotlib`
- `seaborn`
- `pandas`
- `pymdp`

You can install these dependencies using pip:
```bash
pip install numpy matplotlib seaborn pandas pymdp
```

## Project Structure
- **Act_Inf_T_maze_preferences.py**: Main script containing the environment setup, agent's active inference loop, and simulation parameters.
- **Environment setup**: Defines the grid world for the T-maze and the agent's possible actions.
- **Simulation logic**: The core functions that define agent behavior and belief updating.

## Simulation Overview
The simulation models an agent navigating a grid world with a T-maze layout. The agent receives rewards or losses depending on the context (which arm of the maze is better). The agent uses **active inference** to infer its current state and decide on actions that maximize expected rewards.

### Key Features:
- The agent is modeled to handle uncertain environments.
- The agent has access to probabilistic hints about which arm of the maze will provide a better reward.
- The simulation runs multiple rounds with different combinations of reward, punishment, and neutral parameters.

## Key Classes and Functions

### Classes
- **Tmaze_grid**: Defines the T-maze environment, including how the agent moves through the grid, receives hints, and observes rewards.
  
### Functions
- **create_A()**: Generates the observation likelihood matrix (how the agent perceives the environment).
- **create_B()**: Defines the agent’s state transitions (how the agent's actions change its state).
- **create_C()**: Constructs the agent’s reward matrix (how rewards and punishments affect the agent’s decisions).
- **create_D()**: Defines the agent's initial beliefs about its context and location.
- **run_active_inference_loop()**: Executes the active inference loop, where the agent continuously updates its beliefs and chooses actions based on expected free energy.

## Running the Simulation

To run the simulation, execute the script:
```bash
python Act_Inf_T_maze_preferences.py
```

The simulation will run 10 rounds with 7 steps each, for various combinations of reward, punishment, and neutral parameters. The agent’s behavior will be recorded and saved to a CSV file.

### Key Parameters
- `N_rounds`: Number of rounds for each parameter combination.
- `N_steps`: Number of steps in each round.
- `reward_pars`: List of reward parameter values.
- `pun_pars`: List of punishment parameter values.
- `neutral_pars`: List of neutral parameter values.

## Outputs
The results are saved as a CSV file (`df_C.csv`). Each row in the CSV represents a single trial, with the following columns:
- `Reward_par`: The reward parameter used in the trial.
- `Pun_par`: The punishment parameter used in the trial.
- `Neutral_par`: The neutral parameter used in the trial.
- `n`: Trial number.
- `Context`: The context for the trial (which arm was better).
- `N_hints`: Number of hints the agent used.
- `Chosen_Arm`: The arm chosen by the agent.
- `Outcome`: The outcome (reward or loss) of the trial.
- `Posterior`: The agent’s posterior belief about the context at the end of the trial.

## License
This project is open-source and licensed under the MIT License.
```

### Notes:
1. Ensure you adjust file paths and outputs as needed (e.g., where the CSV file is saved).
2. Replace the placeholders with your project's actual paths and details if necessary.
