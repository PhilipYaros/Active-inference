# T-Maze Active Inference Simulation

## Overview
This project implements an Active Inference agent navigating a custom-designed T-maze environment. The environment is a grid-based T-maze with 7 possible locations, adding complexity to the typical T-maze setup. The agent must navigate the maze, receive hints, and plan its actions to maximize rewards while avoiding punishments. This simulation focuses on exploring how changes in the reward, punishment, and neutral parameters (C-matrix) affect the agent's decision-making dynamics.

## Features
- **Custom Grid-Based T-Maze**: The environment consists of 7 location states, forcing the agent to make multi-step decisions, incorporating more complex trial structures.
- **Hint and Reward Mechanism**: The agent receives probabilistic hints about which arm of the maze will offer a reward and navigates the environment accordingly.
- **C-Matrix Exploration**: The reward, punishment, and neutral parameters of the C-matrix are varied to study their impact on agent behavior.
  
  - **Reward Parameter**: Determines the attractiveness of reaching the correct arm.
  - **Punishment Parameter**: Represents the cost of choosing the wrong arm.
  - **Neutral Parameter**: Encodes the "urgency" for the agent to choose an arm. A higher neutral value makes the agent feel less urgency, possibly leading it to gather more information at the hint location, while a more negative value increases urgency, pushing the agent to choose an arm sooner, even with uncertainty.

## Files
- **`Act_Inf_T_maze_preferences.py`**: The main Python file containing the environment setup, Active Inference agent, and parameterized simulations.

## How it Works
1. **Environment Setup**: The T-maze is represented on a 3x5 grid, with specific locations for the start, hint, left arm, and right arm. The agent must move from the start, interpret the hint, and choose an arm to receive a reward (or loss).
   
2. **Agent Behavior**: The agent uses an Active Inference model based on free energy minimization, implemented with the `pymdp` library. It updates its beliefs and actions at each step to maximize expected free energy (balance between epistemic value and pragmatic rewards).

3. **Parameter Exploration**: The reward, punishment, and neutral values in the C-matrix are adjusted to explore how these parameters influence the agent's behavior:
   - **Neutral Parameter's Role**: The neutral parameter introduces urgency. If it is not sufficiently negative, the agent may focus too much on epistemic value (gathering information) by staying in the cue location without moving forward. On the other hand, if the neutral parameter is too negative, the agent may rush to choose an arm even with insufficient certainty.

## Prerequisites
- Python 3.x
- Required Libraries:
  - `pymdp`
  - `matplotlib`
  - `seaborn`
  - `numpy`
  - `pandas`

Install the dependencies using pip:
```bash
pip install pymdp matplotlib seaborn numpy pandas
```

## How to Run
1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd <repo-folder>
    ```

2. Run the simulation script:
    ```bash
    python Act_Inf_T_maze_preferences.py
    ```

3. The simulation will run for multiple rounds with different C-matrix parameter combinations, and the results will be saved to a CSV file (`df_C_1.csv`).

## Results
The simulation results, stored in a CSV file, include:
- **Number of Hints**: How many hints the agent took during the trial.
- **Chosen Arm**: The arm the agent chose at the end of the maze.
- **Outcome**: Whether the agent received a reward or a loss.
- **Belief Posterior**: The agent’s final belief about the correct context.

## Customization
You can modify the parameters in the script to explore different scenarios:
- **p_hint_env**: Probability of receiving a correct hint.
- **p_reward_env**: Probability of receiving a reward at the end of the maze.
- **C-Matrix Parameters**: Reward, punishment, and neutral values can be modified to observe their effects on agent behavior.

## Visualizations
The script includes functions to visualize:
- The agent’s belief distribution over time.
- The agent's current position on the grid.
- The likelihood and transition matrices.

## Contact
For questions or feedback, please reach out at `Philus012@gmail.com`.
