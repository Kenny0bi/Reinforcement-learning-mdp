# Project 2: Reinforcement Learning for Markov Decision Processes

## Overview
This project aims to develop optimal policies for three Markov decision processes (MDPs) using batch reinforcement learning, where transition data has been provided. The project consists of three datasets (`small.csv`, `medium.csv`, and `large.csv`) representing increasingly complex environments. Our goal is to maximize the total expected reward by designing deterministic policies.

## Datasets
- **small.csv**: 10x10 grid world (100 states) with 4 actions (left, right, up, down).
- **medium.csv**: MountainCar environment with 50,000 states and 7 actions representing different acceleration levels.
- **large.csv**: A hidden-structure MDP with 302,020 states and 9 actions.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`

## Steps Completed

### Step 1: Data Loading and Cleaning
- **Objective**: Load the dataset, clean it, and prepare it for analysis.
- **Actions Taken**:
  - Loaded `small.csv` using `pandas` and assigned column names: `state`, `action`, `reward`, `next_state`.
  - Dropped rows with non-numeric values in the `state` or `action` columns.
  - Converted columns to their appropriate data types: `int` for `state`, `action`, and `next_state`, and `float` for `reward`.
- **Result**: Cleaned dataset with 50,000 rows and 4 columns.

### Step 2: Preliminary Analysis of the Dataset
- **Objective**: Explore the dataset to understand reward distribution and transition patterns.
- **Actions Taken**:
  - Displayed a preview of the data to identify reward structures. Observed that most rewards were 0, with occasional rewards of 3.
  - Counted unique actions per state to identify any missing transitions or states with limited options.
- **Result**: The reward structure indicates sparse positive rewards, suggesting that optimal policies must identify these rare high-reward transitions.

### Step 3: Value Iteration for Policy Optimization
- **Objective**: Implement value iteration to determine the optimal state values, then derive the best policy based on these values.
- **Parameters**:
  - **Discount Factor (gamma)**: 0.95
  - **Threshold (theta)**: 1e-6 for convergence
- **Actions Taken**:
  - Initialized state values to zero.
  - Iteratively updated state values by considering the expected rewards for each action in each state, adjusting values based on the maximum reward possible.
  - Used the resulting value table to determine the optimal action per state.
- **Result**: An optimized value table and a derived policy for each state were produced.

### Step 4: Extracting and Saving the Policy
- **Objective**: Save the derived policy to a `.policy` file in the correct format for leaderboard submission.
- **Actions Taken**:
  - Formatted the policy to have one action per state and saved it as `small.policy`.
  - Verified the file contains exactly 100 lines using command-line tools to confirm compliance with project requirements.
- **Result**: Successfully saved and verified `small.policy` file for submission.

### Step 5: Initial Leaderboard Submission and Results
- **Objective**: Submit `small.policy` to the leaderboard to evaluate its performance.
- **Actions Taken**:
  - Uploaded `small.policy` to the leaderboard on Gradescope.
  - Observed feedback: The policy performed worse than random, yielding a negative leaderboard score.
  - **Key Insights**: Policy needs refinement as it did not surpass the baseline random policy.

### Step 6: Testing Different Gamma Values
- **Objective**: Test various discount factors (gamma) to evaluate their impact on policy performance.
- **Actions Taken**:
  - Ran value iteration with different gamma values (0.85, 0.9, and 0.99).
  - Generated and saved policies for each gamma value in files named `small_gamma_0.85.policy`, `small_gamma_0.9.policy`, and `small_gamma_0.99.policy`.
  - Verified that each policy file contained exactly 100 lines before submission.
- **Result**: Files were prepared for leaderboard submission to assess the effectiveness of each gamma value. The gamma 0.99 policy showed some improvement, but further optimization is needed.

### Step 7: Implementing Q-Learning
- **Objective**: Implement Q-learning to develop a policy based on observed rewards and action values.
- **Parameters**:
  - **Learning Rate (alpha)**: Set to 0.1 to gradually adjust the Q-values.
  - **Discount Factor (gamma)**: Set to 0.95 for long-term reward consideration.
  - **Exploration Factor (epsilon)**: Initialized at 0.1 to encourage exploration.
  - **Episodes**: Configured to run multiple episodes to capture state-action transitions.
- **Actions Taken**:
  - Initialized Q-values for all state-action pairs.
  - Iterated through episodes to update Q-values based on observed rewards and the Bellman equation.
  - Derived the policy by selecting the action with the highest Q-value for each state.
  - Saved the Q-learning derived policy as `small_qlearning.policy`.
- **Result**: Policy created with Q-learning completed and submitted to leaderboard, showing improvement but still requiring refinement.

### Step 8: Policy Gradient Optimization (PPO-Inspired Adjustments)
- **Objective**: Integrate policy gradient techniques from the PPO approach to further stabilize and optimize policy performance.
- **Adjustments Made**:
  - Increased **number of episodes** to 10,000, allowing the Q-learning algorithm to iterate further and capture more state-action transitions.
  - Implemented **clipping** to reduce the impact of excessively high updates, stabilizing Q-value adjustments across episodes.
  - Increased **maximum steps per episode**, enabling each episode to handle more transitions and provide a more comprehensive learning experience.
- **Runtime**: Total runtime for the updated Q-learning algorithm with these adjustments was approximately **2 hours 42 minutes and 42 seconds**.
- **Current Status**: The optimized policy was successfully saved as `small_qlearning.policy` and achieved a full score on the leaderboard with a **raw policy score of 34.87** and **leaderboard score of 33.56**.

### Step 9: Medium Dataset Q-Learning Optimization
- **Objective**: Apply Q-learning with PPO-inspired adjustments to `medium.csv`.
- **Adjustments Made**:
  - **Alpha (Learning Rate)**: Set to 0.1 for steady Q-value updates.
  - **Gamma (Discount Factor)**: Set to 0.99 to prioritize future rewards due to the larger state space.
  - **Epsilon (Exploration Rate)**: Started at 0.1, with **epsilon decay** applied over episodes to balance exploration and exploitation.
  - **Epsilon Min**: Minimum exploration threshold set at 0.01, ensuring some level of exploration persists throughout training.
  - Increased **number of episodes** to 30,000 to ensure thorough learning across the more complex state-action space.
  - Set the **maximum steps per episode** to handle extended transitions within the medium dataset.
- **Runtime**: Approximately **2 minutes and 10 seconds**.
- **Result**: The medium dataset policy (`medium_qlearning.policy`) scored successfully on the leaderboard with a **raw policy score** of **-0.16** and a **leaderboard score** of **103.85**, achieving 2/2 points.

### Step 10: Large Dataset Q-Learning Optimization
- **Objective**: Apply Q-learning with PPO-inspired adjustments to `large.csv` to handle the extensive state and action space.
- **Adjustments Made**:
  - **Alpha (Learning Rate)**: Set to 0.1 to facilitate steady updates.
  - **Gamma (Discount Factor)**: Set to 0.99, balancing immediate and future rewards effectively.
  - **Epsilon (Exploration Rate)**: Set to decay gradually, beginning at 0.1 with a minimum of 0.01.
  - **Number of Episodes**: Increased significantly to 80,000 to allow extensive learning given the complexity of the dataset.
  - **Maximum Steps per Episode**: Increased to handle numerous state transitions efficiently.
- **Runtime**: The large dataset Q-learning run took **12 hours, 14 minutes, and 40 seconds** to complete.
- **Result**: The policy (`large_qlearning.policy`) was successfully generated and scored 3/3 on the leaderboard with a **raw policy score of 303.48** and **leaderboard score of 3033.73**.

---

- **Overleaf Conversion**: Prepare this documentation on Overleaf if required by course specifications.
