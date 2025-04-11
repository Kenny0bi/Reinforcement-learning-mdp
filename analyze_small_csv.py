import numpy as np
import pandas as pd

# Load and clean the dataset
file_path = '/Users/macbookpro/Desktop/Project/small.csv'  
data = pd.read_csv(file_path, header=None, names=['state', 'action', 'reward', 'next_state'])

# Drop rows with non-numeric entries in 'state' or 'action' columns
data = data[pd.to_numeric(data['state'], errors='coerce').notna()]
data = data[pd.to_numeric(data['action'], errors='coerce').notna()]

# Convert columns to appropriate data types
data['state'] = data['state'].astype(int)
data['action'] = data['action'].astype(int)
data['reward'] = data['reward'].astype(float)
data['next_state'] = data['next_state'].astype(int)

# Function to perform Value Iteration with a specified discount factor
def value_iteration(gamma):
    theta = 1e-6
    states = data['state'].unique()
    actions = data['action'].unique()
    value_table = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            state_value = value_table[state]
            action_values = []
            
            for action in actions:
                transitions = data[(data['state'] == state) & (data['action'] == action)]
                if not transitions.empty:
                    expected_value = (transitions['reward'] + gamma * transitions['next_state'].map(value_table)).mean()
                    action_values.append(expected_value)
            
            if action_values:
                value_table[state] = max(action_values)
                delta = max(delta, abs(state_value - value_table[state]))
        
        if delta < theta:
            break

    # Extract Policy safely
    policy = {}
    for state in states:
        next_state_values = [value_table.get(ns, 0) for ns in data[data['state'] == state]['next_state']]
        if next_state_values:
            best_action_index = min(np.argmax(next_state_values), len(actions) - 1)  # Ensure within bounds
            policy[state] = actions[best_action_index]
        else:
            policy[state] = 1  # Default to action 1 if no next states are available

    return value_table, policy

# Test different gamma values
gamma_values = [0.85, 0.90, 0.99]
results = {}

for gamma in gamma_values:
    value_table, policy = value_iteration(gamma)
    policy_file = f'/Users/macbookpro/Desktop/Project/small_gamma_{gamma}.policy'
    with open(policy_file, 'w') as file:
        for state in range(1, 101):
            action = policy.get(state, 1)  # Default to action 1 if state not in policy
            file.write(f"{action}\n")
    results[gamma] = policy_file
    print(f"Policy saved to {policy_file} for gamma = {gamma}")

print("Testing policies with different gamma values completed. Check results on the leaderboard.")
