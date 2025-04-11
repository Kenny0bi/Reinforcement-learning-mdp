import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Load and clean the dataset
file_path = '/Users/macbookpro/Desktop/Project/large.csv'  
data = pd.read_csv(file_path, header=None, names=['state', 'action', 'reward', 'next_state'])

# Clean the data
data = data[pd.to_numeric(data['state'], errors='coerce').notna()]
data = data[pd.to_numeric(data['action'], errors='coerce').notna()]
data['state'] = data['state'].astype(int)
data['action'] = data['action'].astype(int)
data['reward'] = data['reward'].astype(float)
data['next_state'] = data['next_state'].astype(int)

# Q-learning parameters for large dataset
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 0.1       # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.9999  # Decay rate for epsilon
num_episodes = 80000  # Number of episodes
max_steps = 500       # Max steps per episode

# Initialize Q-table
states = data['state'].unique()
actions = data['action'].unique()
Q = {(state, action): 0 for state in states for action in actions}
print("Q-learning for large dataset initialized. Starting episodes...")

# Q-learning with epsilon decay
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    current_state = random.choice(states)
    
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            current_action = random.choice(actions)
        else:
            current_action = max(actions, key=lambda a: Q[(current_state, a)])
        
        # Transition based on (state, action)
        transitions = data[(data['state'] == current_state) & (data['action'] == current_action)]
        
        if transitions.empty:
            break  # No valid transition, end episode
        
        # Randomly sample a transition
        next_transition = transitions.sample().iloc[0]
        reward = next_transition['reward']
        next_state = next_transition['next_state']
        
        # Update Q-table
        max_next_q = max(Q[(next_state, a)] for a in actions)
        Q[(current_state, current_action)] += alpha * (reward + gamma * max_next_q - Q[(current_state, current_action)])
        
        # Move to the next state
        current_state = next_state
        
        # End if current state is terminal or not in states
        if current_state not in states:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Extract policy
policy = {state: max(actions, key=lambda a: Q[(state, a)]) for state in states}

# Save policy to file
policy_file = '/Users/macbookpro/Desktop/Project/large_qlearning.policy'
with open(policy_file, 'w') as file:
    for state in tqdm(range(1, 302021), desc="Saving Policy"):
        action = policy.get(state, 1)  # Default to action 1 if state not in policy
        file.write(f"{action}\n")
print(f"Q-learning policy for large dataset saved to {policy_file}")
