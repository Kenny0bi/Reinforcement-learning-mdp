import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Load and clean the dataset
file_path = '/Users/macbookpro/Desktop/Project/medium.csv'
data = pd.read_csv(file_path, header=None, names=['state', 'action', 'reward', 'next_state'])

# Clean the data
data = data[pd.to_numeric(data['state'], errors='coerce').notna()]
data = data[pd.to_numeric(data['action'], errors='coerce').notna()]
data['state'] = data['state'].astype(int)
data['action'] = data['action'].astype(int)
data['reward'] = data['reward'].astype(float)
data['next_state'] = data['next_state'].astype(int)

# Q-learning parameters
alpha = 0.1            # Learning rate
gamma = 0.99           # Discount factor
epsilon = 1.0          # Starting exploration rate
min_epsilon = 0.01     # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate
num_episodes = 30000   # Increased episode count
max_steps = 500        # Steps per episode

# Initialize Q-table
states = data['state'].unique()
actions = data['action'].unique()
Q = {(state, action): 0 for state in states for action in actions}

# Q-learning with epsilon decay
print("Q-learning with epsilon decay initialized. Starting episodes...") 

for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    # Start from a random initial state
    current_state = random.choice(states)
    
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            current_action = random.choice(actions)  # Explore
        else:
            # Exploit best-known action for current state
            current_action = max(actions, key=lambda a: Q.get((current_state, a), 0))
        
        # Retrieve transitions for (state, action)
        transitions = data[(data['state'] == current_state) & (data['action'] == current_action)]
        if transitions.empty:
            break  # No further transitions from this state-action pair
        
        # Sample a transition
        next_transition = transitions.sample().iloc[0]
        reward = next_transition['reward']
        next_state = int(next_transition['next_state'])

        # Ensure next state-action pairs exist in Q-table
        for a in actions:
            if (next_state, a) not in Q:
                Q[(next_state, a)] = 0  # Initialize unseen state-action pairs with 0

        # Q-learning update
        max_next_q = max(Q[(next_state, a)] for a in actions)
        Q[(current_state, current_action)] += alpha * (reward + gamma * max_next_q - Q[(current_state, current_action)])
        
        # Move to the next state
        current_state = next_state

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Extract policy from Q-table
policy = {state: max(actions, key=lambda a: Q.get((state, a), 0)) for state in states}

# Save policy to file
policy_file = '/Users/macbookpro/Desktop/Project/medium_qlearning.policy'
with open(policy_file, 'w') as file:
    for state in tqdm(range(1, 50001)):  # Medium dataset has 50,000 states
        action = policy.get(state, 1)  # Default to action 1 if state not in policy
        file.write(f"{action}\n")
print(f"Q-learning policy saved to {policy_file}")
