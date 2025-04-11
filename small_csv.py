import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Load and clean the dataset
file_path = '/Users/macbookpro/Desktop/Project/small.csv'  
data = pd.read_csv(file_path, header=None, names=['state', 'action', 'reward', 'next_state'])

# Clean the data
data = data[pd.to_numeric(data['state'], errors='coerce').notna()]
data = data[pd.to_numeric(data['action'], errors='coerce').notna()]
data['state'] = data['state'].astype(int)
data['action'] = data['action'].astype(int)
data['reward'] = data['reward'].astype(float)
data['next_state'] = data['next_state'].astype(int)

# Q-learning parameters
alpha = 0.1         # Initial learning rate
gamma = 0.99        # Discount factor
epsilon = 1.0       # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Decay rate for epsilon
decay_rate_alpha = 0.999  # Decay rate for alpha
num_episodes = 10000  # Number of episodes

# Initialize Q-table
states = data['state'].unique()
actions = data['action'].unique()
Q = {(state, action): 0.0 for state in states for action in actions}

print("Q-learning initialized. Starting episodes...")

# Q-learning algorithm with epsilon decay and adaptive learning rate
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    # Start from a random initial state
    current_state = random.choice(states)
    
    steps = 0  # Step counter for debugging
    while steps < 1000:  # Limit to prevent infinite loops
        steps += 1
        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            current_action = random.choice(actions)  # Explore
        else:
            current_action = max(actions, key=lambda a: Q[(current_state, a)])  # Exploit
        
        # Get the transition data for (state, action)
        transitions = data[(data['state'] == current_state) & (data['action'] == current_action)]
        
        if transitions.empty:
            print(f"No transitions available from state {current_state} with action {current_action}")
            break  # End if no transitions for this state-action pair
        
        # Select a random transition if multiple next states are possible
        next_transition = transitions.sample().iloc[0]
        reward = next_transition['reward']
        next_state = next_transition['next_state']
        
        # Q-learning update rule
        max_next_q = max(Q[(next_state, a)] for a in actions)
        Q[(current_state, current_action)] += alpha * (reward + gamma * max_next_q - Q[(current_state, current_action)])
        
        # Move to next state
        current_state = next_state

        # End episode if current_state not in states (no valid next states)
        if current_state not in states:
            break

    # Decay epsilon and alpha after each episode
    epsilon = max(min_epsilon, epsilon * decay_rate)
    alpha *= decay_rate_alpha

print("Q-learning completed. Extracting policy...") 

# Extract policy from Q-table
policy = {state: max(actions, key=lambda a: Q[(state, a)]) for state in states}

# Save policy to file
policy_file = '/Users/macbookpro/Desktop/Project/small_qlearning_improved.policy'
with open(policy_file, 'w') as file:
    for state in tqdm(range(1, 101), desc="Saving Policy"):
        action = policy.get(state, 1)  # Default to action 1 if state not in policy 
        file.write(f"{action}\n")

print(f"Q-learning policy saved to {policy_file}") 
