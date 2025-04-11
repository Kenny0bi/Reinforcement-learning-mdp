# Reinforcement Learning for MDPs using Q-Learning

This project implements batch reinforcement learning to solve Markov Decision Processes (MDPs) using Q-learning. It works across three datasets of increasing complexity — small, medium, and large — with the goal of learning optimal policies to maximize total expected rewards.

## 📂 Datasets
- `small.csv`: Grid world (100 states, 4 actions)
- `medium.csv`: MountainCar-style (50,000 states, 7 actions)
- `large.csv`: Simulated environment (302,020 states, 9 actions)

## 🧠 Methods
We used Q-learning with:
- Epsilon-greedy action selection
- Learning rate decay
- Discount factor tuning
- Adaptive episodes per dataset complexity

Policies were derived and saved in `.policy` format for leaderboard evaluation.

## 🔧 Requirements
- Python 3.x
- `pandas`
- `numpy`
- `tqdm`

Install dependencies:
```bash
pip install pandas numpy tqdm

Running the Scripts
Each file trains a Q-learning agent on a different dataset:
    python small_csv.py
    python medium_csv.py
    python large_csv.py

Policy files will be saved in the working directory.

📝 Output
small_qlearning_improved.policy

medium_qlearning.policy

large_qlearning.policy

Each file contains the optimal action for every state (one action per line).

📈 Leaderboard Results
Small Dataset: Raw score 34.87 | Leaderboard 33.56 ✅

Medium Dataset: Raw score -0.16 | Leaderboard 103.85 ✅

Large Dataset: Raw score 303.48 | Leaderboard 3033.73 ✅

🙌 Author
Kehinde Obidele
Health Informatics | Reinforcement Learning | Decision Support
