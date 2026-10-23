import numpy as np
import random

# Number of states and actions
num_states = 5
num_actions = 2

# Actions
# 0 = Left
# 1 = Right

# Initialize Q-table
Q = np.zeros((num_states, num_actions))

# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.2     # Exploration probability
episodes = 1000

# Environment function
def get_next_state(state, action):

    if action == 0:       # Move Left
        next_state = max(0, state - 1)
    else:                 # Move Right
        next_state = min(num_states - 1, state + 1)

    # Reward
    if next_state == 4:
        reward = 10
    else:
        reward = -1

    return next_state, reward


# Q-Learning
for episode in range(episodes):

    state = 0

    while state != 4:

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(Q[state])

        # Perform action
        next_state, reward = get_next_state(state, action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward +
            gamma * np.max(Q[next_state]) -
            Q[state, action]
        )

        state = next_state


# Display Q-table
print("Learned Q-Table:")
print(Q)


# Test the learned policy
state = 0
path = [state]

while state != 4:

    action = np.argmax(Q[state])

    next_state, reward = get_next_state(state, action)

    state = next_state
    path.append(state)

print("\nLearned Path:")
print(path)

print("\nOptimal Policy:")
for state in range(num_states - 1):

    action = np.argmax(Q[state])

    if action == 0:
        print("State", state, "-> Left")
    else:
        print("State", state, "-> Right")
