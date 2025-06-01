import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 15000
NUM_EPISODES = 1000
WARMUP = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_state(state):
    pos, vel = state
    return [pos, vel]  # remove normalization — let the network learn naturally


# DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# Action selection epsilon greedy strategy
def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state).argmax().item()

# Environment setup
env = gym.make("MountainCar-v0", render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)
steps_done = 0
episode_rewards = []

for episode in range(NUM_EPISODES):
    raw_state, _ = env.reset()
    state = normalize_state(raw_state)
    total_reward = 0
    done = False

    while not done:
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        action = select_action(state, policy_net, epsilon, action_dim)

        raw_next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = normalize_state(raw_next_state)
        done = terminated or truncated

        # Reward shaping
        position = raw_next_state[0]
        reward += abs(position + 0.5)  # small incentive to move right
        if terminated:
            reward += 10 # bonus for reaching goal

        memory.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward
        steps_done += 1

        # Train
        if len(memory) >= max(BATCH_SIZE, WARMUP):
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(total_reward)

    # Update target net
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")


# Save model
torch.save(policy_net.state_dict(), "dqn_mountaincar6.pth")

# Plot reward curve
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN on MountainCar-v0")
plt.grid()
plt.savefig("training_rewards20.png")

env.close()

import gymnasium as gym
import torch
import numpy as np

# Assuming normalize_state, policy_net, and device are already defined

env = gym.make("MountainCar-v0", render_mode=None)
NUM_TEST_EPISODES = 100
test_rewards = []

for ep in range(NUM_TEST_EPISODES):
    raw_state, _ = env.reset()
    state = normalize_state(raw_state)
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()

        raw_next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = normalize_state(raw_next_state)
        total_reward += reward

    test_rewards.append(total_reward)

env.close()

# ✅ Print results
average_reward = np.mean(test_rewards)
print(f"\n✅ Average Reward over {NUM_TEST_EPISODES} Test Episodes: {average_reward:.2f}") 