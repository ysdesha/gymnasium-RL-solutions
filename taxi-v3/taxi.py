import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

alpha = 0.9 #learning rate
gamma = 0.95
epsilon = 1  #exploring rate
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

# 5x5 -> 25 positions * 5 * 4 
q_table = np.zeros((env.observation_space.n, env.action_space.n))

#function for agent to choose action (using epsilon greedy strategy)
def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()       # Explore (random action)
    else:
        return np.argmax(q_table[state,:])     # Exploit (best known action)
    

rewards_per_episode = []

#training
for episode in range(num_episodes):
    state, info = env.reset() 
    terminated = False
    truncated = False
    total_rewards = 0

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        total_rewards += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_rewards)
  
plt.figure(figsize=(10,5))
plt.plot(rewards_per_episode, color='orange')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode (Taxi-v3 Q-learning)')
plt.grid(True)

plt.ylim(-100, 25)  # 👈 Set your desired y-axis limits here
plt.show()


#seeing result
env = gym.make('Taxi-v3',render_mode = 'human')

for episode in range(5):
    state, info = env.reset()
    terminated = False  # added for clarity
    truncated = False
   
    print('Episode', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state,:])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            env.render()
            print('Finished episode' , episode, 'with reward', reward)
            break

env.close()        

