#!/usr/bin/python3
""" navigation.py
Loads Banana.app, a simple environment for training and evaluating a deep RL
agent.  Uses UnityEnvironment.
"""

import numpy as np
import torch
from collections import deque
from dqn_agent import Agent
from unityagents import UnityEnvironment

# Start the environment
env = UnityEnvironment(file_name="Banana.app", no_graphics=True)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Examine the state and action space
# The simulation contains a single agent that navigates a large environment.
# At each time step, it has four actions at its disposal:
# 0 - walk forward
# 1 - walk backward
# 2 - turn left
# 3 - turn right
# The state space has 37 dimensions and contains the agent's velocity, along with
# ray-based perception of objects around agent's forward direction.  A reward of
# +1 is provided for collecting a yellow banana, and a reward of -1 is provided
# for collecting a blue banana.

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Deep Q-Learning
    """
    Params
    ======
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

    Expected Output:
      Episode 100   Average Score: 1.11
      Episode 200   Average Score: 5.26
      Episode 300   Average Score: 8.13
      Episode 400   Average Score: 11.25
      Episode 449   Average Score: 13.08
      Environment solved in 349 episodes!   Average Score: 13.08
    """
    scores = []                        # list containing scores from each episode\n",
    scores_window = deque(maxlen=100)  # last 100 scores\n",
    eps = eps_start                    # initialize epsilon\n",
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment\n",
        state = env_info.vector_observations[0]            # get the current state\n",
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)), end='')
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()
env.close()
