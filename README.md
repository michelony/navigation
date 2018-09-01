# Udacity - Deep Reinforcement Learning - Navigation

## Project Details

This project is the navigation challenge, where a deep RL agent is trained to successfully navigate a strange world
filled with yellow and blue bananas.  Yellow bananas give a reward of +1 and blue bananas give a reward of -1.

The state space has 37 dimensions and contains the agent's velocity along with perception of the objects around the
agent's forward direction.  There are four actions an agent can choose at each time step: walk forward, walk backward,
turn left and turn right.

## Getting Started

In order to run the code, you'll need to download and install [PyTorch](https://pytorch.org/) and [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), as well as [Banana.app](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip).  The Banana.app link is for Mac OS.

## Instructions

I ran the code on the commandline in the Mac OS terminal:
```
$ python3 ./navigation.py
Episode 100	Average Score: 1.11
Episode 200	Average Score: 5.26
Episode 300	Average Score: 8.13
Episode 400	Average Score: 11.25
Episode 449	Average Score: 13.08
Environment solved in 349 episodes!	Average Score: 13.08
```

A file checkpoint.pth will be output with the final model weights.
