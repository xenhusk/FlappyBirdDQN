# Flappy Bird DQN

A reinforcement learning project implementing DQN (Deep Q-Network) to play a simplified version of Flappy Bird.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Training

To start training the agent:

```bash
python3 src/train.py --episodes 1000
```

The best model will be saved to `checkpoints/dqn_flappy.pth`.

## Evaluation

To watch the trained agent play:

```bash
python3 src/evaluate.py --model checkpoints/dqn_flappy.pth --episodes 3
```

## Environment

The Flappy Bird environment is simplified with the following properties:

- State space: [bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y]
- Action space: [0: do nothing, 1: flap]
- Rewards: 
  - +0.1 for surviving each step
  - +1.0 for passing through pipes
  - -1.0 for collision/death

## Architecture

- DQN with 2 hidden layers (128 units each)
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration
