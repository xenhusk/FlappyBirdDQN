import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from envs.flappy_bird_env import FlappyBirdEnv
from agents.dqn_agent import DQNAgent

def evaluate_flappy(model_path, episodes=5, device='cpu', render=True, sleep_time=0.1):
    env = FlappyBirdEnv()
    agent = DQNAgent(4, 2, device=device)
    agent.load(model_path)
    
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
                time.sleep(sleep_time)
            
            a = agent.act(s, epsilon=0.0)
            s, r, done, info = env.step(a)
            total_reward += r
            
        print(f"Episode {ep+1} score {info['score']} return {total_reward:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/dqn_flappy.pth")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    evaluate_flappy(args.model, episodes=args.episodes, device=args.device)
