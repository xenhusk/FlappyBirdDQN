import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from envs.flappy_bird_env import FlappyBirdEnv
from agents.dqn_agent_fixed import DQNAgent
from replay_buffer import ReplayBuffer

# Print CUDA information
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

def train_flappy(
    episodes=1000,
    batch_size=256,  # Increased batch size for faster learning
    buffer_capacity=50000,
    start_train=500,  # Start training earlier
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.99,  # Faster epsilon decay
    target_update_freq=50,  # More frequent target updates
    save_path="checkpoints/dqn_flappy.pth",
    device='cpu'
):
    env = FlappyBirdEnv()
    obs_dim = 4  # [y, velocity, pipe_x, pipe_gap_y]
    n_actions = 2  # [do nothing, flap]
    
    agent = DQNAgent(obs_dim, n_actions, device=device)
    buf = ReplayBuffer(buffer_capacity, obs_dim)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_score = 0
    epsilon = eps_start
    total_steps = 0

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        done = False
        
        # Enable anomaly detection during training
        with torch.cuda.amp.autocast():
            while not done:
                # Move state to GPU for inference
                a = agent.act(s, epsilon)
                s2, r, done, info = env.step(a)
                buf.push(s, a, r, s2, float(done))
                s = s2
                ep_reward += r
                total_steps += 1

                if len(buf) >= start_train:
                    # Process multiple batches for better GPU utilization
                    for _ in range(4):  # Multiple updates per step
                        batch = buf.sample(batch_size)
                        loss = agent.update(batch, batch_size)
                        
                    # Force CUDA sync periodically
                    if total_steps % 100 == 0:
                        torch.cuda.synchronize()

            if total_steps % target_update_freq == 0:
                agent.hard_update()

        score = info.get('score', 0)
        if score > best_score:
            best_score = score
            agent.save(save_path)

        epsilon = max(eps_end, epsilon * eps_decay)
        if ep % 10 == 0:
            print(f"Episode {ep} score {score} reward {ep_reward:.2f} eps {epsilon:.3f} buf {len(buf)}")

    print(f"Training finished. Best score: {best_score}")
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train_flappy(episodes=args.episodes, device=args.device)
