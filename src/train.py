import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from envs.flappy_bird_env import FlappyBirdEnvNew, TrainingManager
from agents.dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# Print CUDA information and force dedicated GPU usage
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Force use of dedicated GPU (usually GPU 1 on laptops)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)  # Use dedicated GPU
        print(f"Using dedicated GPU: {torch.cuda.get_device_name(1)}")
        # Set memory allocation strategy for better performance
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

def train_flappy(
    episodes=5000,  # More episodes for better learning
    batch_size=256,  # Larger batches for better GPU utilization
    buffer_capacity=200000,  # Larger buffer for more diverse experiences
    start_train=2000,  # More warmup data
    eps_start=1.0,
    eps_end=0.01,  # Higher final epsilon for continued exploration
    eps_decay=0.995,  # Faster decay initially
    target_update_freq=100,  # Less frequent target updates
    save_freq=200,  # Less frequent saves
    save_path="checkpoints/dqn_flappy_improved.pth",
    config_path="checkpoints/training_config.json",
    device='cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda' if torch.cuda.is_available() else 'cpu',
    render_mode='none'  # 'none', 'fast', or 'human'
):
    # Create improved environment
    env = FlappyBirdEnvNew(training_mode=True, render_mode=render_mode)
    manager = TrainingManager(env)
    
    obs_dim = env.observation_space_dim  # Should be 8 now
    n_actions = env.action_space_dim
    
    print(f"Environment: {obs_dim} observations, {n_actions} actions")
    
    # Save training configuration
    config = {
        'obs_dim': obs_dim,
        'n_actions': n_actions,
        'episodes': episodes,
        'batch_size': batch_size,
        'buffer_capacity': buffer_capacity,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'eps_decay': eps_decay,
        'target_update_freq': target_update_freq,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize agent and replay buffer
    agent = DQNAgent(obs_dim, n_actions, device=device)
    buffer = ReplayBuffer(buffer_capacity, obs_dim)
    
    # Initialize plotting
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training metrics
    rewards_history = []
    scores_history = []
    losses_history = []
    epsilon_history = []
    avg_scores = []
    
    def update_plots():
        if len(rewards_history) == 0:
            return
            
        episodes_range = range(1, len(rewards_history) + 1)
        
        # Rewards plot
        ax1.clear()
        ax1.plot(episodes_range, rewards_history, alpha=0.7)
        if len(rewards_history) > 50:
            # Moving average
            window = 50
            moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(rewards_history) + 1), moving_avg, 'r-', linewidth=2, label='50-ep average')
            ax1.legend()
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Scores plot
        ax2.clear()
        ax2.plot(episodes_range, scores_history, alpha=0.7)
        if len(scores_history) > 50:
            window = 50
            moving_avg = np.convolve(scores_history, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(scores_history) + 1), moving_avg, 'r-', linewidth=2, label='50-ep average')
            ax2.legend()
        ax2.set_title('Game Scores')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        # Losses plot
        ax3.clear()
        if losses_history and any(l is not None for l in losses_history):
            valid_losses = [l for l in losses_history if l is not None]
            if valid_losses:
                ax3.plot(valid_losses, alpha=0.7)
                ax3.set_title('Training Loss')
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('Loss')
                ax3.grid(True, alpha=0.3)
        
        # Epsilon decay plot
        ax4.clear()
        ax4.plot(episodes_range, epsilon_history, 'g-', linewidth=2)
        ax4.set_title('Exploration Rate (Epsilon)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

    best_score = 0
    best_avg_score = 0
    epsilon = eps_start
    total_training_steps = 0
    
    print("Starting training...")
    print(f"Training for {episodes} episodes")
    print(f"Using device: {device}")
    
    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            ep_reward = 0
            done = False
            steps_in_episode = 0
            
            while not done:
                # Get action from agent
                action = agent.act(state, epsilon)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition in replay buffer
                buffer.push(state, action, reward, next_state, float(done))
                
                state = next_state
                ep_reward += reward
                steps_in_episode += 1
                
                # Train agent
                if len(buffer) >= start_train:
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch, batch_size)
                    losses_history.append(loss)
                    total_training_steps += 1
                    
                    # Update target network
                    if total_training_steps % target_update_freq == 0:
                        agent.hard_update()
                else:
                    losses_history.append(None)

            # Update statistics
            rewards_history.append(ep_reward)
            scores_history.append(info['score'])
            epsilon_history.append(epsilon)
            
            # Calculate moving average for recent performance
            if len(scores_history) >= 100:
                recent_avg = np.mean(scores_history[-100:])
                avg_scores.append(recent_avg)
            else:
                avg_scores.append(np.mean(scores_history))
            
            # Log episode with manager
            current_loss = losses_history[-1] if losses_history and losses_history[-1] is not None else None
            manager.log_episode(ep_reward, epsilon, current_loss)
            
            # Update plots every 10 episodes
            if ep % 10 == 0:
                update_plots()
            
            # Save best model based on score
            if info['score'] > best_score:
                best_score = info['score']
                best_model_path = save_path.replace('.pth', f'_best_score_{best_score}.pth')
                agent.save(best_model_path)
                print(f"🏆 New best score: {best_score} (Episode {ep})")
            
            # Save best model based on average performance
            current_avg = avg_scores[-1]
            if current_avg > best_avg_score and len(scores_history) >= 100:
                best_avg_score = current_avg
                best_avg_path = save_path.replace('.pth', f'_best_avg_{best_avg_score:.1f}.pth')
                agent.save(best_avg_path)
                print(f"📈 New best average: {best_avg_score:.2f} (Episode {ep})")
            
            # Regular saves
            if ep % save_freq == 0:
                checkpoint_path = save_path.replace('.pth', f'_ep_{ep}.pth')
                agent.save(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Decay epsilon
            epsilon = max(eps_end, epsilon * eps_decay)
            
            # Print progress
            if ep % 50 == 0 or info['score'] > 0:
                stats = env.get_training_stats()
                phase_names = ['Easy', 'Medium', 'Hard']
                current_phase = phase_names[env.curriculum_phase] if hasattr(env, 'curriculum_phase') else 'Normal'
                print(f"Episode {ep:4d}: Score={info['score']:2d}, "
                      f"Reward={ep_reward:6.2f}, Eps={epsilon:.3f}, "
                      f"Avg={current_avg:.2f}, Best={best_score}, "
                      f"Phase={current_phase}, Buffer={len(buffer)}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    finally:
        # Final save
        final_path = save_path.replace('.pth', '_final.pth')
        agent.save(final_path)
        
        # Save training history
        history = {
            'rewards': rewards_history,
            'scores': scores_history,
            'avg_scores': avg_scores,
            'epsilon': epsilon_history,
            'best_score': best_score,
            'best_avg_score': best_avg_score,
            'total_episodes': len(scores_history),
            'config': config
        }
        
        history_path = save_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        print(f"\n✅ Training completed!")
        print(f"📊 Final stats - Best Score: {best_score}, Best Avg: {best_avg_score:.2f}")
        print(f"💾 Model saved: {final_path}")
        print(f"📈 History saved: {history_path}")
        
        # Keep plots open
        plt.ioff()
        plt.show()
        
        env.close()
    
    return agent

def evaluate_model(model_path, num_episodes=10, render_mode='human'):
    """Enhanced evaluation with better error handling and compatibility checking"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Try to load config to determine model architecture
    config_path = model_path.replace('.pth', '_history.json').replace('_final', '').replace(f'_best_score_{model_path.split("_")[-1].replace(".pth", "")}', '').replace(f'_best_avg_{model_path.split("_")[-1].replace(".pth", "")}', '')
    
    obs_dim = 8  # Default to improved environment
    
    # Check if we have config info
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                history = json.load(f)
                if 'config' in history and 'obs_dim' in history['config']:
                    obs_dim = history['config']['obs_dim']
                    print(f"📋 Loaded config: {obs_dim} observations")
        except:
            print("⚠️ Could not load config, using default")
    
    # Create environment with appropriate observation space
    if obs_dim == 4:
        print("🔄 Using legacy environment (4 observations)")
        env = FlappyBirdEnvNew(training_mode=False, render_mode=render_mode)
    else:
        print("🚀 Using improved environment (8 observations)")
        env = FlappyBirdEnvNew(training_mode=False, render_mode=render_mode)
    
    n_actions = env.action_space_dim
    
    # Load agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(obs_dim, n_actions, device=device)
    
    try:
        agent.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"❌ Architecture mismatch: Model expects different input size")
            print(f"💡 Try using --obs-dim to specify the correct observation dimension")
            print(f"🔍 Error: {e}")
            return
        else:
            raise e
    
    scores = []
    rewards = []
    
    print(f"\n🎮 Running {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action from agent (no exploration)
            action = agent.act(state, epsilon=0)
            
            # Take action in environment
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Handle pygame events to prevent freezing
            if hasattr(env, 'handle_events'):
                if not env.handle_events():
                    print("🛑 Evaluation stopped by user")
                    env.close()
                    return
        
        scores.append(info['score'])
        rewards.append(total_reward)
        
        print(f"Episode {ep + 1:2d}: Score = {info['score']:2d}, "
              f"Reward = {total_reward:7.2f}, Steps = {steps:3d}")
    
    # Print summary statistics
    print(f"\n📊 Evaluation Results ({num_episodes} episodes):")
    print(f"   Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"   Best Score: {np.max(scores)}")
    print(f"   Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Success Rate: {sum(1 for s in scores if s > 0) / len(scores) * 100:.1f}%")
    
    env.close()

def list_available_models(checkpoint_dir="checkpoints"):
    """List all available model files"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    models = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            models.append(file)
    
    if not models:
        print(f"❌ No model files found in {checkpoint_dir}")
        return
    
    print(f"📁 Available models in {checkpoint_dir}:")
    for i, model in enumerate(sorted(models), 1):
        print(f"   {i}. {model}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate Flappy Bird DQN")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "list"])
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--model-path", type=str, default="checkpoints/dqn_flappy_improved.pth")
    parser.add_argument("--render-mode", type=str, default="human", choices=["none", "fast", "human"])
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_flappy(episodes=args.episodes, render_mode=args.render_mode)
    elif args.mode == "eval":
        evaluate_model(args.model_path, args.eval_episodes, args.render_mode)
    elif args.mode == "list":
        list_available_models()