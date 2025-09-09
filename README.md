# Flappy Bird DQN 🐦

A high-performance reinforcement learning project implementing Deep Q-Network (DQN) to play Flappy Bird. This project features advanced optimizations including curriculum learning, GPU acceleration, and sophisticated reward engineering.

## 🎯 Performance

The trained agent achieves impressive results:
- **Best Score**: 144 points
- **Average Score**: 64.0 ± 58.17 points
- **Success Rate**: 80% (4/5 games score points)
- **Training Best**: 17 points during training

## 🚀 Features

### Advanced DQN Implementation
- **Deep Q-Network** with 6-layer architecture (512 hidden units)
- **Double Q-Learning** for improved stability
- **Experience Replay** with 200k capacity buffer
- **Target Network** with periodic updates
- **Gradient Clipping** to prevent exploding gradients

### Curriculum Learning
- **Easy Phase** (Episodes 1-500): Large gaps (200px), centered positions
- **Medium Phase** (Episodes 501-1500): Normal gaps (150px)
- **Hard Phase** (Episodes 1501+): Small gaps (120px), varied positions

### GPU Optimization
- **CUDA Support** with automatic GPU detection
- **Memory Optimization** for efficient GPU utilization
- **Batch Processing** with optimized chunk sizes
- **cuDNN Acceleration** for faster training

### Enhanced Environment
- **8-Dimensional State Space**: Bird position, velocity, pipe distances, gap boundaries
- **Simplified Reward Function**: Clear positive/negative signals
- **Dynamic Difficulty**: Curriculum-based pipe generation
- **Real-time Visualization**: Human-readable game rendering

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/xenhusk/FlappyBirdDQN.git
cd FlappyBirdDQN
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start - Play the Game

Want to play Flappy Bird yourself? Simply run:
```bash
python flappy_bird_game.py
```

**Game Controls:**
- **SPACE**: Flap (jump) / Start game / Restart
- **ESC**: Quit game

### Training

Train the AI agent with optimized parameters:
```bash
cd src
python train.py --mode train --episodes 5000 --render-mode none
```

**Training Parameters:**
- `--episodes`: Number of training episodes (default: 5000)
- `--render-mode`: `none` (fast), `fast` (occasional), `human` (full visualization)

### Human Play

Play the game yourself:
```bash
python flappy_bird_game.py
```

**Controls:**
- **SPACE**: Flap (jump)
- **ESC**: Quit game
- **SPACE**: Start new game (on start/game over screen)

### AI Evaluation

Watch the trained agent play:
```bash
python train.py --mode eval --model-path checkpoints/dqn_flappy_improved_best_score_17.pth --eval-episodes 5 --render-mode human
```

**Evaluation Options:**
- `--model-path`: Path to trained model
- `--eval-episodes`: Number of evaluation episodes
- `--render-mode`: Visualization mode

### List Available Models

View all trained models:
```bash
python train.py --mode list
```

## 🏗️ Architecture

### Neural Network
```
Input (8) → Hidden (512) → Hidden (512) → Hidden (512) → Hidden (256) → Hidden (128) → Output (2)
```
- **Input**: 8-dimensional state vector
- **Hidden Layers**: 5 layers with ReLU activation and dropout
- **Output**: Q-values for 2 actions (no flap, flap)

### Environment Details
- **State Space**: [bird_y, bird_velocity, pipe_distance, gap_center, gap_top, gap_bottom, relative_position, pipe_center_distance]
- **Action Space**: [0: do nothing, 1: flap]
- **Rewards**:
  - +10.0 for passing through pipes
  - +0.1 for surviving each step
  - -1.0 + survival_bonus for collision/death
  - +0.5 for good positioning near pipes

### Training Configuration
- **Batch Size**: 256
- **Learning Rate**: 5e-4
- **Gamma**: 0.99
- **Epsilon Decay**: 0.995
- **Target Update**: Every 100 steps
- **Buffer Size**: 200,000 experiences

## 📊 Results

### Training Progress
The agent successfully learns through curriculum phases:
1. **Easy Phase**: Learns basic navigation with large gaps
2. **Medium Phase**: Adapts to normal difficulty
3. **Hard Phase**: Masters challenging small gaps

### Performance Metrics
- **Training Best Score**: 17 points
- **Evaluation Best Score**: 144 points
- **Average Evaluation Score**: 64.0 points
- **Success Rate**: 80%
- **Longest Survival**: 9,743 steps

## 🔧 Technical Details

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pygame
- Matplotlib

### GPU Requirements
- NVIDIA GPU with CUDA support (recommended)
- 4GB+ VRAM for optimal performance
- Automatic fallback to CPU if GPU unavailable

### File Structure
```
FlappyBirdDQN/
├── src/
│   ├── agents/
│   │   └── dqn_agent.py          # DQN agent implementation
│   ├── envs/
│   │   └── flappy_bird_env.py    # Game environment with curriculum learning
│   ├── models/
│   │   └── q_network.py          # Neural network architecture
│   ├── train.py                  # Main training script
│   └── replay_buffer.py          # Experience replay buffer
├── checkpoints/                  # Trained models
├── flappy_bird_game.py          # Base game implementation
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🎯 Key Optimizations

1. **Reward Engineering**: Simplified, effective reward structure
2. **Curriculum Learning**: Gradual difficulty progression
3. **GPU Acceleration**: Optimized for NVIDIA GPUs
4. **Network Architecture**: Deep network with proper regularization
5. **Hyperparameter Tuning**: Carefully optimized for Flappy Bird
6. **Experience Replay**: Large buffer with efficient sampling

## 📈 Future Improvements

- [ ] Prioritized Experience Replay
- [ ] Dueling DQN architecture
- [ ] Multi-step learning
- [ ] Advanced curriculum strategies
- [ ] Real-time performance monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project demonstrates advanced reinforcement learning techniques and serves as an excellent example of DQN implementation with modern optimizations.