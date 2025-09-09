import random
import numpy as np
import torch
import torch.nn.functional as F
from models.q_network import QNetwork

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device='cpu', lr=5e-4, gamma=0.99, tau=0.005):
        self.device = torch.device(device)
        
        # Configure GPU memory usage and optimization
        if self.device.type == 'cuda':
            # Clear cache and set memory fraction for dedicated GPU
            torch.cuda.empty_cache()
            # Enable cuDNN autotuner and benchmark mode
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Better performance
            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        
        # Initialize networks with memory-efficient settings
        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions).to(self.device)
        
        # Enable double Q-learning
        self.double_q = True
        
        # Enable gradient clipping
        self.clip_grad = 1.0
        
        # Copy weights
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau  # if tau==1.0 -> hard update
        self.n_actions = n_actions
        
        # Print model device placement
        print(f"Model device: {next(self.q.parameters()).device}")
        if self.device.type == 'cuda':
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
        return int(torch.argmax(qvals, dim=1).item())

    def update(self, batch, batch_size):
        """Update Q network using sampled mini-batch with double Q-learning"""
        # Move data to GPU efficiently using pin_memory for faster transfers
        s = torch.from_numpy(np.array(batch['s'])).float().to(self.device)
        a = torch.from_numpy(np.array(batch['a'])).long().unsqueeze(1).to(self.device)
        r = torch.from_numpy(np.array(batch['r'])).float().unsqueeze(1).to(self.device)
        s2 = torch.from_numpy(np.array(batch['s2'])).float().to(self.device)
        d = torch.from_numpy(np.array(batch['d'])).float().unsqueeze(1).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        
        # Process in chunks if batch is large
        chunk_size = 512  # Optimized chunk size for better GPU utilization
        total_loss = 0
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        
        # Pre-allocate tensors on GPU for better memory efficiency
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean').to(self.device)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            
            chunk_s = s[start_idx:end_idx]
            chunk_a = a[start_idx:end_idx]
            chunk_r = r[start_idx:end_idx]
            chunk_s2 = s2[start_idx:end_idx]
            chunk_d = d[start_idx:end_idx]
            
            # Get current Q values
            q_vals = self.q(chunk_s).gather(1, chunk_a)
            
            # Compute target Q values using double Q-learning
            with torch.no_grad():
                if self.double_q:
                    # Get actions from online network
                    next_actions = self.q(chunk_s2).max(1)[1].unsqueeze(1)
                    # Get Q-values from target network for those actions
                    q_next = self.q_target(chunk_s2).gather(1, next_actions)
                else:
                    q_next = self.q_target(chunk_s2).max(1)[0].unsqueeze(1)
                    
                q_target = chunk_r + self.gamma * q_next * (1.0 - chunk_d)
            
            # Compute loss with pre-allocated loss function
            loss = self.loss_fn(q_vals, q_target)
            total_loss += loss.item()
            # Scale loss for better numerical stability with larger batches
            (loss / num_chunks).backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.clip_grad)
        self.optimizer.step()
        
        return total_loss / num_chunks

    def soft_update(self):
        for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def hard_update(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.hard_update()
