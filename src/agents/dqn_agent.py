import random
import numpy as np
import torch
import torch.nn.functional as F
from models.q_network import QNetwork

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device='cpu', lr=5e-3, gamma=0.99, tau=1.0):
        self.device = torch.device(device)
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # Initialize networks
        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions).to(self.device)
        
        # Ensure the networks are in training mode
        self.q.train()
        self.q_target.train()
        
        # Copy weights and move optimizer to GPU
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau  # if tau==1.0 -> hard update
        self.n_actions = n_actions
        
        # Configure mixed precision training
        self.scaler = torch.amp.GradScaler()
        
        # Print model device placement
        print(f"Model device: {next(self.q.parameters()).device}")

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32).to(self.device, non_blocking=True).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                qvals = self.q(s)
        return int(torch.argmax(qvals, dim=1).item())

    def update(self, batch, batch_size):
        # Move data to GPU efficiently
        s = torch.as_tensor(batch['s'], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch['a'], dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.as_tensor(batch['r'], dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(batch['s2'], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch['d'], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.q_target(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1.0 - d)

        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            loss = F.mse_loss(q_vals, q_target)
        
        # Use mixed precision training
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)  # Add gradient clipping
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.step()
        return loss.item()

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
