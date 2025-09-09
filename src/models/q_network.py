import torch.nn as nn
import torch

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=512):  # Larger network for better GPU utilization
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden // 4, n_actions)
        )

    def forward(self, x):
        return self.net(x)
