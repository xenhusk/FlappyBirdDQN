import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s2, done):
        self.states[self.ptr] = s
        self.next_states[self.ptr] = s2
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        # Pre-process numpy arrays for faster transfer to GPU
        return dict(
            s=np.ascontiguousarray(self.states[idx]),
            a=np.ascontiguousarray(self.actions[idx]),
            r=np.ascontiguousarray(self.rewards[idx]),
            s2=self.next_states[idx],
            d=self.dones[idx],
        )

    def __len__(self):
        return self.size
