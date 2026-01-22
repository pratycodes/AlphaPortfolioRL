import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, prev_weights, action, reward, next_state, next_prev_weights, greedy_action):
        self.buffer.append((state, prev_weights, action, reward, next_state, next_prev_weights, greedy_action))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, prev_w, action, reward, next_state, next_prev_w, greedy_action = zip(*batch)
        
        return (
            np.array(state),
            np.array(prev_w),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(next_prev_w),
            np.array(greedy_action)
        )

    def __len__(self):
        return len(self.buffer)