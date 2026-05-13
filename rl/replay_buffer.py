# rl/replay_buffer.py
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional
import random

@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[dict] = None

class PrioritizedReplayBuffer:
    """بافر بازپخش با اولویت"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def push(self, transition: Transition, priority: float = None):
        """افزودن transition جدید"""
        if priority is None:
            priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """نمونه‌گیری از بافر"""
        if self.size == 0:
            return None
        
        # محاسبه احتمال‌ها
        probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])
        
        # نمونه‌گیری
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # محاسبه وزن‌ها
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # افزایش beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # جمع‌آوری نمونه‌ها
        batch = [self.buffer[idx] for idx in indices]
        
        # تبدیل به تانسور
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.FloatTensor(np.array([t.action for t in batch]))
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        return (states, actions, rewards, next_states, dones, weights, indices)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """به‌روزرسانی اولویت‌ها"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha
    
    def __len__(self):
        return self.size

class ReplayBuffer:
    """بافر بازپخش استاندارد"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition: Transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.FloatTensor(np.array([t.action for t in batch]))
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(1)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)