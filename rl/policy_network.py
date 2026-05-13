# rl/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = 'relu'
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = False

class MLP(nn.Module):
    """شبکه عصبی چندلایه ساده"""
    def __init__(self, input_dim: int, output_dim: int, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        layers = []
        current_dim = input_dim
        
        # لایه‌های پنهان
        for i in range(config.num_layers):
            layers.append(nn.Linear(current_dim, config.hidden_size))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_size))
            
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            current_dim = config.hidden_size
        
        # لایه خروجی
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class GaussianPolicy(nn.Module):
    """سیاست گاوسی برای SAC"""
    def __init__(self, state_dim: int, action_dim: int, config: NetworkConfig):
        super().__init__()
        
        # شبکه برای میانگین
        self.mean_net = MLP(state_dim, action_dim, config)
        
        # شبکه برای انحراف معیار (log std)
        self.log_std_net = MLP(state_dim, action_dim, config)
        
        # محدود کردن log_std
        self.log_std_min = -20
        self.log_std_max = 2
        
        # برای ذخیره action و log_prob
        self.saved_action = None
        self.saved_log_prob = None
        
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """محاسبه توزیع action"""
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        
        # محدود کردن log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # ایجاد توزیع نرمال
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # reparameterization trick
        
        # محاسبه log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # محدود کردن action بین 0 و 1
        action = torch.sigmoid(action)
        
        return action, log_prob
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """نمونه‌گیری از سیاست"""
        action, log_prob = self.forward(state)
        self.saved_action = action
        self.saved_log_prob = log_prob
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ارزیابی action داده‌شده"""
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        # محاسبه log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # محاسبه entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy

class QNetwork(nn.Module):
    """شبکه Q برای SAC"""
    def __init__(self, state_dim: int, action_dim: int, config: NetworkConfig):
        super().__init__()
        self.q_net = MLP(state_dim + action_dim, 1, config)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)

class DoubleQNetwork(nn.Module):
    """دو شبکه Q برای کاهش بیش‌برآورد"""
    def __init__(self, state_dim: int, action_dim: int, config: NetworkConfig):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, config)
        self.q2 = QNetwork(state_dim, action_dim, config)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """حداقل مقدار Q بین دو شبکه"""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class SACPolicy(nn.Module):
    """سیاست کامل SAC"""
    def __init__(self, state_dim: int, action_dim: int, config: Optional[NetworkConfig] = None):
        super().__init__()
        
        if config is None:
            config = NetworkConfig()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # شبکه‌ها
        self.policy = GaussianPolicy(state_dim, action_dim, config)
        self.q_network = DoubleQNetwork(state_dim, action_dim, config)
        self.target_q_network = DoubleQNetwork(state_dim, action_dim, config)
        
        # به‌روزرسانی شبکه هدف
        self._update_target_network(tau=1.0)
        
        # تنظیم اتوماتیک temperature
        self.auto_entropy_tuning = True
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            self.alpha = self.log_alpha.exp().item()  # مقدار اولیه
        else:
            self.alpha = 0.2
        
    def _update_target_network(self, tau: float = 0.005):
        """به‌روزرسانی شبکه هدف با وزن‌های ترکیبی"""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """انتخاب action برای state داده‌شده"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.policy(state_tensor, deterministic)
        
        return action.squeeze(0).numpy()
    
    def save(self, path: str):
        """ذخیره مدل"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        }, path)
    
    def load(self, path: str):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']