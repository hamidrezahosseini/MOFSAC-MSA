# rl/sac_trainer.py
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, Tuple
import wandb
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl.environment import EOEnvironment
from rl.policy_network import SACPolicy, NetworkConfig
from rl.replay_buffer import PrioritizedReplayBuffer, Transition
from config import config

class SACTrainer:
    """آموزش‌دهنده الگوریتم SAC"""
    
    def __init__(self, env: EOEnvironment, config_dict: Optional[Dict] = None):
        self.env = env
        self.config = config_dict if config_dict else config.SAC_CONFIG
        
        # پارامترها
        self.learning_rate = self.config['learning_rate']
        self.buffer_size = self.config['buffer_size']
        self.batch_size = self.config['batch_size']
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.alpha = self.config['alpha']
        self.auto_entropy_tuning = self.config['auto_entropy_tuning']
        
        # ابعاد state و action
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # تنظیمات شبکه
        network_config = NetworkConfig(
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers']
        )
        
        # ایجاد مدل SAC
        self.policy = SACPolicy(self.state_dim, self.action_dim, network_config)
        
        # بهینه‌سازها
        self.policy_optimizer = optim.Adam(
            self.policy.policy.parameters(), 
            lr=self.learning_rate
        )
        
        self.q_optimizer = optim.Adam(
            self.policy.q_network.parameters(), 
            lr=self.learning_rate
        )
        
        # بافر بازپخش
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            alpha=0.6,
            beta=0.4
        )
        
        # آمارهای آموزش
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -np.inf
        
        # مسیر ذخیره
        self.save_dir = config.TRAINING_CONFIG['model_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # log
        self.log_dir = config.TRAINING_CONFIG['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        
        # استفاده از GPU اگر موجود باشد
        self.device = torch.device(config.DEVICE)
        self.policy.to(self.device)
        
    def train_step(self) -> Dict[str, float]:
        """یک مرحله آموزش"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # نمونه‌گیری از بافر
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        # انتقال به device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # آموزش شبکه Q
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.policy(next_states)
            target_q1, target_q2 = self.policy.target_q_network(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.policy.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # محاسبه loss شبکه Q
        current_q1, current_q2 = self.policy.q_network(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # آموزش شبکه Q
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # آموزش سیاست
        new_actions, log_probs = self.policy.policy(states)
        q1_new, q2_new = self.policy.q_network(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.policy.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # تنظیم اتوماتیک alpha
        if self.policy.auto_entropy_tuning:
            alpha_loss = -(self.policy.log_alpha * (log_probs + self.policy.target_entropy).detach()).mean()
            
            self.policy.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.policy.alpha_optimizer.step()
            self.policy.alpha = self.policy.log_alpha.exp().item()
            self.policy.alpha = self.policy.log_alpha.exp()
        
        # به‌روزرسانی اولویت‌ها
        with torch.no_grad():
            td_error = torch.abs(current_q1 - target_q).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_error)
        
        # به‌روزرسانی شبکه هدف
        self.policy._update_target_network(self.tau)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.policy.alpha.item() if self.policy.auto_entropy_tuning else self.policy.alpha,
            'avg_q': q_new.mean().item()
        }
    
    def collect_experience(self, num_steps: int = 1) -> Dict[str, Any]:
        """جمع‌آوری تجربه از محیط"""
        episode_info = []
        
        for _ in range(num_steps):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # انتخاب action
                action = self.policy.act(state)
                
                # اجرای action در محیط
                next_state, reward, done, info = self.env.step(action)
                
                # ذخیره transition
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info.copy()
                )
                
                priority = abs(reward) + 1.0
                self.replay_buffer.push(transition, priority)
                
                # به‌روزرسانی آمار
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                if done:
                    break
            
            # ذخیره اطلاعات اپیزود
            episode_summary = self.env.get_episode_summary()
            episode_summary.update({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'total_steps': self.total_steps
            })
            
            episode_info.append(episode_summary)
            self.total_episodes += 1
            
            # به‌روزرسانی بهترین reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model('best_model.pt')
        
        return {
            'episodes': episode_info,
            'buffer_size': len(self.replay_buffer)
        }
    
    def train(self, total_timesteps: int, eval_env: Optional[EOEnvironment] = None):
        """آموزش کامل"""
        print(f"Starting SAC training for {total_timesteps} timesteps...")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Device: {self.device}")
        
        # تنظیمات WandB (اختیاری)
        use_wandb = False
        if use_wandb:
            wandb.init(project="eo-rl-sac", config=self.config)
        
        # نوار پیشرفت
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        # ذخیره تاریخچه
        history = {
            'rewards': [],
            'episode_lengths': [],
            'losses': [],
            'improvements': []
        }
        
        while self.total_steps < total_timesteps:
            # جمع‌آوری تجربه
            collect_info = self.collect_experience(num_steps=5)
            
            # آموزش
            train_info = self.train_step()
            
            # به‌روزرسانی نوار پیشرفت
            pbar.update(5)
            pbar.set_postfix({
                'steps': self.total_steps,
                'buffer': len(self.replay_buffer),
                'best_r': self.best_reward
            })
            
            # ذخیره تاریخچه
            if collect_info['episodes']:
                episode = collect_info['episodes'][-1]
                history['rewards'].append(episode['episode_reward'])
                history['episode_lengths'].append(episode['episode_length'])
                history['improvements'].append(episode.get('improvement', 0))
            
            if train_info:
                history['losses'].append(train_info.get('q_loss', 0))
            
            # ارسال به WandB
            if use_wandb and collect_info['episodes']:
                wandb.log({
                    'total_steps': self.total_steps,
                    'episode_reward': episode['episode_reward'],
                    'episode_length': episode['episode_length'],
                    'improvement': episode.get('improvement', 0),
                    'q_loss': train_info.get('q_loss', 0),
                    'policy_loss': train_info.get('policy_loss', 0),
                    'alpha': train_info.get('alpha', 0),
                    'buffer_size': len(self.replay_buffer)
                })
            
            # ارزیابی دوره‌ای
            if eval_env and self.total_steps % config.TRAINING_CONFIG['eval_freq'] == 0:
                eval_results = self.evaluate(eval_env, num_episodes=3)
                print(f"\nEvaluation at step {self.total_steps}:")
                print(f"  Avg reward: {eval_results['avg_reward']:.2f}")
                print(f"  Avg improvement: {eval_results['avg_improvement']:.2f}")
                
                if use_wandb:
                    wandb.log({
                        'eval_avg_reward': eval_results['avg_reward'],
                        'eval_avg_improvement': eval_results['avg_improvement']
                    })
            
            # ذخیره دوره‌ای مدل
            if self.total_steps % config.TRAINING_CONFIG['save_freq'] == 0:
                self.save_model(f'checkpoint_step_{self.total_steps}.pt')
        
        pbar.close()
        
        # ذخیره مدل نهایی
        self.save_model('final_model.pt')
        
        # ذخیره تاریخچه
        self.save_training_history(history)
        
        # رسم نمودارها
        self.plot_training_history(history)
        
        print(f"\nTraining completed!")
        print(f"Total steps: {self.total_steps}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Best reward: {self.best_reward:.2f}")
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self, eval_env: EOEnvironment, num_episodes: int = 5) -> Dict[str, float]:
        """ارزیابی مدل آموزش‌دیده"""
        rewards = []
        improvements = []
        
        for _ in range(num_episodes):
            state = eval_env.reset()
            episode_reward = 0
            
            while True:
                action = self.policy.act(state, deterministic=True)
                next_state, reward, done, info = eval_env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
            
            # استخراج بهبود SP از info
            episode_summary = eval_env.get_episode_summary()
            if episode_summary:
                improvements.append(episode_summary.get('improvement', 0))
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'num_episodes': num_episodes
        }
    
    def save_model(self, filename: str):
        """ذخیره مدل"""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.policy.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, filename: str):
        """بارگذاری مدل"""
        path = os.path.join(self.save_dir, filename)
        self.policy.load(path)
        print(f"Model loaded from {path}")
    
    def save_training_history(self, history: Dict):
        """ذخیره تاریخچه آموزش"""
        path = os.path.join(self.log_dir, 'training_history.npy')
        np.save(path, history)
        print(f"Training history saved to {path}")
    
    def plot_training_history(self, history: Dict):
        """رسم نمودارهای تاریخچه آموزش"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # نمودار rewards
        axes[0, 0].plot(history['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # نمودار improvements
        axes[0, 1].plot(history['improvements'])
        axes[0, 1].set_title('SP Improvements')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Improvement')
        axes[0, 1].grid(True)
        
        # نمودار losses
        axes[1, 0].plot(history['losses'])
        axes[1, 0].set_title('Q Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # نمودار episode lengths
        axes[1, 1].plot(history['episode_lengths'])
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Length')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # ذخیره نمودار
        plot_path = os.path.join(self.log_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=300)
        print(f"Training plots saved to {plot_path}")
        
        plt.show()