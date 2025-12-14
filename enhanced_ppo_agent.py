"""Enhanced PPO Agent with experience replay buffer."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_iters: int = 4
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    minibatch_size: int = 32
    value_clip_ratio: float = 0.2
    max_train_iters: int = 1000


class PPOBuffer:
    """Buffer for storing trajectories."""
    
    def __init__(self, obs_dim: tuple, size: int, gamma: float = 0.99, lam: float = 0.95):
        """Initialize buffer."""
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        val: float,
        logp: float,
        done: bool
    ) -> None:
        """Store a single transition."""
        assert self.ptr < self.max_size, "Buffer overflow"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val: float = 0) -> None:
        """Calculate GAE and rewards-to-go."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Rewards-to-go
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.ptr
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all data from buffer."""
        assert self.ptr == self.max_size, f"Buffer not full (has {self.ptr} items, needs {self.max_size})"
        self.ptr, self.path_start_idx = 0, 0
        
        # Advantage normalization
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        return {
            'obs': self.obs_buf,
            'act': self.act_buf,
            'ret': self.ret_buf,
            'adv': self.adv_buf,
            'logp_old': self.logp_buf,
            'val_old': self.val_buf
        }
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """Compute discounted cumulative sums."""
        return np.cumsum(x * discount ** np.arange(len(x)))


class CNNActorCritic(nn.Module):
    """CNN-based actor-critic network."""
    
    def __init__(self, in_channels: int, num_actions: int):
        """Initialize the network."""
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Feature extraction
        features = self.features(x)
        
        # Policy and value heads
        logits = self.policy(features)
        value = self.value(features).squeeze(-1)
        
        # Action distribution
        dist = Categorical(logits=logits)
        
        return dist, value, logits


class PPOAgent:
    """PPO Agent with experience replay."""
    
    def __init__(
        self,
        obs_shape: tuple,
        num_actions: int,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the agent."""
        self.config = config or PPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and optimizer
        self.model = CNNActorCritic(obs_shape[0], num_actions).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.model.policy.parameters(), 'lr': self.config.pi_lr},
            {'params': self.model.value.parameters(), 'lr': self.config.vf_lr}
        ])
        
        # Buffer for storing trajectories
        self.buffer = PPOBuffer(
            obs_dim=obs_shape,
            size=self.config.batch_size,
            gamma=self.config.gamma,
            lam=self.config.lam
        )
        
        # Training state
        self.step_count = 0
    
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select an action using the current policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            dist, value, _ = self.model(obs_tensor)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        val: float,
        logp: float,
        done: bool
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.store(obs, act, rew, val, logp, done)
    
    def update(self) -> Dict[str, float]:
        """Update the policy and value networks."""
        # Get all data from buffer
        data = self.buffer.get()
        obs = torch.FloatTensor(data['obs']).to(self.device)
        act = torch.LongTensor(data['act']).to(self.device)
        ret = torch.FloatTensor(data['ret']).to(self.device)
        adv = torch.FloatTensor(data['adv']).to(self.device)
        logp_old = torch.FloatTensor(data['logp_old']).to(self.device)
        val_old = torch.FloatTensor(data['val_old']).to(self.device)
        
        # Training statistics
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_frac': []
        }
        
        # Train for the given number of iterations
        for _ in range(self.config.train_iters):
            # Get minibatches
            indices = np.random.permutation(self.config.batch_size)
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = obs[mb_indices]
                mb_act = act[mb_indices]
                mb_ret = ret[mb_indices]
                mb_adv = adv[mb_indices]
                mb_logp_old = logp_old[mb_indices]
                mb_val_old = val_old[mb_indices]
                
                # Forward pass
                dist, value, _ = self.model(mb_obs)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                
                # Policy loss (clipped surrogate objective)
                ratio = (logp - mb_logp_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_clipped = mb_val_old + (value - mb_val_old).clamp(
                    -self.config.value_clip_ratio,
                    self.config.value_clip_ratio
                )
                value_loss1 = (value - mb_ret).pow(2)
                value_loss2 = (value_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Log statistics
                with torch.no_grad():
                    # KL divergence
                    approx_kl = (mb_logp_old - dist.log_prob(mb_act)).mean().item()
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                    
                    stats['policy_loss'].append(policy_loss.item())
                    stats['value_loss'].append(value_loss.item())
                    stats['entropy'].append(entropy.item())
                    stats['approx_kl'].append(approx_kl)
                    stats['clip_frac'].append(clip_fraction)
                
                # Early stopping if KL divergence is too large
                if approx_kl > 1.5 * self.config.target_kl:
                    break
        
        # Compute mean statistics
        return {k: np.mean(v) for k, v in stats.items()}
    
    def save(self, path: str) -> None:
        """Save the model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.step_count
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PPOAgent':
        """Load a trained model."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Create agent
        # Note: obs_shape and num_actions need to be set correctly
        agent = cls(obs_shape=(1, 32, 32), num_actions=3, config=config, device=device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.step_count = checkpoint['step']
        
        return agent
