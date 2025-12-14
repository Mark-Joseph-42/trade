from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.1  # Reduced from 0.2 for more stable updates
    pi_lr: float = 1e-4  # Reduced from 3e-4 for smaller steps
    vf_lr: float = 1e-4  # Separate learning rate for value function
    train_iters: int = 4  # Reduced from 80 for stability
    target_kl: float = 0.015  # Slightly increased to allow more exploration
    entropy_coef: float = 0.02  # Increased to encourage exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    value_clip_ratio: float = 0.2
    max_train_iters: int = 10
    minibatch_size: int = 32  # Add minibatch size for better stability


class CNNActorCritic(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x)
        z = self.fc(z)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)  # Remove last dimension for value
        return logits, value


class PPOAgent:
    def __init__(
        self,
        obs_shape,
        num_actions: int,
        config: PPOConfig,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = int(obs_shape[0])
        self.model = CNNActorCritic(in_channels, num_actions).to(self.device)
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        self.model.apply(init_weights)
        
        # Separate optimizers for policy and value function
        self.pi_optimizer = optim.AdamW(
            list(self.model.policy_head.parameters()) + list(self.model.features.parameters()),
            lr=self.config.pi_lr,
            eps=1e-5,
            weight_decay=0.01
        )
        self.vf_optimizer = optim.AdamW(
            list(self.model.value_head.parameters()) + list(self.model.features.parameters()),
            lr=self.config.vf_lr,
            eps=1e-5,
            weight_decay=0.01
        )
        
        # Gradient scaler for mixed precision training
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)
        else:
            self.scaler = None

    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.model.eval()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.squeeze(0).item())

    def value(self, obs: np.ndarray) -> float:
        self.model.eval()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(obs_t)
        return float(value.squeeze(0).item())

    def _compute_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp_old: torch.Tensor,
        adv: torch.Tensor,
        ret: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Add input validation
        if not torch.isfinite(obs).all():
            raise ValueError("Non-finite values in observations")
            
        with torch.amp.autocast(
            device_type='cuda' if torch.cuda.is_available() else 'cpu',
            enabled=torch.cuda.is_available()
        ):
            # Forward pass
            logits, value = self.model(obs)
            
            # Policy loss
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act)
            ratio = torch.exp(logp - logp_old)
            
            # Advantage normalization with numerical stability
            adv_mean = adv.mean()
            adv_std = adv.std()
            if torch.isnan(adv_std) or adv_std < 1e-6:
                adv_std = torch.ones_like(adv_std)
            # Clip advantages to reduce variance
            adv_norm = torch.clamp(
                (adv - adv_mean) / (adv_std + 1e-8),
                -10.0,  # Clip to prevent extreme values
                10.0
            )
            
            # Clipped surrogate objective with more conservative clipping
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config.clip_ratio, 
                1.0 + self.config.clip_ratio
            ) * adv_norm
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function update with separate learning rate
            value = value.squeeze(-1)
            value_clipped = value
            if self.config.value_clip_ratio > 0:
                value_old = value.detach()
                value_clipped = value_old + torch.clamp(
                    value - value_old, 
                    -self.config.value_clip_ratio, 
                    self.config.value_clip_ratio
                )
            
            # Ensure shapes match for value loss
            ret = ret.view(-1)  # Ensure ret is [batch_size]
            value_clipped = value_clipped.view(-1)  # Ensure value_clipped is [batch_size]
            
            # Value loss with clipping
            value_loss = 0.5 * F.mse_loss(value_clipped, ret, reduction='mean')
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss with entropy regularization
            loss = (
                policy_loss 
                + self.config.value_coef * value_loss 
                - self.config.entropy_coef * entropy
            )
            
            # KL divergence for early stopping
            with torch.no_grad():
                log_ratio = logp - logp_old
                approx_kl = 0.5 * (log_ratio.pow(2)).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                
        # Logging
        info = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl),
            "clip_frac": float(clip_frac),
            "value_mean": float(value.mean().item()),
            "value_std": float(value.std().item()),
        }
        
        return loss, info

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        # Convert numpy arrays to PyTorch tensors
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        act = torch.from_numpy(batch["act"]).long().to(self.device)
        logp_old = torch.from_numpy(batch["logp"]).float().to(self.device)
        adv = torch.from_numpy(batch["adv"]).float().to(self.device)
        ret = torch.from_numpy(batch["ret"]).float().to(self.device)
        
        # Input validation
        for name, tensor in [("obs", obs), ("act", act), ("logp_old", logp_old), 
                           ("adv", adv), ("ret", ret)]:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values in {name}")

        n = obs.size(0)
        idx = np.arange(n)
        
        # Initialize info dictionary
        info = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0,
            'explained_variance': 0.0
        }
        
        # Early stopping flag
        early_stop = False

        for i in range(self.config.train_iters):
            if early_stop:
                break
                
            # Shuffle indices for minibatch updates
            np.random.shuffle(idx)
            
            # Process in minibatches
            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                mb_idx = idx[start:end]
                
                # Get minibatch
                mb_obs = obs[mb_idx].contiguous()
                mb_act = act[mb_idx].contiguous()
                mb_logp_old = logp_old[mb_idx].contiguous()
                mb_adv = adv[mb_idx].contiguous()
                mb_ret = ret[mb_idx].contiguous()
                
                # Compute policy loss
                with torch.amp.autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    enabled=torch.cuda.is_available()
                ):
                    # Forward pass for policy
                    logits, _ = self.model(mb_obs)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(mb_act)
                    ratio = torch.exp(logp - mb_logp_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                
                # Update policy
                self.pi_optimizer.zero_grad()
                if torch.cuda.is_available() and self.scaler is not None:
                    self.scaler.scale(policy_loss).backward()
                    self.scaler.unscale_(self.pi_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.policy_head.parameters()) + list(self.model.features.parameters()),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True
                    )
                    self.scaler.step(self.pi_optimizer)
                else:
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.policy_head.parameters()) + list(self.model.features.parameters()),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True
                    )
                    self.pi_optimizer.step()
                
                # Compute value loss with a fresh forward pass
                with torch.amp.autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    enabled=torch.cuda.is_available()
                ):
                    # Forward pass for value function
                    _, value = self.model(mb_obs)
                    value = value.squeeze(-1)
                    
                    # Detach the value for clipping to prevent backprop through the clipping
                    value_detached = value.detach()
                    value_clipped = value_detached + torch.clamp(
                        value - value_detached,
                        -self.config.value_clip_ratio,
                        self.config.value_clip_ratio
                    )
                    # Ensure shapes match
                    value_target = mb_ret.view(-1)
                    value_loss = 0.5 * F.mse_loss(value_clipped, value_target, reduction='mean')
                
                # Update value function
                self.vf_optimizer.zero_grad()
                if torch.cuda.is_available() and self.scaler is not None:
                    self.scaler.scale(value_loss).backward()
                    self.scaler.unscale_(self.vf_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.value_head.parameters()) + list(self.model.features.parameters()),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True
                    )
                    self.scaler.step(self.vf_optimizer)
                    self.scaler.update()
                else:
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.value_head.parameters()) + list(self.model.features.parameters()),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True
                    )
                    self.vf_optimizer.step()
                
                # Update training statistics
                with torch.no_grad():
                    # Compute KL divergence and other metrics
                    log_ratio = logp - mb_logp_old
                    approx_kl = 0.5 * (log_ratio.pow(2)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
                    
                    # Update info with current batch statistics
                    info.update({
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item(),
                        'entropy': dist.entropy().mean().item(),
                        'approx_kl': approx_kl,
                        'clip_frac': clip_frac,
                        'explained_variance': float(1 - (mb_ret.view(-1) - value.detach()).var() / (mb_ret.view(-1).var() + 1e-8))
                    })
                
                # Check for NaN/Inf in gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"Warning: Non-finite gradients in {name}")
                        early_stop = True
                        break
                
                if early_stop:
                    break
                    
                # Early stopping based on KL divergence - only check after first iteration
                if i > 0 and info.get("approx_kl", float('inf')) > 10.0 * self.config.target_kl:
                    print(f"Early stopping at iteration {i} due to high KL divergence")
                    early_stop = True
                    break
                    
                # If we're on the first iteration and KL is very high, adjust the learning rate
                if i == 0 and info.get("approx_kl", 0) > 1.0:
                    old_lr = self.config.pi_lr
                    self.config.pi_lr = max(1e-6, self.config.pi_lr * 0.1)  # Reduce learning rate
                    self.config.vf_lr = max(1e-6, self.config.vf_lr * 0.1)
                    print(f"High initial KL divergence ({info.get('approx_kl', 0):.4f}), reducing learning rate from {old_lr} to {self.config.pi_lr}")
                    
                    # Update optimizers with new learning rates
                    for param_group in self.pi_optimizer.param_groups:
                        param_group['lr'] = self.config.pi_lr
                    for param_group in self.vf_optimizer.param_groups:
                        param_group['lr'] = self.config.vf_lr
                
                # Early stopping if policy collapses
                if info.get("entropy", 0.0) < 0.1:  # Threshold for entropy collapse
                    print(f"Early stopping at iteration {i} due to low entropy")
                    early_stop = True
                    break

        return info

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = rewards.shape[0]
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values[t + 1] * nonterminal - values[t]
            lastgaelam = delta + self.config.gamma * self.config.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values[:-1]
        return adv.astype(np.float32), ret.astype(np.float32)
