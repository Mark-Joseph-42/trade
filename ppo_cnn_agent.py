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
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    train_iters: int = 80
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64


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
        value = self.value_head(z).squeeze(-1)
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.pi_lr)

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
        logits, value = self.model(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        ratio = torch.exp(logp - logp_old)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, ret)
        entropy = dist.entropy().mean()
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        with torch.no_grad():
            log_ratio = logp - logp_old
            approx_kl = 0.5 * (log_ratio.pow(2)).mean().item()
        info = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl),
        }
        return loss, info

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        act = torch.from_numpy(batch["act"]).long().to(self.device)
        logp_old = torch.from_numpy(batch["logp"]).float().to(self.device)
        adv = torch.from_numpy(batch["adv"]).float().to(self.device)
        ret = torch.from_numpy(batch["ret"]).float().to(self.device)

        n = obs.size(0)
        idx = np.arange(n)
        last_info: Dict[str, float] = {}

        for _ in range(self.config.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, self.config.batch_size):
                end = start + self.config.batch_size
                mb_idx = idx[start:end]
                mb_obs = obs[mb_idx]
                mb_act = act[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                loss, info = self._compute_loss(mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                last_info = info

            if last_info.get("approx_kl", 0.0) > 1.5 * self.config.target_kl:
                break

        return last_info

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
