"""Minimal test for PPO agent."""
import unittest
import numpy as np
import torch
import torch.nn as nn
from ppo_cnn_agent import PPOAgent, PPOConfig, CNNActorCritic

class TestMinimal(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (2, 32, 32)
        self.num_actions = 3
        self.batch_size = 4
        self.config = PPOConfig(
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            pi_lr=1e-4,
            train_iters=1,
            batch_size=2,
            max_train_iters=1
        )
        
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = CNNActorCritic(self.obs_shape[0], self.num_actions)
        x = torch.randn(1, *self.obs_shape)
        logits, value = model(x)
        self.assertEqual(logits.shape, (1, self.num_actions))
        self.assertEqual(value.shape, (1,))  # Value should be 1D
        
    def test_act_method(self):
        """Test act method of PPOAgent."""
        agent = PPOAgent(self.obs_shape, self.num_actions, self.config)
        obs = np.random.randn(*self.obs_shape).astype(np.float32)
        action, log_prob, value = agent.act(obs)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.num_actions)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        
    def test_compute_loss(self):
        """Test loss computation."""
        agent = PPOAgent(self.obs_shape, self.num_actions, self.config)
        batch = {
            'obs': np.random.randn(self.batch_size, *self.obs_shape).astype(np.float32),
            'act': np.random.randint(0, self.num_actions, size=(self.batch_size,)),
            'logp': np.random.randn(self.batch_size).astype(np.float32),
            'adv': np.random.randn(self.batch_size).astype(np.float32),
            'ret': np.random.randn(self.batch_size, 1).astype(np.float32)
        }
        
        # Convert to tensors
        obs = torch.from_numpy(batch['obs']).to(agent.device)
        act = torch.from_numpy(batch['act']).to(agent.device)
        logp_old = torch.from_numpy(batch['logp']).to(agent.device)
        adv = torch.from_numpy(batch['adv']).to(agent.device)
        ret = torch.from_numpy(batch['ret']).to(agent.device)
        
        # Test _compute_loss
        loss, info = agent._compute_loss(obs, act, logp_old, adv, ret)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertIn('policy_loss', info)
        self.assertIn('value_loss', info)
        self.assertIn('entropy', info)
        
    def test_update_step(self):
        """Test a single update step."""
        agent = PPOAgent(self.obs_shape, self.num_actions, self.config)
        batch = {
            'obs': np.random.randn(self.batch_size, *self.obs_shape).astype(np.float32),
            'act': np.random.randint(0, self.num_actions, size=(self.batch_size,)),
            'logp': np.random.randn(self.batch_size).astype(np.float32),
            'adv': np.random.randn(self.batch_size).astype(np.float32),
            'ret': np.random.randn(self.batch_size, 1).astype(np.float32),
            'rew': np.random.randn(self.batch_size).astype(np.float32),
            'val': np.random.randn(self.batch_size, 1).astype(np.float32),
            'done': np.zeros(self.batch_size, dtype=bool)
        }
        
        # Test update
        stats = agent.update(batch)
        self.assertIsInstance(stats, dict)
        self.assertIn('policy_loss', stats)
        self.assertIn('value_loss', stats)
        self.assertIn('entropy', stats)

if __name__ == '__main__':
    unittest.main()
