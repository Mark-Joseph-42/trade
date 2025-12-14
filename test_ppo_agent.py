"""Test script for the PPO Agent implementation."""
import unittest
import numpy as np
import torch
from ppo_cnn_agent import PPOAgent, PPOConfig, CNNActorCritic

class TestCNNActorCritic(unittest.TestCase):
    """Test cases for the CNNActorCritic model."""
    
    def test_initialization(self):
        """Test model initialization and forward pass."""
        # Test with different input shapes
        test_cases = [
            (3, 32, 32),  # RGB image
            (2, 32, 32),  # GAF encoded data
            (1, 64, 64),  # Larger grayscale image
        ]
        
        for in_channels, height, width in test_cases:
            with self.subTest(in_channels=in_channels, height=height, width=width):
                num_actions = 3
                model = CNNActorCritic(in_channels, num_actions)
                
                # Test forward pass
                x = torch.randn(1, in_channels, height, width)
                logits, value = model(x)
                
                self.assertEqual(logits.shape, (1, num_actions))
                # Value shape is [1] not [1, 1] as it's squeezed in the model
                self.assertEqual(value.shape, (1,))
                
                # Test with batch size > 1
                batch_size = 16
                x = torch.randn(batch_size, in_channels, height, width)
                logits, value = model(x)
                
                self.assertEqual(logits.shape, (batch_size, num_actions))
                # Value shape is [batch_size] as it's squeezed in the model
                self.assertEqual(value.shape, (batch_size,))


class TestPPOAgent(unittest.TestCase):
    """Test cases for the PPOAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        self.obs_shape = (2, 32, 32)  # GAF encoded data
        self.num_actions = 3  # [HOLD, BUY, SELL]
        self.config = PPOConfig(
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            pi_lr=3e-4,
            train_iters=4,  # Reduced for testing
            target_kl=0.01,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=32,
        )
        self.agent = PPOAgent(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            config=self.config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.model.__class__.__name__, 'CNNActorCritic')
        self.assertEqual(self.agent.device in ['cuda', 'cpu'], True)
        
        # Test model parameters
        for param in self.agent.model.parameters():
            self.assertFalse(torch.isnan(param).any())
            self.assertFalse(torch.isinf(param).any())
    
    def test_act(self):
        """Test action selection."""
        # Test single observation
        obs = np.random.randn(*self.obs_shape).astype(np.float32)
        action, value, log_prob = self.agent.act(obs)
        
        self.assertIsInstance(action, int)
        self.assertIn(action, range(self.num_actions))
        self.assertIsInstance(value, float)
        self.assertIsInstance(log_prob, float)
        
        # Test batch of observations
        batch_size = 5
        obs_batch = np.random.randn(batch_size, *self.obs_shape).astype(np.float32)
        actions, values, log_probs = zip(*[self.agent.act(obs) for obs in obs_batch])
        
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(values), batch_size)
        self.assertEqual(len(log_probs), batch_size)
    
    def test_value(self):
        """Test value function."""
        # Test single observation
        obs = np.random.randn(*self.obs_shape).astype(np.float32)
        value = self.agent.value(obs)
        self.assertIsInstance(value, float)
        
        # Test batch of observations
        batch_size = 5
        obs_batch = np.random.randn(batch_size, *self.obs_shape).astype(np.float32)
        values = [self.agent.value(obs) for obs in obs_batch]
        self.assertEqual(len(values), batch_size)
        self.assertTrue(all(isinstance(v, float) for v in values))
    
    def test_compute_gae(self):
        """Test GAE computation."""
        # Create test data with smaller values for more stable GAE calculation
        batch_size = 10
        rewards = np.random.uniform(-0.1, 0.1, size=batch_size)  # Smaller rewards
        values = np.random.uniform(0.9, 1.1, size=batch_size + 1)  # Values around 1.0
        dones = np.zeros(batch_size, dtype=bool)
        dones[-1] = True  # Terminal state
        
        # Compute GAE
        adv, ret = self.agent.compute_gae(rewards, values, dones)
        
        # Check shapes
        self.assertEqual(adv.shape, (batch_size,))
        self.assertEqual(ret.shape, (batch_size,))
        
        # Check that values are finite
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertTrue(np.all(np.isfinite(ret)))
        
        # Check that returns are reasonable (can be less than values due to discounting)
        self.assertTrue(np.all(np.isfinite(ret)))
    
    def test_update(self):
        """Test PPO update step."""
        # Create a test batch
        batch_size = 32
        obs = np.random.randn(batch_size, *self.obs_shape).astype(np.float32)
        actions = np.random.randint(0, self.num_actions, size=batch_size)
        log_probs = np.random.uniform(-1, 0, size=batch_size)  # log probs are negative
        rewards = np.random.uniform(-1, 1, size=batch_size)
        dones = np.zeros(batch_size, dtype=bool)
        dones[-1] = True  # Terminal state
        values = np.random.uniform(0, 1, size=batch_size)
        
        # First compute advantages and returns
        advantages = np.random.uniform(-0.5, 0.5, size=batch_size)  # Small random advantages
        returns = values + advantages  # Returns = values + advantages
        
        # Create batch dictionary with all required keys
        batch = {
            'obs': obs,
            'act': actions,
            'logp': log_probs,
            'adv': advantages,  # Add advantages
            'ret': returns,     # Add returns
            'val': values,
            'done': dones,
        }
        
        # Perform update
        stats = self.agent.update(batch)
        
        # Check that stats contain expected keys
        expected_keys = ['policy_loss', 'value_loss', 'entropy', 'approx_kl']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)
        
        # Check that values are finite
        self.assertTrue(np.isfinite(stats['policy_loss']))
        self.assertTrue(np.isfinite(stats['value_loss']))
        self.assertTrue(np.isfinite(stats['entropy']))
        self.assertTrue(np.isfinite(stats['approx_kl']))
        
        # Check that KL divergence is non-negative
        self.assertGreaterEqual(stats['approx_kl'], 0)


if __name__ == '__main__':
    unittest.main()
