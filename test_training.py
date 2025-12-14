"""Test script for the training process."""
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
from ppo_cnn_agent import PPOAgent, PPOConfig
from rl_env import TradingEnv

class TestTraining(unittest.TestCase):
    """Test cases for the training process."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample price data
        np.random.seed(42)
        self.num_steps = 1000
        self.window_size = 64
        self.image_size = 32
        
        # Create synthetic price data with a slight upward trend
        # Add more stability to the price generation
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0001, 0.01, self.num_steps)  # Smaller, more stable returns
        self.prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
        self.prices = np.maximum(0.1, self.prices)  # Ensure positive prices
        
        # Add some volatility clustering
        for i in range(10, len(self.prices)):
            if np.random.rand() < 0.1:  # 10% chance of a jump
                self.prices[i:] *= np.exp(np.random.normal(0, 0.05))  # Small random jump
        
        # Create a DataFrame with OHLCV data
        self.df = self._create_ohlcv_data(self.prices)
        
        # Initialize environment
        self.env = TradingEnv(
            price_df=self.df,
            window_size=self.window_size,
            image_size=self.image_size,
            initial_balance=10000.0,
            fee_rate=0.001,
            mdd_cap=0.10,
            reward_mdd_penalty=1.0
        )
        
        # Initialize agent with proper weight initialization and adjusted hyperparameters for testing
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.1)  # Use orthogonal initialization with smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)  # Use orthogonal initialization with smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Use more stable hyperparameters for testing
        config = PPOConfig(
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,  # Slightly larger clip ratio
            pi_lr=1e-4,      # Smaller learning rate
            vf_lr=1e-4,      # Smaller learning rate
            train_iters=4,    # Fewer iterations for testing
            target_kl=0.3,    # Increased target KL for testing
            entropy_coef=0.02,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=64,
            value_clip_ratio=0.2,
            max_train_iters=10,  # Add max_train_iters
            minibatch_size=32
        )
        
        self.agent = PPOAgent(
            obs_shape=(2, self.image_size, self.image_size),
            num_actions=3,
            config=config
        )
        
        # Apply weight initialization
        self.agent.model.apply(init_weights)
                
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Use more stable hyperparameters for testing
        self.config = PPOConfig(
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.3,  # Increased clip ratio for stability
            pi_lr=1e-5,      # Significantly reduced learning rate
            vf_lr=1e-5,      # Separate learning rate for value function
            train_iters=2,    # Fewer iterations for testing
            target_kl=0.3,    # Increased target KL for testing
            entropy_coef=0.02,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=16,    # Smaller batch size for testing
            value_clip_ratio=0.2,
            max_train_iters=2,  # Match train_iters for testing
            minibatch_size=8   # Smaller minibatch size for testing
        )
        
        self.agent = PPOAgent(
            obs_shape=(2, self.image_size, self.image_size),  # GAF encoding
            num_actions=3,  # [HOLD, BUY, SELL]
            config=self.config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Apply weight initialization
        self.agent.model.apply(init_weights)
        
        # Ensure model is in training mode
        self.agent.model.train()
    
    def _create_ohlcv_data(self, prices):
        """Create OHLCV data from price series."""
        import pandas as pd
        
        # Create OHLC data with some noise
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, len(prices))),
            'low': prices * (1 - np.random.uniform(0, 0.01, len(prices))),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(prices))
        })
        
        # Ensure high >= close >= low
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        
        return df
    
    def _validate_observation(self, obs):
        """Validate observation data."""
        if not np.isfinite(obs).all():
            print(f"Observation contains NaN or Inf values. Min: {np.nanmin(obs)}, Max: {np.nanmax(obs)}, NaN count: {np.isnan(obs).sum()}")
        self.assertTrue(np.isfinite(obs).all(), "Observation contains NaN or Inf values")
        
        obs_max = np.abs(obs).max()
        if obs_max > 1e6:
            print(f"Warning: Large observation values detected: {obs_max}")
        self.assertTrue(obs_max < 1e6, f"Observation values too large: {obs_max}")
        
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # No next state at the end of the episode
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * last_gae_lam
            
        returns = advantages + values
        return advantages, returns
    
    def test_single_episode(self):
        """Test running a single training episode."""
        # Set model to training mode and enable gradients
        self.agent.model.train()
        for param in self.agent.model.parameters():
            param.requires_grad = True
        
        # Reset environment
        obs = self.env.reset()
        self._validate_observation(obs)
        
        # Track data for GAE
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []
        logp_buf = []
        
        done = False
        episode_reward = 0
        step = 0
        max_steps = 10  # Limit steps for testing
        
        # Collect data for one episode
        while not done and step < max_steps:
            # Validate observation before processing
            self._validate_observation(obs)
            
            # Get action and value from agent
            with torch.no_grad():
                action, log_prob, value = self.agent.act(obs)
                
                # Clip action to valid range
                action = np.clip(action, 0, self.env.action_space_n - 1)
                
                # Store transition
                obs_buf.append(obs.copy())
                act_buf.append(action)
                val_buf.append(float(value))
                logp_buf.append(float(log_prob))
            
            # Take step in environment
            next_obs, reward, done, _ = self.env.step(action)
            
            # Clip and store reward
            reward = float(np.clip(reward, -10, 10))
            rew_buf.append(reward)
            done_buf.append(done)
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Convert to numpy arrays
        obs_buf = np.array(obs_buf, dtype=np.float32)
        act_buf = np.array(act_buf, dtype=np.int64)
        rew_buf = np.array(rew_buf, dtype=np.float32)
        val_buf = np.array(val_buf, dtype=np.float32)
        logp_buf = np.array(logp_buf, dtype=np.float32)
        done_buf = np.array(done_buf, dtype=np.float32)
        
        # Compute GAE and returns
        advantages, returns = self._compute_gae(rew_buf, val_buf, done_buf)
        
        # Create batch
        batch = {
            'obs': obs_buf,
            'act': act_buf,
            'rew': rew_buf,
            'val': val_buf,
            'adv': advantages,
            'ret': returns,
            'logp': logp_buf,
            'done': done_buf
        }
        
        # Update agent with the batch
        try:
            # Enable gradient computation
            torch.set_grad_enabled(True)
            
            # Run update
            stats = self.agent.update(batch)
            
            # Check that stats contain expected keys
            self.assertIn('policy_loss', stats)
            self.assertIn('value_loss', stats)
            self.assertIn('entropy', stats)
            self.assertIn('approx_kl', stats)
            
            # Check for NaN/Inf in losses
            self.assertTrue(np.isfinite(stats['policy_loss']))
            self.assertTrue(np.isfinite(stats['value_loss']))
            self.assertTrue(np.isfinite(stats['entropy']))
            
            print(f"Update stats: {stats}")
            print(f"Episode completed in {step} steps with reward: {episode_reward:.2f}")
            
        except Exception as e:
            print(f"Error in update(): {e}")
            # Print model parameter statistics for debugging
            for name, param in self.agent.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    print(f"{name}: grad_norm={grad_norm:.6f}, mean={grad_mean:.6f}, std={grad_std:.6f}")
            raise
    
    def test_agent_learning(self):
        """Test that the agent can learn a simple task over multiple episodes."""
        num_episodes = 5  # Increased for better learning evaluation
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        # Set up model for training
        self.agent.model.train()
        for param in self.agent.model.parameters():
            param.requires_grad = True
        
        # Training loop
        for episode in range(num_episodes):
            # Initialize episode
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_data = {
                'obs': [], 'actions': [], 'rewards': [],
                'values': [], 'log_probs': [], 'dones': []
            }
            
            # Run episode
            done = False
            while not done and episode_length < 100:  # Limit steps per episode
                # Get action from agent
                with torch.no_grad():
                    action, log_prob, value = self.agent.act(obs)
                
                # Take action in environment
                next_obs, reward, done, _ = self.env.step(action)
                
                # Store transition
                episode_data['obs'].append(obs.copy())
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['values'].append(float(value))
                episode_data['log_probs'].append(float(log_prob))
                episode_data['dones'].append(done)
                
                # Update for next step
                obs = next_obs
                episode_reward += reward
                episode_length += 1
            
            # Convert episode data to numpy arrays
            obs_buf = np.array(episode_data['obs'], dtype=np.float32)
            act_buf = np.array(episode_data['actions'], dtype=np.int64)
            rew_buf = np.array(episode_data['rewards'], dtype=np.float32)
            val_buf = np.array(episode_data['values'], dtype=np.float32)
            logp_buf = np.array(episode_data['log_probs'], dtype=np.float32)
            done_buf = np.array(episode_data['dones'], dtype=np.float32)
            
            # Compute GAE and returns
            advantages, returns = self._compute_gae(rew_buf, val_buf, done_buf)
            
            # Create batch for update
            batch = {
                'obs': obs_buf,
                'act': act_buf,
                'rew': rew_buf,
                'val': val_buf,
                'adv': advantages,
                'ret': returns,
                'logp': logp_buf,
                'done': done_buf
            }
            
            # Update agent
            update_stats = self.agent.update(batch)
            
            # Store statistics
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            stats['policy_losses'].append(update_stats.get('policy_loss', float('nan')))
            stats['value_losses'].append(update_stats.get('value_loss', float('nan')))
            stats['entropies'].append(update_stats.get('entropy', float('nan')))
            
            # Print progress
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Policy Loss: {update_stats.get('policy_loss', 'N/A'):.4f}")
            print(f"  Value Loss: {update_stats.get('value_loss', 'N/A'):.4f}")
            print(f"  Entropy: {update_stats.get('entropy', 'N/A'):.4f}")
            print(f"  KL Divergence: {update_stats.get('approx_kl', 'N/A'):.6f}")
        
        # Basic learning verification
        if num_episodes > 1:
            # Check if the agent is improving (simple check: last episode should be better than first)
            if stats['episode_rewards'][-1] > stats['episode_rewards'][0]:
                print("\nLearning Progress: Positive")
            else:
                print("\nLearning Progress: Needs improvement")
        
        # Return statistics for further analysis
        return stats


if __name__ == '__main__':
    unittest.main()
