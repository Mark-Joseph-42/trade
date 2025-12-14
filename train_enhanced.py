"""Training script for the enhanced trading environment with PPO agent."""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from enhanced_ppo_agent import PPOAgent, PPOConfig
from enhanced_rl_env import EnhancedTradingEnv
from enhanced_data_pipeline import DataPipeline, DataConfig


def create_env(price_df: pd.DataFrame) -> EnhancedTradingEnv:
    """Create and return a trading environment."""
    return EnhancedTradingEnv(
        price_df=price_df,
        window_size=64,
        image_size=32,
        initial_balance=10000.0,
        fee_rate=0.001,
        mdd_cap=0.10,
        use_gaf=True,
        normalize=True
    )


def train():
    """Train the PPO agent on the trading environment."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set up logging
    log_dir = "runs/ppo_trading"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Download and prepare data
    print("Downloading and preparing data...")
    data_config = DataConfig(
        symbols=["AAPL"],
        timeframes=["1d"],
        start_date="2020-01-01",
        end_date="2023-01-01"
    )
    pipeline = DataPipeline(data_config)
    data = pipeline.run()
    
    # Get the processed data
    price_df = data["AAPL"]["1d"]
    
    # Create environments
    env = create_env(price_df)
    test_env = create_env(price_df)  # Separate environment for testing
    
    # Initialize agent
    obs_shape = env.observation_shape
    num_actions = env.action_space_n
    
    config = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_iters=4,
        target_kl=0.01,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        batch_size=2048,  # Increased batch size for more stable training
        minibatch_size=64,
        value_clip_ratio=0.2
    )
    
    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        config=config,
        device=device
    )
    
    # Training parameters
    num_episodes = 1000
    max_steps = 1000
    save_interval = 10
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_reward = -float('inf')
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Run episode
        while not done and episode_steps < max_steps:
            # Get action from agent
            action, log_prob, value = agent.act(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=log_prob,
                done=done
            )
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Store the transition
            agent.store_transition(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=log_prob,
                done=done
            )
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Check if we have enough samples for an update
            if agent.buffer.ptr >= agent.config.batch_size or done:
                # Compute GAE and returns for the current trajectory
                with torch.no_grad():
                    _, _, last_val = agent.act(obs)
                    agent.buffer.finish_path(last_val=last_val)
                
                # Only update if we have enough samples
                if agent.buffer.ptr >= agent.config.batch_size:
                    stats = agent.update()
                    
                    # Log training metrics
                    if stats:
                        writer.add_scalar('train/loss_policy', stats['policy_loss'], global_step)
                        writer.add_scalar('train/loss_value', stats['value_loss'], global_step)
                        writer.add_scalar('train/entropy', stats['entropy'], global_step)
                        writer.add_scalar('train/kl', stats['approx_kl'], global_step)
                        
                        if 'clip_frac' in stats:
                            writer.add_scalar('train/clip_frac', stats['clip_frac'], global_step)
        
        # Log episode metrics
        writer.add_scalar('train/episode_reward', episode_reward, episode)
        writer.add_scalar('train/episode_length', episode_steps, episode)
        
        # Evaluate agent
        if episode % 5 == 0:
            eval_reward = evaluate_agent(agent, test_env, num_episodes=3)
            writer.add_scalar('eval/mean_reward', eval_reward, episode)
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                save_path = os.path.join(log_dir, f"best_model.pt")
                torch.save({
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'config': config,
                    'episode': episode,
                    'reward': eval_reward,
                }, save_path)
                print(f"\nNew best model saved with reward: {eval_reward:.2f}")
        
        # Print progress
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Eval Reward: {eval_reward if 'eval_reward' in locals() else 'N/A':.2f}")
    
    # Save final model
    final_path = os.path.join(log_dir, "final_model.pt")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'config': config,
        'episode': num_episodes,
        'reward': episode_reward,
    }, final_path)
    
    print("\nTraining completed!")
    print(f"Best evaluation reward: {best_reward:.2f}")
    print(f"Final model saved to: {final_path}")
    
    # Close environments
    env.close()
    test_env.close()
    writer.close()


def evaluate_agent(agent, env, num_episodes: int = 10) -> float:
    """Evaluate the agent on the environment."""
    total_reward = 0.0
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _ = agent.act(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


if __name__ == "__main__":
    train()
