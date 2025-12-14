import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from hardware_profiler import get_hardware_profile, NodeRole
from ppo_cnn_agent import PPOAgent, PPOConfig
from trading_env import GAFTradingEnv

def create_sample_data(window_size=50, num_points=1000):
    """Generate sample price data for testing."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(num_points) * 0.5)
    return pd.DataFrame({
        'open': prices - np.random.uniform(0.1, 1, num_points),
        'high': prices + np.random.uniform(0.1, 1, num_points),
        'low': prices - np.random.uniform(0.1, 1, num_points),
        'close': prices,
        'volume': np.random.randint(100, 10000, num_points)
    })

# Constants
BUFFER_DIR = "data/shared_buffer"
MODEL_DIR = "models"
BATCH_SIZE = 2048

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(BUFFER_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def save_batch_to_buffer(buffer: List[Tuple], node_id: str):
    """Save collected batch to shared buffer."""
    timestamp = int(time.time())
    filename = f"batch_{timestamp}_{node_id}.pt"
    filepath = os.path.join(BUFFER_DIR, filename)
    torch.save(buffer, filepath)
    return filepath

def collect_data(env: GAFTradingEnv, model: PPOAgent, steps: int, device: str):
    """Run environment and collect experience."""
    buffer = []
    obs = env.reset()
    done = False
    total_steps = 0
    
    while not done and total_steps < steps:
        # Get action from policy
        with torch.no_grad():
            action, log_prob, _ = model.act(obs)
            action = np.array([action])  # Convert to numpy array for the environment
        
        # Take step in environment
        next_obs, reward, done, _ = env.step(action)
        
        # Store experience
        buffer.append((obs, action, reward, next_obs, done))
        
        # Update observation
        obs = next_obs
        total_steps += 1
        
        if total_steps % 100 == 0:
            env.render()
    
    return buffer

def run_actor(device: str):
    """Run actor node (CPU)."""
    print("Running in ACTOR mode (CPU)")
    
    # Initialize environment with sample data
    df = create_sample_data()
    env = GAFTradingEnv(df)
    
    # Initialize config and model
    config = PPOConfig()
    model = PPOAgent(env.observation_space.shape, env.action_space.n, config, device=device)
    model_path = os.path.join(MODEL_DIR, "global_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.model.eval()
    
    # Main loop
    while True:
        # Collect experience
        buffer = collect_data(env, model, BATCH_SIZE, device)
        
        # Save to shared buffer
        save_batch_to_buffer(buffer, "cpu_actor")
        
        # Load latest model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded updated model weights")

def run_learner(device: str):
    """Run learner node (GPU)."""
    print("Running in LEARNER mode (GPU)")
    
    # Initialize config and model
    config = PPOConfig()
    obs_shape = (1, 50, 50)  # Update if your GAF image size is different
    action_dim = 3  # Hold, Buy, Sell
    model = PPOAgent(obs_shape, action_dim, config, device=device)
    
    # Main training loop
    while True:
        # Check for new data
        batch_files = list(Path(BUFFER_DIR).glob("batch_*.pt"))
        
        if batch_files:
            print(f"Found {len(batch_files)} batch files, processing...")
            
            # Load all batches
            all_transitions = []
            for file in batch_files:
                batch = torch.load(file, map_location=device)
                all_transitions.extend(batch)
                os.remove(file)  # Remove processed batch
            
            # Update model
            if all_transitions:
                print(f"Training on {len(all_transitions)} transitions")
                # Convert to tensors and move to device
                obs, actions, rewards, next_obs, dones = zip(*all_transitions)
                obs = torch.FloatTensor(np.array(obs)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_obs = torch.FloatTensor(np.array(next_obs)).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Update model
                loss = model.update((obs, actions, rewards, next_obs, dones))
                print(f"Model updated. Loss: {loss:.4f}")
                
                # Save updated model
                torch.save(model.state_dict(), 
                         os.path.join(MODEL_DIR, "global_model.pth"))
        
        # Wait before checking for new data
        time.sleep(5)

def main():
    # Setup
    setup_directories()
    
    # Get hardware profile
    profile = get_hardware_profile()
    device = "cuda" if profile.role == NodeRole.GPU_LEARNER else "cpu"
    
    # Run appropriate logic based on role
    if profile.role == NodeRole.CPU_ACTOR:
        run_actor(device)
    elif profile.role == NodeRole.GPU_LEARNER:
        run_learner(device)
    else:
        raise ValueError(f"Unknown node role: {profile.role}")

if __name__ == "__main__":
    main()
