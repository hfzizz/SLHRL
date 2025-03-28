import os
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Optional
import csv
import json

def create_output_dir(base_dir: str = 'results', agent_name: str = 'agent') -> str:
    """
    Create a timestamped output directory for saving models and results.
    
    Args:
        base_dir: Base directory for results
        agent_name: Name of the agent
        
    Returns:
        Path to the created directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{agent_name}_{timestamp}")
    os.makedirs(output_dir)
    
    return output_dir

def save_model(model: torch.nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """
    Save a PyTorch model and optionally its optimizer state.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        optimizer: Optional optimizer to save state
    """
    state_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        state_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(state_dict, path)
    print(f"Model saved to {path}")

def load_model(model: torch.nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer]]:
    """
    Load a PyTorch model and optionally its optimizer state.
    
    Args:
        model: PyTorch model to load into
        path: Path to the saved model
        optimizer: Optional optimizer to load state into
        
    Returns:
    
        Tuple of (loaded_model, loaded_optimizer)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def save_metrics(metrics: Dict[str, List[float]], path: str) -> None:
    """
    Save training/evaluation metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics (keys are metric names, values are lists of values)
        path: Path to save the CSV file
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['episode'] + list(metrics.keys()))
        
        # Write data rows
        for i in range(len(next(iter(metrics.values())))):
            row = [i] + [metrics[key][i] for key in metrics.keys()]
            writer.writerow(row)
    
    print(f"Metrics saved to {path}")

def plot_learning_curve(metrics: Dict[str, List[float]], path: str, window_size: int = 10) -> None:
    """
    Plot learning curves and save the figure.
    
    Args:
        metrics: Dictionary of metrics (keys are metric names, values are lists of values)
        path: Path to save the plot
        window_size: Size of the moving average window
    """
    plt.figure(figsize=(12, 8))
    
    for i, (key, values) in enumerate(metrics.items()):
        episodes = list(range(len(values)))
        
        # Calculate moving average
        if len(values) >= window_size:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(episodes[window_size-1:], moving_avg, label=f"{key} (MA-{window_size})")
        
        # Also plot raw values with transparency
        plt.plot(episodes, values, alpha=0.3)
    
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(path)
    print(f"Learning curves plotted and saved to {path}")

def save_config(config: Dict, path: str) -> None:
    """
    Save configuration parameters to a JSON file.
    
    Args:
        config: Dictionary of configuration parameters
        path: Path to save the JSON file
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {path}")

def calculate_discounted_returns(rewards: List[float], dones: List[bool], gamma: float) -> np.ndarray:
    """
    Calculate discounted returns for a sequence of rewards.
    
    Args:
        rewards: List of rewards
        dones: List of done flags
        gamma: Discount factor
        
    Returns:
        Numpy array of discounted returns
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns

def compute_gae(rewards: List[float], values: List[float], next_values: List[float], 
                dones: List[bool], gamma: float, lam: float) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_values: List of next state value estimates
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Numpy array of advantages
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            last_advantage = 0.0
        
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    return advantages