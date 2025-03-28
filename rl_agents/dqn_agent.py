import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import gymnasium as gym
import os

from .utils import save_model, load_model, create_output_dir, save_metrics, plot_learning_curve, save_config

class QNetwork(nn.Module):
    """
    Q-Network for DQN agent.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize Q-Network.
        
        Args:
            input_dim: Dimension of input (observation space)
            output_dim: Dimension of output (action space)
            hidden_dim: Dimension of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(x)

class ReplayBuffer:
    """
    Replay buffer for DQN.
    """
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Current size of the buffer
        """
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network (DQN) agent.
    """
    def __init__(self, 
                 observation_space: gym.Space, 
                 action_space: gym.Space,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_target_every: int = 100,
                 hidden_dim: int = 128,
                 device: str = 'auto'):
        """
        Initialize DQN agent.
        
        Args:
            observation_space: Observation space of the environment
            action_space: Action space of the environment
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay factor
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            update_target_every: Number of steps between target network updates
            hidden_dim: Dimension of hidden layers in the Q-network
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Determine input and output dimensions
        if isinstance(observation_space, gym.spaces.Box):
            self.input_dim = np.prod(observation_space.shape)
        elif isinstance(observation_space, gym.spaces.Dict):
            # For Dict spaces, concatenate all arrays in the observation
            # This is just a simple approach - you might want something more sophisticated
            self.input_dim = sum(np.prod(space.shape) for space in observation_space.spaces.values())
        else:
            raise ValueError(f"Unsupported observation space: {type(observation_space)}")
        
        if isinstance(action_space, gym.spaces.Discrete):
            self.output_dim = action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = QNetwork(self.input_dim, self.output_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.input_dim, self.output_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Set hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        
        # Training stats
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        
        # Configuration for saving
        self.config = {
            'lr': lr,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'update_target_every': update_target_every,
            'hidden_dim': hidden_dim,
            'device': str(self.device),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
    
    def preprocess_observation(self, observation: Any) -> torch.Tensor:
        """
        Preprocess the observation to be fed into the network.
        
        Args:
            observation: Raw observation from the environment
            
        Returns:
            Preprocessed observation tensor
        """
        if isinstance(observation, dict):
            # For Dict observation spaces, flatten and concatenate all arrays
            processed = np.concatenate([arr.flatten() for arr in observation.values()])
        else:
            processed = observation.flatten()
        
        return torch.FloatTensor(processed).to(self.device)
    
    def select_action(self, observation: Any, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: select a random action
            return self.action_space.sample()
        else:
            # Exploitation: select the best action according to the policy
            with torch.no_grad():
                state = self.preprocess_observation(observation)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def update(self) -> float:
        """
        Update the policy network using a batch from the replay buffer.
        
        Returns:
            Loss value
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states and expected Q values
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Compute Huber loss
        loss = nn.SmoothL1Loss()(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, env: gym.Env, episodes: int, max_steps_per_episode: int = 1000, 
              log_every: int = 10, save_every: int = 100,
              output_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            env: Gym environment
            episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            log_every: Log stats every n episodes
            save_every: Save model every n episodes
            output_dir: Directory to save models and results
            
        Returns:
            Dictionary of training metrics
        """
        # Create output directory if not provided
        if output_dir is None:
            output_dir = create_output_dir(agent_name='dqn')
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save configuration
        save_config(self.config, os.path.join(output_dir, 'config.json'))
        
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': [],
            'epsilon': []
        }
        
        for episode in range(1, episodes + 1):
            observation, info = env.reset()
            total_reward = 0
            episode_loss = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(observation)
                
                # Take action in the environment
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Add transition to replay buffer
                self.buffer.add(observation, action, reward, next_observation, done)
                
                # Update policy
                loss = self.update()
                episode_loss += loss
                
                # Update target network periodically
                self.step_count += 1
                if self.step_count % self.update_target_every == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Move to the next state
                observation = next_observation
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            metrics['episode_losses'].append(episode_loss / steps if steps > 0 else 0)
            metrics['epsilon'].append(self.epsilon)
            
            # Log progress
            if episode % log_every == 0:
                avg_reward = np.mean(metrics['episode_rewards'][-log_every:])
                avg_length = np.mean(metrics['episode_lengths'][-log_every:])
                avg_loss = np.mean([l for l in metrics['episode_losses'][-log_every:] if l > 0])
                print(f"Episode {episode}/{episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
            
            # Save model periodically
            if episode % save_every == 0:
                model_path = os.path.join(output_dir, f'dqn_model_episode_{episode}.pt')
                save_model(self.policy_net, model_path, self.optimizer)
                
                # Save metrics
                metrics_path = os.path.join(output_dir, 'metrics.csv')
                save_metrics(metrics, metrics_path)
                
                # Plot learning curve
                plot_path = os.path.join(output_dir, 'learning_curve.png')
                plot_learning_curve(metrics, plot_path)
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'dqn_model_final.pt')
        save_model(self.policy_net, final_model_path, self.optimizer)
        
        # Save final metrics
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        save_metrics(metrics, metrics_path)
        
        # Plot final learning curve
        plot_path = os.path.join(output_dir, 'learning_curve.png')
        plot_learning_curve(metrics, plot_path)
        
        return metrics
    
    def evaluate(self, env: gym.Env, episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            env: Gym environment
            episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        lengths = []
        
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Select action (no exploration)
                action = self.select_action(observation, training=False)
                
                # Take action in the environment
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if render:
                    env.render()
                
                total_reward += reward
                steps += 1
            
            rewards.append(total_reward)
            lengths.append(steps)
            
            print(f"Evaluation Episode {episode+1}/{episodes} - "
                  f"Reward: {total_reward:.2f}, Length: {steps}")
        
        # Compute metrics
        avg_reward = np.mean(rewards)
        avg_length = np.mean(lengths)
        
        evaluation_metrics = {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        print(f"Evaluation Results - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        return evaluation_metrics
    
    def save(self, path: str) -> None:
        """
        Save the agent's model and optimizer state.
        
        Args:
            path: Path to save the model
        """
        save_model(self.policy_net, path, self.optimizer)
    
    def load(self, path: str) -> None:
        """
        Load the agent's model and optimizer state.
        
        Args:
            path: Path to the saved model
        """
        self.policy_net, self.optimizer = load_model(self.policy_net, path, self.optimizer)
        self.target_net.load_state_dict(self.policy_net.state_dict())