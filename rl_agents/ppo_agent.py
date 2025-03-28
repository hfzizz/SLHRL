import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict, Any, Optional
import os

from .utils import save_model, load_model, create_output_dir, save_metrics, plot_learning_curve, save_config, compute_gae

class ActorCritic(nn.Module):
    """
    Combined actor-critic network.
    """
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 256):
        """
        Initialize actor-critic network.
        
        Args:
            input_dim: Dimension of input (observation space)
            n_actions: Number of possible actions
            hidden_dim: Dimension of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        features = self.feature_extractor(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state: torch.Tensor, 
                             action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action distribution, value, action, and log probability.
        
        Args:
            state: State tensor
            action: Optional action tensor (for evaluating existing actions)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self(state)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """
    def __init__(self, 
                 observation_space: gym.Space, 
                 action_space: gym.Space,
                 lr: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 update_epochs: int = 4,
                 batch_size: int = 64,
                 hidden_dim: int = 256,
                 device: str = 'auto'):
        """
        Initialize PPO agent.
        
        Args:
            observation_space: Observation space of the environment
            action_space: Action space of the environment
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update policy per rollout
            batch_size: Batch size for training
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Determine input and output dimensions
        if isinstance(observation_space, gym.spaces.Box):
            self.input_dim = np.prod(observation_space.shape)
        elif isinstance(observation_space, gym.spaces.Dict):
            # For Dict spaces, concatenate all arrays in the observation
            self.input_dim = sum(np.prod(space.shape) for space in observation_space.spaces.values())
        else:
            raise ValueError(f"Unsupported observation space: {type(observation_space)}")
        
        if isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize actor-critic network
        self.network = ActorCritic(self.input_dim, self.n_actions, hidden_dim).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Set hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_bonuses = []
        
        # Configuration for saving
        self.config = {
            'lr': lr,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_ratio': clip_ratio,
            'value_coef': value_coef,
            'entropy_coef': entropy_coef,
            'max_grad_norm': max_grad_norm,
            'update_epochs': update_epochs,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim,
            'device': str(self.device),
            'input_dim': self.input_dim,
            'n_actions': self.n_actions
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
    
    def select_action(self, observation: Any, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action using the current policy.
        
        Args:
            observation: Observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state = self.preprocess_observation(observation)
        
        with torch.no_grad():
            if training:
                action, log_prob, _, value = self.network.get_action_and_value(state)
                return action.item(), log_prob.item(), value.item()
            else:
                logits, value = self.network(state)
                # In evaluation, take the most probable action
                action = torch.argmax(logits).item()
                return action, 0.0, value.item()
    
    def update(self, rollout: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """
        Update the policy and value function.
        
        Args:
            rollout: Dictionary containing collected trajectories
            
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        # Convert rollout data to tensors
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout['advantages']).to(self.device)
        
        # Normalize advantages (optional but often helps)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # Create dataloader for mini-batch updates
            dataset_size = len(states)
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current action distributions and values
                _, batch_log_probs, batch_entropy, batch_values = self.network.get_action_and_value(batch_states, batch_actions)
                batch_values = batch_values.squeeze(-1)
                
                # Compute ratio between old and new policy
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                
                # PPO policy loss
                policy_loss1 = -batch_advantages * ratio
                policy_loss2 = -batch_advantages * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(batch_values, batch_returns)
                
                # Entropy bonus
                entropy_bonus = batch_entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
        
        # Average losses over all updates
        n_updates = self.update_epochs * (len(states) // self.batch_size + 1)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def collect_rollout(self, env: gym.Env, rollout_steps: int) -> Dict[str, np.ndarray]:
        """
        Collect a rollout of trajectories.
        
        Args:
            env: Gym environment
            rollout_steps: Number of steps to collect
            
        Returns:
            Dictionary containing collected trajectories
        """
        # Initialize arrays
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Initialize environment
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        
        for step in range(rollout_steps):
            states.append(observation)
            
            # Select action
            action, log_prob, value = self.select_action(observation)
            
            # Take action in the environment
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            # Move to the next state
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                observation, _ = env.reset()
        
        # Get final value (bootstrap value for incomplete episode)
        _, _, next_value = self.select_action(observation)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        
        # Compute returns and advantages
        advantages = compute_gae(
            rewards=rewards,
            values=values,
            next_values=np.append(values[1:], next_value),
            dones=dones,
            gamma=self.gamma,
            lam=self.gae_lambda
        )
        returns = advantages + values
        
        # Update episode stats
        self.episode_rewards.extend(episode_rewards)
        self.episode_lengths.extend(episode_lengths)
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'returns': returns,
            'advantages': advantages
        }
    
    def train(self, env: gym.Env, total_steps: int, rollout_steps: int = 2048, 
              log_every: int = 1, save_every: int = 10,
              output_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the agent for a specified number of steps.
        
        Args:
            env: Gym environment
            total_steps: Total number of steps to train for
            rollout_steps: Number of steps per rollout
            log_every: Log metrics every N updates
            save_every: Save model every N updates
            output_dir: Directory to save model and metrics
            
        Returns:
            Dictionary containing training metrics
        """
        # Create output directory if needed
        if output_dir is not None:
            output_dir = create_output_dir(output_dir)
            save_config(self.config, os.path.join(output_dir, 'config.json'))
        
        # Initialize metrics tracking
        metrics = {
            'update': [],
            'step': [],
            'episode_reward_mean': [],
            'episode_length_mean': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        # Training loop
        total_updates = total_steps // rollout_steps
        total_episodes = 0
        steps_so_far = 0
        
        print(f"Starting training for {total_steps} steps ({total_updates} updates)...")
        
        for update in range(1, total_updates + 1):
            # Collect rollout
            rollout = self.collect_rollout(env, rollout_steps)
            steps_so_far += rollout_steps
            
            # Update policy and value function
            policy_loss, value_loss, entropy = self.update(rollout)
            
            # Track metrics
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.entropy_bonuses.append(entropy)
            
            # Calculate episode metrics since last update
            new_episodes = len(self.episode_rewards) - total_episodes
            if new_episodes > 0:
                recent_rewards = self.episode_rewards[-new_episodes:]
                recent_lengths = self.episode_lengths[-new_episodes:]
                episode_reward_mean = np.mean(recent_rewards)
                episode_length_mean = np.mean(recent_lengths)
                total_episodes = len(self.episode_rewards)
            else:
                # No new episodes completed in this update
                episode_reward_mean = 0.0 if not self.episode_rewards else self.episode_rewards[-1]
                episode_length_mean = 0.0 if not self.episode_lengths else self.episode_lengths[-1]
            
            # Update metrics
            metrics['update'].append(update)
            metrics['step'].append(steps_so_far)
            metrics['episode_reward_mean'].append(episode_reward_mean)
            metrics['episode_length_mean'].append(episode_length_mean)
            metrics['policy_loss'].append(policy_loss)
            metrics['value_loss'].append(value_loss)
            metrics['entropy'].append(entropy)
            
            # Log progress
            if update % log_every == 0:
                print(f"Update {update}/{total_updates} | Steps {steps_so_far}/{total_steps}")
                print(f"  Recent episodes: {new_episodes}")
                print(f"  Mean episode reward: {episode_reward_mean:.2f}")
                print(f"  Mean episode length: {episode_length_mean:.2f}")
                print(f"  Policy loss: {policy_loss:.4f}")
                print(f"  Value loss: {value_loss:.4f}")
                print(f"  Entropy: {entropy:.4f}")
                print("-" * 40)
            
            # Save model and metrics
            if output_dir is not None and update % save_every == 0:
                model_path = os.path.join(output_dir, f"model_step_{steps_so_far}.pt")
                save_model(self.network, model_path)
                
                metrics_path = os.path.join(output_dir, 'metrics.json')
                save_metrics(metrics, metrics_path)
                
                # Create learning curve plot
                plot_path = os.path.join(output_dir, 'learning_curve.png')
                plot_learning_curve(
                    metrics['step'], 
                    metrics['episode_reward_mean'],
                    title="PPO Learning Curve",
                    xlabel="Steps",
                    ylabel="Mean Episode Reward",
                    output_path=plot_path
                )
        
        # Save final model and metrics
        if output_dir is not None:
            final_model_path = os.path.join(output_dir, "model_final.pt")
            save_model(self.network, final_model_path)
            
            metrics_path = os.path.join(output_dir, 'metrics.json')
            save_metrics(metrics, metrics_path)
            
            # Create final learning curve plot
            plot_path = os.path.join(output_dir, 'learning_curve.png')
            plot_learning_curve(
                metrics['step'], 
                metrics['episode_reward_mean'],
                title="PPO Learning Curve",
                xlabel="Steps",
                ylabel="Mean Episode Reward",
                output_path=plot_path
            )
        
        print(f"Training completed after {total_updates} updates ({steps_so_far} steps)")
        print(f"Total episodes: {total_episodes}")
        print(f"Final mean episode reward: {metrics['episode_reward_mean'][-1]:.2f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        save_model(self.network, path)
        
        # Save config to a JSON file
        config_path = os.path.splitext(path)[0] + "_config.json"
        save_config(self.config, config_path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's model.
        
        Args:
            path: Path to the saved model
        """
        load_model(self.network, path, self.device)
        
        # Try to load config if it exists
        config_path = os.path.splitext(path)[0] + "_config.json"
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                print(f"Loaded config from {config_path}")
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def evaluate(self, env: gym.Env, num_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
        """
        Evaluate the agent on a given environment.
        
        Args:
            env: Gym environment
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Tuple of (mean reward, std reward)
        """
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            observation, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                if render:
                    env.render()
                
                # Select action (in evaluation mode)
                action, _, _ = self.select_action(observation, training=False)
                
                # Take action in the environment
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward}, Length = {episode_length}")
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(lengths)
        
        print(f"Evaluation over {num_episodes} episodes:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean episode length: {mean_length:.2f}")
        
        return mean_reward, std_reward