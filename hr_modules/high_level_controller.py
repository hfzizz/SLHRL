import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional, Union
import os

# Define strategic decision categories
STRATEGY_FARMING = 0
STRATEGY_QUESTING = 1
STRATEGY_TRAINING = 2
STRATEGY_GATE_PROGRESSION = 3

# Skill allocation categories
ALLOCATE_STRENGTH = 0
ALLOCATE_AGILITY = 1
ALLOCATE_INTELLIGENCE = 2
ALLOCATE_COMBAT_SKILL = 3  # Generic combat skill upgrade

# Engagement decisions
ENGAGE_BOSS = 0
FOCUS_DAILY_QUESTS = 1
FOCUS_PENALTY_QUESTS = 2
RETREAT = 3

class HighLevelNetwork(nn.Module):
    """
    Neural network for the high-level controller (Manager) in hierarchical reinforcement learning.
    This network makes strategic decisions about farming, questing, training, 
    gate progression, and skill point allocation.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize the high-level network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dim: Dimension of hidden layers
        """
        super(HighLevelNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Strategy decision head (farming, questing, training, gate progression)
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 strategy options
        )
        
        # Skill allocation head
        self.skill_allocation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 allocation options
        )
        
        # Combat engagement head
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 engagement options
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (strategy_logits, skill_allocation_logits, engagement_logits)
        """
        features = self.backbone(state)
        
        strategy_logits = self.strategy_head(features)
        skill_allocation_logits = self.skill_allocation_head(features)
        engagement_logits = self.engagement_head(features)
        
        return strategy_logits, skill_allocation_logits, engagement_logits

class HighLevelController:
    """
    High-level controller (Manager) for hierarchical reinforcement learning.
    
    This controller makes strategic decisions about:
    1. Overall strategy (farming, questing, training, gate progression)
    2. Skill point allocation (strength, agility, intelligence, combat skills)
    3. Combat engagement (boss fights, daily quests, penalty quests, retreat)
    """
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 0.0003,
                 gamma: float = 0.99,
                 decision_interval: int = 20,  # Make strategic decisions every N steps
                 device: str = 'auto'):
        """
        Initialize the high-level controller.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dim: Dimension of hidden layers
            lr: Learning rate
            gamma: Discount factor
            decision_interval: Steps between high-level decisions
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.decision_interval = decision_interval
        
        # Initialize network
        self.network = HighLevelNetwork(state_dim, hidden_dim).to(self.device)
        self.target_network = HighLevelNetwork(state_dim, hidden_dim).to(self.device)
        
        # Copy weights from network to target network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # For storing experience
        self.buffer = {
            'states': [],
            'strategies': [],
            'skill_allocations': [],
            'engagements': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        # Current decisions and their lifetimes
        self.current_strategy = None
        self.current_skill_allocation = None
        self.current_engagement = None
        self.steps_since_decision = 0
        
        # Training stats
        self.strategy_loss_history = []
        self.allocation_loss_history = []
        self.engagement_loss_history = []
        self.reward_history = []
        
        # Probability distributions for exploration
        self.strategy_probs = np.ones(4) / 4  # Equal probability for all strategies initially
        self.allocation_probs = np.ones(4) / 4  # Equal probability for all allocations initially
        self.engagement_probs = np.ones(4) / 4  # Equal probability for all engagements initially
    
    def preprocess_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the state dictionary into a tensor.
        
        Args:
            state: State dictionary with various agent and environment information
            
        Returns:
            Processed state tensor
        """
        # Extract relevant information from state
        # This should be adapted based on your actual state representation
        agent_stats = state.get('agent_stats', np.zeros(10))
        env_info = state.get('environment_info', np.zeros(5))
        quest_status = state.get('quest_status', np.zeros(3))
        
        # Concatenate all state components
        combined_state = np.concatenate([agent_stats, env_info, quest_status])
        
        # Convert to tensor
        return torch.FloatTensor(combined_state).to(self.device)
    
    def make_decisions(self, state: Dict[str, Any], training: bool = True, 
                      epsilon: float = 0.1) -> Tuple[int, int, int]:
        """
        Make high-level strategic decisions.
        
        Args:
            state: Current state dictionary
            training: Whether the controller is in training mode
            epsilon: Exploration rate
            
        Returns:
            Tuple of (strategy, skill_allocation, engagement) decisions
        """
        # Check if it's time to make new decisions
        if (self.current_strategy is not None and 
            self.steps_since_decision < self.decision_interval):
            self.steps_since_decision += 1
            return self.current_strategy, self.current_skill_allocation, self.current_engagement
        
        # Reset decision timer
        self.steps_since_decision = 0
        
        # Preprocess state
        state_tensor = self.preprocess_state(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Get action logits
            strategy_logits, skill_allocation_logits, engagement_logits = self.network(state_tensor)
            
            if training and np.random.random() < epsilon:
                # Exploration: sample from current probability distributions
                strategy = np.random.choice(4, p=self.strategy_probs)
                skill_allocation = np.random.choice(4, p=self.allocation_probs)
                engagement = np.random.choice(4, p=self.engagement_probs)
            else:
                # Exploitation: choose highest logit
                strategy = torch.argmax(strategy_logits, dim=1).item()
                skill_allocation = torch.argmax(skill_allocation_logits, dim=1).item()
                engagement = torch.argmax(engagement_logits, dim=1).item()
        
        # Store current decisions
        self.current_strategy = strategy
        self.current_skill_allocation = skill_allocation
        self.current_engagement = engagement
        
        return strategy, skill_allocation, engagement
    
    def interpret_decisions(self, strategy: int, skill_allocation: int, 
                           engagement: int) -> Dict[str, Any]:
        """
        Interpret the numerical decisions into actionable directives.
        
        Args:
            strategy: Strategy decision index
            skill_allocation: Skill allocation decision index
            engagement: Engagement decision index
            
        Returns:
            Dictionary of interpreted decisions
        """
        # Strategy interpretation
        strategy_map = {
            STRATEGY_FARMING: "focus_on_farming",
            STRATEGY_QUESTING: "focus_on_questing",
            STRATEGY_TRAINING: "focus_on_training",
            STRATEGY_GATE_PROGRESSION: "focus_on_gate_progression"
        }
        
        # Skill allocation interpretation
        allocation_map = {
            ALLOCATE_STRENGTH: "allocate_to_strength",
            ALLOCATE_AGILITY: "allocate_to_agility",
            ALLOCATE_INTELLIGENCE: "allocate_to_intelligence",
            ALLOCATE_COMBAT_SKILL: "allocate_to_combat_skills"
        }
        
        # Engagement interpretation
        engagement_map = {
            ENGAGE_BOSS: "engage_boss_fight",
            FOCUS_DAILY_QUESTS: "focus_on_daily_quests",
            FOCUS_PENALTY_QUESTS: "focus_on_penalty_quests",
            RETREAT: "retreat_and_recover"
        }
        
        return {
            "strategy": strategy_map.get(strategy, "unknown_strategy"),
            "allocation": allocation_map.get(skill_allocation, "unknown_allocation"),
            "engagement": engagement_map.get(engagement, "unknown_engagement")
        }
    
    def translate_to_subgoals(self, decisions: Dict[str, Any], 
                             agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate high-level decisions into specific subgoals for the low-level controller.
        
        Args:
            decisions: Interpreted decisions
            agent_state: Current agent state
            
        Returns:
            Dictionary of subgoals for the low-level controller
        """
        subgoals = {}
        
        # Strategy subgoals
        if decisions["strategy"] == "focus_on_farming":
            subgoals["target_selection"] = "weak_monsters"
            subgoals["resource_priority"] = "gold"
        elif decisions["strategy"] == "focus_on_questing":
            subgoals["target_selection"] = "quest_targets"
            subgoals["resource_priority"] = "experience"
        elif decisions["strategy"] == "focus_on_training":
            subgoals["target_selection"] = "training_dummies"
            subgoals["resource_priority"] = "skill_experience"
        elif decisions["strategy"] == "focus_on_gate_progression":
            subgoals["target_selection"] = "gate_guardians"
            subgoals["resource_priority"] = "advancement"
        
        # Allocation subgoals
        if decisions["allocation"] == "allocate_to_strength":
            subgoals["skill_points_target"] = "strength"
        elif decisions["allocation"] == "allocate_to_agility":
            subgoals["skill_points_target"] = "agility"
        elif decisions["allocation"] == "allocate_to_intelligence":
            subgoals["skill_points_target"] = "intelligence"
        elif decisions["allocation"] == "allocate_to_combat_skills":
            # Choose a specific combat skill based on current skill levels
            skills = agent_state.get("skills", {})
            unlocked_combat_skills = [skill for skill, data in skills.items() 
                                     if data.get("unlocked", False) and data.get("category") == "combat"]
            
            if unlocked_combat_skills:
                # Prioritize skills that can be evolved
                evolvable_skills = [skill for skill in unlocked_combat_skills 
                                   if "evolved_form" in skills[skill]]
                
                if evolvable_skills:
                    # Find the closest skill to evolution
                    closest_to_evolution = None
                    min_levels_needed = float('inf')
                    
                    for skill in evolvable_skills:
                        skill_data = skills[skill]
                        current_level = skill_data.get("level", 0)
                        evolution_level = skill_data.get("evolution_level", 999)
                        levels_needed = evolution_level - current_level
                        
                        if levels_needed < min_levels_needed and levels_needed > 0:
                            min_levels_needed = levels_needed
                            closest_to_evolution = skill
                    
                    if closest_to_evolution:
                        subgoals["skill_points_target"] = closest_to_evolution
                    else:
                        # All skills are already at evolution level, pick one randomly
                        subgoals["skill_points_target"] = np.random.choice(unlocked_combat_skills)
                else:
                    # No evolvable skills, pick one randomly
                    subgoals["skill_points_target"] = np.random.choice(unlocked_combat_skills)
            else:
                # No combat skills unlocked yet, focus on unlocking one
                subgoals["skill_points_target"] = "unlock_combat_skill"
        
        # Engagement subgoals
        if decisions["engagement"] == "engage_boss_fight":
            subgoals["combat_approach"] = "aggressive"
            subgoals["skill_usage"] = "prioritize_damage"
        elif decisions["engagement"] == "focus_on_daily_quests":
            subgoals["combat_approach"] = "balanced"
            subgoals["skill_usage"] = "prioritize_efficiency"
        elif decisions["engagement"] == "focus_on_penalty_quests":
            subgoals["combat_approach"] = "cautious"
            subgoals["skill_usage"] = "prioritize_survival"
        elif decisions["engagement"] == "retreat_and_recover":
            subgoals["combat_approach"] = "evasive"
            subgoals["skill_usage"] = "prioritize_escape"
        
        return subgoals
    
    def store_transition(self, state: Dict[str, Any], strategy: int, skill_allocation: int,
                         engagement: int, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Store a transition in the experience buffer.
        
        Args:
            state: Current state
            strategy: Strategy decision
            skill_allocation: Skill allocation decision
            engagement: Engagement decision
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert states to tensor representation
        state_tensor = self.preprocess_state(state).cpu().numpy()
        next_state_tensor = self.preprocess_state(next_state).cpu().numpy()
        
        self.buffer['states'].append(state_tensor)
        self.buffer['strategies'].append(strategy)
        self.buffer['skill_allocations'].append(skill_allocation)
        self.buffer['engagements'].append(engagement)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state_tensor)
        self.buffer['dones'].append(done)
        
        # Update reward history
        self.reward_history.append(reward)
        
        # Limit buffer size
        max_buffer_size = 100000
        if len(self.buffer['states']) > max_buffer_size:
            for key in self.buffer:
                self.buffer[key] = self.buffer[key][-max_buffer_size:]
    
    def update(self, batch_size: int = 64) -> Tuple[float, float, float]:
        """
        Update the high-level policy.
        
        Args:
            batch_size: Number of samples to use for update
            
        Returns:
            Tuple of (strategy_loss, allocation_loss, engagement_loss)
        """
        # Check if we have enough samples
        if len(self.buffer['states']) < batch_size:
            return 0.0, 0.0, 0.0
        
        # Sample random batch
        indices = np.random.choice(len(self.buffer['states']), batch_size, replace=False)
        
        states = torch.FloatTensor([self.buffer['states'][i] for i in indices]).to(self.device)
        strategies = torch.LongTensor([self.buffer['strategies'][i] for i in indices]).to(self.device)
        skill_allocations = torch.LongTensor([self.buffer['skill_allocations'][i] for i in indices]).to(self.device)
        engagements = torch.LongTensor([self.buffer['engagements'][i] for i in indices]).to(self.device)
        rewards = torch.FloatTensor([self.buffer['rewards'][i] for i in indices]).to(self.device)
        next_states = torch.FloatTensor([self.buffer['next_states'][i] for i in indices]).to(self.device)
        dones = torch.FloatTensor([self.buffer['dones'][i] for i in indices]).to(self.device)
        
        # Get current Q-values
        strategy_logits, skill_allocation_logits, engagement_logits = self.network(states)
        
        # Convert logits to Q-values for selected actions
        strategy_values = strategy_logits.gather(1, strategies.unsqueeze(1)).squeeze(1)
        allocation_values = skill_allocation_logits.gather(1, skill_allocations.unsqueeze(1)).squeeze(1)
        engagement_values = engagement_logits.gather(1, engagements.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_strategy_logits, next_allocation_logits, next_engagement_logits = self.target_network(next_states)
            
            next_strategy_values = next_strategy_logits.max(1)[0]
            next_allocation_values = next_allocation_logits.max(1)[0]
            next_engagement_values = next_engagement_logits.max(1)[0]
            
            # Terminal states have zero future reward
            next_strategy_values = next_strategy_values * (1 - dones)
            next_allocation_values = next_allocation_values * (1 - dones)
            next_engagement_values = next_engagement_values * (1 - dones)
            
            target_strategy_values = rewards + self.gamma * next_strategy_values
            target_allocation_values = rewards + self.gamma * next_allocation_values
            target_engagement_values = rewards + self.gamma * next_engagement_values
        
        # Compute losses
        strategy_loss = F.mse_loss(strategy_values, target_strategy_values)
        allocation_loss = F.mse_loss(allocation_values, target_allocation_values)
        engagement_loss = F.mse_loss(engagement_values, target_engagement_values)
        
        # Total loss
        total_loss = strategy_loss + allocation_loss + engagement_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store losses
        self.strategy_loss_history.append(strategy_loss.item())
        self.allocation_loss_history.append(allocation_loss.item())
        self.engagement_loss_history.append(engagement_loss.item())
        
        # Update probability distributions based on Q-values (softmax)
        with torch.no_grad():
            strategy_probs = F.softmax(strategy_logits.mean(0), dim=0).cpu().numpy()
            allocation_probs = F.softmax(skill_allocation_logits.mean(0), dim=0).cpu().numpy()
            engagement_probs = F.softmax(engagement_logits.mean(0), dim=0).cpu().numpy()
            
            # Smooth the probabilities to ensure some exploration
            self.strategy_probs = 0.9 * strategy_probs + 0.1 * (np.ones(4) / 4)
            self.allocation_probs = 0.9 * allocation_probs + 0.1 * (np.ones(4) / 4)
            self.engagement_probs = 0.9 * engagement_probs + 0.1 * (np.ones(4) / 4)
            
            # Normalize
            self.strategy_probs /= self.strategy_probs.sum()
            self.allocation_probs /= self.allocation_probs.sum()
            self.engagement_probs /= self.engagement_probs.sum()
        
        return strategy_loss.item(), allocation_loss.item(), engagement_loss.item()
    
    def update_target_network(self, tau: float = 0.01):
        """
        Update the target network with soft update.
        
        Args:
            tau: Interpolation parameter
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset(self):
        """Reset controller state at the end of an episode."""
        self.current_strategy = None
        self.current_skill_allocation = None
        self.current_engagement = None
        self.steps_since_decision = 0
    
    def save(self, path: str):
        """
        Save the high-level controller model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'strategy_loss_history': self.strategy_loss_history,
            'allocation_loss_history': self.allocation_loss_history,
            'engagement_loss_history': self.engagement_loss_history,
            'reward_history': self.reward_history,
            'strategy_probs': self.strategy_probs,
            'allocation_probs': self.allocation_probs,
            'engagement_probs': self.engagement_probs
        }, path)
    
    def load(self, path: str):
        """
        Load the high-level controller model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.strategy_loss_history = checkpoint['strategy_loss_history']
        self.allocation_loss_history = checkpoint['allocation_loss_history']
        self.engagement_loss_history = checkpoint['engagement_loss_history']
        self.reward_history = checkpoint['reward_history']
        self.strategy_probs = checkpoint['strategy_probs']
        self.allocation_probs = checkpoint['allocation_probs']
        self.engagement_probs = checkpoint['engagement_probs']