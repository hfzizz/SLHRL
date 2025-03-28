import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os

# Define action constants for the low level agent (must match agent_stats.py)
ACTION_BASIC_ATTACK = 0
ACTION_MOVE = 1
ACTION_CRITICAL_ATTACK = 2  # Changed to match agent_stats.py
ACTION_QUICK_DODGE = 3      # Added from agent_stats.py
ACTION_SPRINT = 4
ACTION_STEALTH = 5
ACTION_BLOODLUST = 6
ACTION_DAGGER_THROW = 7     # Added from agent_stats.py
ACTION_USE_HEALTH_POTION = 8
ACTION_USE_MANA_POTION = 9
ACTION_RETREAT = 10
ACTION_REST = 11
ACTION_SHADOW_EXTRACTION = 12  # Added job skill

class LowLevelNetwork(nn.Module):
    """
    Neural network for the low-level controller (Worker) in hierarchical reinforcement learning.
    This network makes tactical decisions like skill usage, movement, and combat actions
    based on high-level directives.
    """
    def __init__(self, state_dim: int, subgoal_dim: int, n_actions: int, hidden_dim: int = 256):
        """
        Initialize the low-level network.
        
        Args:
            state_dim: Dimension of the state space
            subgoal_dim: Dimension of the subgoal space from high-level controller
            n_actions: Number of available actions
            hidden_dim: Dimension of hidden layers
        """
        super(LowLevelNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        self.n_actions = n_actions
        
        # Input processing for state and subgoal
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.subgoal_encoder = nn.Sequential(
            nn.Linear(subgoal_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined representation
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        
        # Action head (Q-values for each action)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Value head (state value)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            subgoal: Subgoal tensor from high-level controller
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        state_features = self.state_encoder(state)
        subgoal_features = self.subgoal_encoder(subgoal)
        
        # Concatenate state and subgoal features
        combined = torch.cat([state_features, subgoal_features], dim=-1)
        features = self.combined_layer(combined)
        
        action_logits = self.action_head(features)
        state_value = self.value_head(features)
        
        return action_logits, state_value


class SkillManager:
    """
    Manages skill information, cooldowns, prerequisites, and effects.
    Used by the low-level controller to make informed decisions about skill usage.
    """
    def __init__(self):
        """Initialize the skill manager with default skill definitions."""
        self.skills = {
            "basic_attack": {
                "id": ACTION_BASIC_ATTACK,
                "cooldown": 0,
                "mana_cost": 0,
                "damage_mult": 1.0,
                "requires_target": True
            },
            "critical_attack": {  # Changed from "critical_strike" to match agent_stats.py
                "id": ACTION_CRITICAL_ATTACK,
                "cooldown": 3,
                "mana_cost": 10,
                "damage_mult": 2.5,
                "requires_target": True,
                "effect": "high_damage"
            },
            "quick_dodge": {  # Changed from "shield_block" to match agent_stats.py
                "id": ACTION_QUICK_DODGE,
                "cooldown": 5,
                "mana_cost": 15,
                "damage_mult": 0.0,
                "requires_target": False,
                "effect": "damage_reduction"
            },
            "sprint": {
                "id": ACTION_SPRINT,
                "cooldown": 8,
                "mana_cost": 20,
                "damage_mult": 0.0,
                "requires_target": False,
                "effect": "mobility"
            },
            "stealth": {
                "id": ACTION_STEALTH,
                "cooldown": 12,
                "mana_cost": 30,
                "damage_mult": 0.0,
                "requires_target": False,
                "effect": "invisibility"
            },
            "bloodlust": {
                "id": ACTION_BLOODLUST,
                "cooldown": 15,
                "mana_cost": 40,
                "damage_mult": 1.5,
                "requires_target": False,
                "effect": "strength_boost"
            },
            "dagger_throw": {  # Changed from "lightning_strike" to match agent_stats.py
                "id": ACTION_DAGGER_THROW,
                "cooldown": 10,
                "mana_cost": 35,
                "damage_mult": 3.0,
                "requires_target": True,
                "effect": "aoe_damage"
            },
            "shadow_extraction": {  # Added new job skill
                "id": ACTION_SHADOW_EXTRACTION,
                "cooldown": 20,
                "mana_cost": 50,
                "damage_mult": 2.0,
                "requires_target": True,
                "effect": "special_damage"
            }
        }
        
        self.potions = {
            "health_potion": {
                "id": ACTION_USE_HEALTH_POTION,
                "effect": "restore_health",
                "value": 50
            },
            "mana_potion": {
                "id": ACTION_USE_MANA_POTION,
                "effect": "restore_mana",
                "value": 50
            }
        }

    def get_skill_by_id(self, skill_id: int) -> Dict[str, Any]:
        """
        Get skill information by ID.

        Args:
            skill_id: ID of the skill
            
        Returns:
            Skill information dictionary
        """
        for skill_name, skill_info in self.skills.items():
            if skill_info["id"] == skill_id:
                return {**skill_info, "name": skill_name}

        for potion_name, potion_info in self.potions.items():
            if potion_info["id"] == skill_id:
                return {**potion_info, "name": potion_name}

        return None

    def get_skill_id(self, skill_name: str) -> int:
        """
        Get skill ID by name.

        Args:
            skill_name: Name of the skill
            
        Returns:
            Skill ID
        """
        if skill_name in self.skills:
            return self.skills[skill_name]["id"]
        elif skill_name in self.potions:
            return self.potions[skill_name]["id"]
        else:
            return -1

    def is_skill_available(self, skill_name: str, agent_state: Dict[str, Any]) -> bool:
        """
        Check if a skill is available to use.

        Args:
            skill_name: Name of the skill
            agent_state: Current agent state
            
        Returns:
            True if the skill is available
        """
        if skill_name not in self.skills:
            return False

        skill = self.skills[skill_name]

        # Check if skill is unlocked
        if not agent_state.get("skills", {}).get(skill_name, {}).get("unlocked", False):
            return False

        # Check cooldown
        if agent_state.get("skills", {}).get(skill_name, {}).get("current_cooldown", 0) > 0:
            return False

        # Check mana cost
        if agent_state.get("mana", 0) < skill.get("mana_cost", 0):
            return False

        # Check if target is required and available
        if skill.get("requires_target", False) and not agent_state.get("has_target", False):
            return False

        return True

    def get_appropriate_skills(self, agent_state: Dict[str, Any], 
                            combat_situation: str) -> List[int]:
        """
        Get a list of appropriate skills for the current situation.

        Args:
            agent_state: Current agent state
            combat_situation: Current combat situation (normal, boss, low_health, etc.)
            
        Returns:
            List of appropriate skill IDs
        """
        appropriate_skills = []

        # Basic attack is always available
        appropriate_skills.append(ACTION_BASIC_ATTACK)

        # Check each skill for availability and appropriateness
        for skill_name, skill_info in self.skills.items():
            if not self.is_skill_available(skill_name, agent_state):
                continue
            
            # Assess situation for skill appropriateness
            if combat_situation == "boss":
                # Boss fights: prioritize high damage and survival skills
                if skill_info.get("effect") in ["high_damage", "damage_reduction", "strength_boost"]:
                    appropriate_skills.append(skill_info["id"])
            
            elif combat_situation == "low_health":
                # Low health: prioritize defensive and mobility skills
                if skill_info.get("effect") in ["damage_reduction", "mobility", "invisibility"]:
                    appropriate_skills.append(skill_info["id"])
            
            elif combat_situation == "multiple_enemies":
                # Multiple enemies: prioritize AoE and mobility skills
                if skill_info.get("effect") in ["aoe_damage", "mobility"]:
                    appropriate_skills.append(skill_info["id"])
            
            elif combat_situation == "normal":
                # Normal combat: use efficient damage skills
                appropriate_skills.append(skill_info["id"])

        # Add potions if needed
        if agent_state.get("health", 100) / agent_state.get("max_health", 100) < 0.5:
            appropriate_skills.append(ACTION_USE_HEALTH_POTION)

        if agent_state.get("mana", 100) / agent_state.get("max_mana", 100) < 0.3:
            appropriate_skills.append(ACTION_USE_MANA_POTION)

        return appropriate_skills

class LowLevelController:
    """
    Low-level controller (Worker) for hierarchical reinforcement learning.
    
    This controller handles tactical decisions like:
    1. Combat skill selection and execution
    2. Movement and positioning
    3. Resource management (health, mana)
    4. Responding to immediate threats
    """
    def __init__(self, 
                 state_dim: int,
                 subgoal_dim: int,
                 n_actions: int = 12,  # Default action space size
                 hidden_dim: int = 256,
                 lr: float = 0.0003,
                 gamma: float = 0.99,
                 device: str = 'auto'):
        """
        Initialize the low-level controller.
        
        Args:
            state_dim: Dimension of the state space
            subgoal_dim: Dimension of the subgoal space
            n_actions: Number of available actions
            hidden_dim: Dimension of hidden layers
            lr: Learning rate
            gamma: Discount factor
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialize network
        self.network = LowLevelNetwork(state_dim, subgoal_dim, n_actions, hidden_dim).to(self.device)
        self.target_network = LowLevelNetwork(state_dim, subgoal_dim, n_actions, hidden_dim).to(self.device)
        
        # Copy weights from network to target network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize skill manager
        self.skill_manager = SkillManager()
        
        # For storing experience
        self.buffer = {
            'states': [],
            'subgoals': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'next_subgoals': [],
            'dones': []
        }
        
        # Current state
        self.current_state = None
        self.current_subgoal = None
        
        # Training stats
        self.q_value_history = []
        self.loss_history = []
        self.reward_history = []
        self.action_distribution = [0] * n_actions
    
    def preprocess_state_dict(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the state dictionary into a tensor.
        
        Args:
            state_dict: State dictionary with various agent and environment information
            
        Returns:
            Processed state tensor
        """
        # Extract relevant information from state
        agent_stats = state_dict.get('agent_stats', np.zeros(10))
        combat_state = state_dict.get('combat_state', np.zeros(5))
        target_info = state_dict.get('target_info', np.zeros(5))
        inventory = state_dict.get('inventory', np.zeros(3))
        
        # Concatenate all state components
        combined_state = np.concatenate([agent_stats, combat_state, target_info, inventory])
        
        # Convert to tensor
        return torch.FloatTensor(combined_state).to(self.device)
    
    def preprocess_subgoal_dict(self, subgoal_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the subgoal dictionary into a tensor.
        
        Args:
            subgoal_dict: Subgoal dictionary from high-level controller
            
        Returns:
            Processed subgoal tensor
        """
        # Convert subgoal directives to numerical representation
        # This implementation will depend on your specific subgoal structure
        
        # Example: encode target_selection directive
        target_selection = subgoal_dict.get('target_selection', 'normal')
        target_encoding = np.zeros(4)
        if target_selection == 'weak_monsters':
            target_encoding[0] = 1.0
        elif target_selection == 'quest_targets':
            target_encoding[1] = 1.0
        elif target_selection == 'training_dummies':
            target_encoding[2] = 1.0
        elif target_selection == 'gate_guardians':
            target_encoding[3] = 1.0
        
        # Example: encode resource_priority directive
        resource_priority = subgoal_dict.get('resource_priority', 'balanced')
        resource_encoding = np.zeros(4)
        if resource_priority == 'gold':
            resource_encoding[0] = 1.0
        elif resource_priority == 'experience':
            resource_encoding[1] = 1.0
        elif resource_priority == 'skill_experience':
            resource_encoding[2] = 1.0
        elif resource_priority == 'advancement':
            resource_encoding[3] = 1.0
        
        # Example: encode combat_approach directive
        combat_approach = subgoal_dict.get('combat_approach', 'balanced')
        approach_encoding = np.zeros(4)
        if combat_approach == 'aggressive':
            approach_encoding[0] = 1.0
        elif combat_approach == 'balanced':
            approach_encoding[1] = 1.0
        elif combat_approach == 'cautious':
            approach_encoding[2] = 1.0
        elif combat_approach == 'evasive':
            approach_encoding[3] = 1.0
        
        # Example: encode skill_usage directive
        skill_usage = subgoal_dict.get('skill_usage', 'balanced')
        skill_encoding = np.zeros(4)
        if skill_usage == 'prioritize_damage':
            skill_encoding[0] = 1.0
        elif skill_usage == 'prioritize_efficiency':
            skill_encoding[1] = 1.0
        elif skill_usage == 'prioritize_survival':
            skill_encoding[2] = 1.0
        elif skill_usage == 'prioritize_escape':
            skill_encoding[3] = 1.0
        
        # Concatenate all encodings
        subgoal_vector = np.concatenate([target_encoding, resource_encoding, 
                                        approach_encoding, skill_encoding])
        
        # Convert to tensor
        return torch.FloatTensor(subgoal_vector).to(self.device)
    
    def select_action(self, state_dict: Dict[str, Any], subgoal_dict: Dict[str, Any], 
                     epsilon: float = 0.1, training: bool = True) -> int:
        """
        Select an action based on current state and subgoal.
        
        Args:
            state_dict: Current state dictionary
            subgoal_dict: Current subgoal dictionary
            epsilon: Exploration rate
            training: Whether the controller is in training mode
            
        Returns:
            Selected action ID
        """
        # Preprocess state and subgoal
        state = self.preprocess_state_dict(state_dict)
        subgoal = self.preprocess_subgoal_dict(subgoal_dict)
        
        # Save current state and subgoal
        self.current_state = state
        self.current_subgoal = subgoal
        
        # Exploration: random action with probability epsilon
        if training and np.random.random() < epsilon:
            # Get appropriate skills for the current situation
            combat_situation = self._assess_combat_situation(state_dict)
            appropriate_actions = self.skill_manager.get_appropriate_skills(
                state_dict, combat_situation)
            
            # If no appropriate actions, use all actions
            if not appropriate_actions:
                appropriate_actions = list(range(self.n_actions))
            
            # Select a random action from appropriate actions
            action = np.random.choice(appropriate_actions)
        else:
            # Exploitation: use the network
            with torch.no_grad():
                state_batch = state.unsqueeze(0)  # Add batch dimension
                subgoal_batch = subgoal.unsqueeze(0)  # Add batch dimension
                
                # Get Q-values
                q_values, _ = self.network(state_batch, subgoal_batch)
                
                if training:
                    # During training, sometimes select from top-k actions
                    if np.random.random() < 0.3:  # 30% chance to use top-k
                        k = min(3, self.n_actions)  # Use top-3 actions
                        topk_actions = torch.topk(q_values, k=k, dim=1)[1][0]
                        action = topk_actions[np.random.randint(0, k)].item()
                    else:
                        action = torch.argmax(q_values, dim=1).item()
                else:
                    # During evaluation, always use the best action
                    action = torch.argmax(q_values, dim=1).item()
        
        # Update action distribution
        self.action_distribution[action] += 1
        
        return action
    
    def _assess_combat_situation(self, state_dict: Dict[str, Any]) -> str:
        """
        Assess the current combat situation to help with skill selection.
        
        Args:
            state_dict: Current state dictionary
            
        Returns:
            Combat situation string
        """
        # Check health level
        health_ratio = state_dict.get('health', 100) / state_dict.get('max_health', 100)
        if health_ratio < 0.3:
            return "low_health"
        
        # Check if fighting a boss
        if state_dict.get('is_boss', False):
            return "boss"
        
        # Check if fighting multiple enemies
        if state_dict.get('num_enemies', 1) > 1:
            return "multiple_enemies"
        
        # Default situation
        return "normal"
    
    def store_transition(self, state_dict: Dict[str, Any], subgoal_dict: Dict[str, Any],
                         action: int, reward: float, next_state_dict: Dict[str, Any],
                         next_subgoal_dict: Dict[str, Any], done: bool):
        """
        Store a transition in the experience buffer.
        
        Args:
            state_dict: Current state dictionary
            subgoal_dict: Current subgoal dictionary
            action: Action taken
            reward: Reward received
            next_state_dict: Next state dictionary
            next_subgoal_dict: Next subgoal dictionary
            done: Whether the episode is done
        """
        # Preprocess states and subgoals
        state = self.preprocess_state_dict(state_dict).cpu().numpy()
        subgoal = self.preprocess_subgoal_dict(subgoal_dict).cpu().numpy()
        next_state = self.preprocess_state_dict(next_state_dict).cpu().numpy()
        next_subgoal = self.preprocess_subgoal_dict(next_subgoal_dict).cpu().numpy()
        
        # Store in buffer
        self.buffer['states'].append(state)
        self.buffer['subgoals'].append(subgoal)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['next_subgoals'].append(next_subgoal)
        self.buffer['dones'].append(done)
        
        # Update reward history
        self.reward_history.append(reward)
        
        # Limit buffer size
        max_buffer_size = 100000
        if len(self.buffer['states']) > max_buffer_size:
            for key in self.buffer:
                self.buffer[key] = self.buffer[key][-max_buffer_size:]
    
    def update(self, batch_size: int = 64) -> float:
        """
        Update the low-level policy using DQN learning.
        
        Args:
            batch_size: Number of samples to use for update
            
        Returns:
            Loss value
        """
        # Check if we have enough samples
        if len(self.buffer['states']) < batch_size:
            return 0.0
        
        # Sample random batch
        indices = np.random.choice(len(self.buffer['states']), batch_size, replace=False)
        
        states = torch.FloatTensor([self.buffer['states'][i] for i in indices]).to(self.device)
        subgoals = torch.FloatTensor([self.buffer['subgoals'][i] for i in indices]).to(self.device)
        actions = torch.LongTensor([self.buffer['actions'][i] for i in indices]).to(self.device)
        rewards = torch.FloatTensor([self.buffer['rewards'][i] for i in indices]).to(self.device)
        next_states = torch.FloatTensor([self.buffer['next_states'][i] for i in indices]).to(self.device)
        next_subgoals = torch.FloatTensor([self.buffer['next_subgoals'][i] for i in indices]).to(self.device)
        dones = torch.FloatTensor([self.buffer['dones'][i] for i in indices]).to(self.device)
        
        # Compute current Q values
        current_q_values, _ = self.network(states, subgoals)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            # Double DQN: find best actions using online network
            online_q_values, _ = self.network(next_states, next_subgoals)
            best_actions = torch.argmax(online_q_values, dim=1)
            
            # Evaluate those actions using target network
            next_q_values, _ = self.target_network(next_states, next_subgoals)
            next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Target Q = reward + gamma * next_Q (if not done)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store loss and Q-values
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q_values.mean().item())
        
        return loss.item()
    
    def update_target_network(self, tau: float = 0.01):
        """
        Soft update of target network.
        
        Args:
            tau: Interpolation parameter
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset(self):
        """Reset controller state at the end of an episode."""
        self.current_state = None
        self.current_subgoal = None
    
    def interpret_action(self, action: int, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret the numerical action into a meaningful action description.
        
        Args:
            action: Action ID
            state_dict: Current state dictionary
            
        Returns:
            Dictionary describing the action
        """
        action_info = {}
        
        if action == ACTION_BASIC_ATTACK:
            action_info = {
                "name": "basic_attack",
                "description": "Perform a basic attack",
                "target": state_dict.get("current_target", "none")
            }
        elif action == ACTION_MOVE:
            action_info = {
                "name": "move",
                "description": "Move to a better position",
                "direction": "tactical_position"
            }
        elif action in [ACTION_CRITICAL_ATTACK, ACTION_QUICK_DODGE, ACTION_SPRINT, 
                    ACTION_STEALTH, ACTION_BLOODLUST, ACTION_DAGGER_THROW,
                    ACTION_SHADOW_EXTRACTION]:
            skill_info = self.skill_manager.get_skill_by_id(action)
            if skill_info:
                action_info = {
                    "name": skill_info["name"],
                    "description": f"Use skill: {skill_info['name']}",
                    "target": state_dict.get("current_target", "none") if skill_info.get("requires_target", False) else "self",
                    "mana_cost": skill_info.get("mana_cost", 0),
                    "effect": skill_info.get("effect", "none")
                }
            else:
                action_info = {
                    "name": "unknown_skill",
                    "description": "Attempt to use unknown skill"
                }
        elif action == ACTION_USE_HEALTH_POTION:
            action_info = {
                "name": "use_health_potion",
                "description": "Use a health potion to restore health",
                "target": "self",
                "effect": "restore_health"
            }
        elif action == ACTION_USE_MANA_POTION:
            action_info = {
                "name": "use_mana_potion",
                "description": "Use a mana potion to restore mana",
                "target": "self",
                "effect": "restore_mana"
            }
        elif action == ACTION_RETREAT:
            action_info = {
                "name": "retreat",
                "description": "Retreat from combat",
                "target": "self",
                "effect": "escape"
            }
        elif action == ACTION_REST:
            action_info = {
                "name": "rest",
                "description": "Rest to recover health and mana",
                "target": "self",
                "effect": "recovery"
            }
        else:
            action_info = {
                "name": "unknown_action",
                "description": f"Unknown action with ID {action}"
            }
        
        return action_info
            
    def save(self, path: str):
        """
        Save the low-level controller model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
            'reward_history': self.reward_history,
            'action_distribution': self.action_distribution
        }, path)
    
    def load(self, path: str):
        """
        Load the low-level controller model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.loss_history = checkpoint['loss_history']
        self.q_value_history = checkpoint['q_value_history']
        self.reward_history = checkpoint['reward_history']
        self.action_distribution = checkpoint['action_distribution']