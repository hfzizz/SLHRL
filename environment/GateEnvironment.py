import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union

from environment import BaseEnvironment

class GateEnvironment(BaseEnvironment):
    """
    The Gate Environment for SLHRL framework.
    Implements a leveled dungeon with increasing difficulty and boss levels.
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.max_level = 100
        self.current_level = 1
        self.in_penalty_zone = False
        self.penalty_time_remaining = 0
        self.daily_quest_completed = False
        self.difficulty_multiplier = 1.0
        
        # Define action and observation spaces
        # Actions: move (0-3), attack (4), use skill (5-7), etc.
        self.action_space = spaces.Discrete(8)
        
        # Observation space: agent stats, level info, enemy presence, etc.
        # Using a simplified representation for now
        self.observation_space = spaces.Dict({
            'agent_stats': spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),  # health, mana, strength, etc.
            'level_info': spaces.Box(low=0, high=self.max_level, shape=(2,), dtype=np.int32),  # current level, is_boss
            'enemies': spaces.Box(low=0, high=np.inf, shape=(10, 4), dtype=np.float32),  # up to 10 enemies with 4 attributes
            'penalty_zone': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # in_penalty_zone, time_remaining
        })
        
        self.render_mode = render_mode
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize the environment state"""
        self.agent_stats = {
            'health': 100.0,
            'mana': 50.0,
            'strength': 10.0,
            'agility': 10.0,
            'skill_points': 0
        }
        self.current_level = 1
        self.is_boss_level = False
        self.enemies = []
        self._generate_level()
    
    def _generate_level(self) -> None:
        """Generate enemies and challenges for the current level"""
        # Check if current level is a boss level
        self.is_boss_level = (self.current_level % 10 == 0)
        
        # Calculate difficulty
        if self.current_level < 10:
            self.difficulty_multiplier = 1.0 + 0.3 * (self.current_level - 1)
        else:
            self.difficulty_multiplier = 1.0 + 0.3 * (self.current_level // 10 * 9) + 0.3 * (self.current_level % 10)
        
        # If boss level, triple the difficulty
        if self.is_boss_level:
            self.difficulty_multiplier *= 3.0
        
        # Generate enemies based on difficulty
        self.enemies = self._spawn_enemies()
    
    def _spawn_enemies(self) -> list:
        """Spawn enemies based on current level and difficulty"""
        num_enemies = max(1, min(10, self.current_level // 2))
        if self.is_boss_level:
            # Boss level has one powerful enemy
            return [{
                'health': 100.0 * self.difficulty_multiplier,
                'damage': 20.0 * self.difficulty_multiplier,
                'defense': 10.0 * self.difficulty_multiplier,
                'is_boss': True
            }]
        else:
            # Regular level has multiple regular enemies
            return [{
                'health': 50.0 * self.difficulty_multiplier / num_enemies,
                'damage': 10.0 * self.difficulty_multiplier / num_enemies,
                'defense': 5.0 * self.difficulty_multiplier / num_enemies,
                'is_boss': False
            } for _ in range(num_enemies)]
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Convert current state to observation dict"""
        agent_stats_array = np.array([
            self.agent_stats['health'],
            self.agent_stats['mana'],
            self.agent_stats['strength'],
            self.agent_stats['agility'],
            self.agent_stats['skill_points']
        ], dtype=np.float32)
        
        level_info = np.array([self.current_level, int(self.is_boss_level)], dtype=np.int32)
        
        # Convert enemies to array format
        enemy_array = np.zeros((10, 4), dtype=np.float32)
        for i, enemy in enumerate(self.enemies[:10]):
            enemy_array[i] = [
                enemy['health'],
                enemy['damage'],
                enemy['defense'],
                int(enemy['is_boss'])
            ]
        
        penalty_info = np.array([
            int(self.in_penalty_zone),
            self.penalty_time_remaining / 240  # Normalized to [0,1] assuming 4 hours = 240 minutes
        ], dtype=np.float32)
        
        return {
            'agent_stats': agent_stats_array,
            'level_info': level_info,
            'enemies': enemy_array,
            'penalty_zone': penalty_info
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.current_level = min(max(1, options['level']), self.max_level)
        else:
            self.current_level = 1
            
        if options and 'penalty_zone' in options:
            self.in_penalty_zone = options['penalty_zone']
            self.penalty_time_remaining = 240 if self.in_penalty_zone else 0  # 4 hours in minutes
        else:
            self.in_penalty_zone = False
            self.penalty_time_remaining = 0
            
        self._initialize_state()
        observation = self._get_observation()
        info = {
            'current_level': self.current_level,
            'is_boss_level': self.is_boss_level,
            'difficulty': self.difficulty_multiplier,
            'in_penalty_zone': self.in_penalty_zone
        }
        
        return observation, info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        # Handle different zones
        if self.in_penalty_zone:
            # Penalty zone logic
            observation, reward, done, truncated, info = self._penalty_zone_step(action)
        else:
            # Main Gate logic
            observation, reward, done, truncated, info = self._gate_step(action)
        
        return observation, reward, done, truncated, info
    
    def _penalty_zone_step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Handle a step in the penalty zone"""
        reward = -0.1  # Small negative reward for being in penalty zone
        done = False
        truncated = False
        
        # Simulate survival challenge
        self.penalty_time_remaining -= 1
        
        # Random monster attack chance
        if np.random.random() < 0.2:  # 20% chance of monster attack
            damage = 10.0 * (1.0 - min(0.8, self.agent_stats['agility'] / 100))
            self.agent_stats['health'] -= damage
            reward -= 0.5  # Additional penalty for taking damage
        
        # Handle player action
        if action == 4:  # Attack
            reward += 0.2  # Small reward for appropriate action
        
        # Check if penalty time is over
        if self.penalty_time_remaining <= 0:
            self.in_penalty_zone = False
            self.agent_stats['skill_points'] += 3  # Reward for completing penalty quest
            reward += 10.0  # Big reward for surviving
            done = True
        
        # Check if agent died
        if self.agent_stats['health'] <= 0:
            reward -= 20.0  # Big penalty for dying
            done = True
        
        info = {
            'penalty_time_remaining': self.penalty_time_remaining,
            'health': self.agent_stats['health']
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _gate_step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Handle a step in the main Gate environment"""
        reward = 0.0
        done = False
        truncated = False
        
        # Process action
        if action < 4:  # Movement
            # Just a small negative reward for taking time
            reward -= 0.01
        elif action == 4:  # Attack
            # Attack the first enemy
            if self.enemies:
                enemy = self.enemies[0]
                damage = max(0, self.agent_stats['strength'] - enemy['defense'] * 0.5)
                enemy['health'] -= damage
                reward += damage * 0.1  # Reward proportional to damage dealt
                
                # Check if enemy defeated
                if enemy['health'] <= 0:
                    self.enemies.pop(0)
                    reward += 5.0  # Reward for defeating an enemy
                    
                    # If all enemies defeated, level complete
                    if not self.enemies:
                        self.current_level += 1
                        reward += 20.0  # Big reward for completing level
                        if self.is_boss_level:
                            reward += 50.0  # Extra reward for defeating boss
                        
                        # Check if reached max level
                        if self.current_level > self.max_level:
                            done = True
                            reward += 100.0  # Massive reward for completing the game
                        else:
                            self._generate_level()
                else:
                    # Enemy counterattack
                    player_damage = max(0, enemy['damage'] - self.agent_stats['agility'] * 0.2)
                    self.agent_stats['health'] -= player_damage
                    
                    # Check if player died
                    if self.agent_stats['health'] <= 0:
                        done = True
                        reward -= 50.0  # Big penalty for dying
        elif action >= 5:  # Use skill
            # Use mana to deal more damage
            if self.agent_stats['mana'] >= 10:
                self.agent_stats['mana'] -= 10
                
                if self.enemies:
                    skill_damage = self.agent_stats['strength'] * 2
                    for enemy in self.enemies:
                        enemy['health'] -= skill_damage
                    
                    # Remove defeated enemies
                    self.enemies = [e for e in self.enemies if e['health'] > 0]
                    
                    reward += skill_damage * 0.2  # Reward for skill usage
                    
                    # Check if level completed
                    if not self.enemies:
                        self.current_level += 1
                        reward += 20.0
                        if self.is_boss_level:
                            reward += 50.0
                        
                        if self.current_level > self.max_level:
                            done = True
                            reward += 100.0
                        else:
                            self._generate_level()
            else:
                reward -= 1.0  # Penalty for trying to use skill without mana
        
        info = {
            'current_level': self.current_level,
            'is_boss_level': self.is_boss_level,
            'enemies_remaining': len(self.enemies),
            'health': self.agent_stats['health'],
            'mana': self.agent_stats['mana']
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment"""
        if self.render_mode == 'human':
            # Simple text-based rendering for human
            if self.in_penalty_zone:
                print(f"=== PENALTY ZONE ===")
                print(f"Time remaining: {self.penalty_time_remaining} minutes")
            else:
                print(f"=== GATE LEVEL {self.current_level} ===")
                if self.is_boss_level:
                    print("!!! BOSS LEVEL !!!")
                print(f"Difficulty: x{self.difficulty_multiplier:.2f}")
            
            print(f"Player: Health={self.agent_stats['health']:.1f}, Mana={self.agent_stats['mana']:.1f}, " 
                  f"Strength={self.agent_stats['strength']:.1f}, Agility={self.agent_stats['agility']:.1f}")
            
            print(f"Enemies: {len(self.enemies)}")
            for i, enemy in enumerate(self.enemies):
                print(f"  Enemy {i+1}: Health={enemy['health']:.1f}, " 
                      f"Damage={enemy['damage']:.1f}, {'[BOSS]' if enemy['is_boss'] else ''}")
            
            return None
        elif self.render_mode == 'rgb_array':
            # For simplicity, just return a placeholder colored array
            # In a real implementation, this would render a proper visual representation
            image = np.zeros((300, 400, 3), dtype=np.uint8)
            
            # Color based on environment state
            if self.in_penalty_zone:
                image[:, :, 0] = 180  # Reddish for penalty zone
            elif self.is_boss_level:
                image[:, :, 2] = 180  # Bluish for boss levels
            else:
                image[:, :, 1] = 180  # Greenish for regular levels
                
            return image
        
        return None