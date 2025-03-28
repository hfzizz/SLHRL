import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, Optional

class BaseEnvironment(gym.Env, ABC):
    """
    Abstract base class for all environments in the SLHRL framework.
    Inherits from gym.Env and enforces implementation of core methods.
    """
    
    def __init__(self):
        super().__init__()
        self.observation_space = None  # Will be defined in child classes
        self.action_space = None       # Will be defined in child classes
        self.metadata = {'render_modes': ['human', 'rgb_array']}
        self.render_mode = None
        
    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional seed for the random number generator
            options: Additional options for resetting the environment
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        pass
    
    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The next observation
            reward: The reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        Render the environment.
        
        Returns:
            Rendered output depending on the render mode
        """
        pass
    
    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        pass
    
    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set the seed for the environment's random number generator.
        
        Args:
            seed: The seed value
            
        Returns:
            List containing the seed
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]