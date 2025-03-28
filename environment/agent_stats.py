import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class AgentStats:
    """
    Class to manage the agent's statistics, abilities, and progression.
    """
    
    def __init__(self, 
                 initial_health: float = 100.0,
                 initial_mana: float = 50.0,
                 initial_strength: float = 10.0,
                 initial_agility: float = 10.0,
                 initial_intelligence: float = 10.0,
                 initial_skill_points: int = 0):
        """
        Initialize agent statistics.
        
        Args:
            initial_health: Starting health points
            initial_mana: Starting mana points
            initial_strength: Starting strength value
            initial_agility: Starting agility value
            initial_intelligence: Starting intelligence value
            initial_skill_points: Starting skill points
        """
        # Basic stats
        self.max_health = initial_health
        self.health = initial_health
        self.max_mana = initial_mana
        self.mana = initial_mana
        self.strength = initial_strength
        self.agility = initial_agility
        self.intelligence = initial_intelligence
        self.skill_points = initial_skill_points
        
        # Level and experience
        self.level = 1
        self.exp = 0
        self.exp_to_next_level = self._calculate_exp_required(self.level)
        
        # Skills and abilities
        self.skills = {}
        self.active_skills = []
        self.passive_skills = []
        
        # Skill levels and experience (independent of character level)
        self.skill_exp = {}
        self.skill_levels = {}
        
        # Shadow army
        self.shadow_soldiers = []
        self.max_shadow_soldiers = 1  # Start with 1 slot, increases with level
        
        # Initialize starting skills
        self._initialize_skills()
    
    def _initialize_skills(self) -> None:
        """Initialize the starting skills for the agent."""
        self.skills = {
            # Basic skills
            "basic_attack": {
                "name": "Basic Attack",
                "damage": lambda: self.strength * 1.0,
                "mana_cost": 0,
                "cooldown": 0,
                "type": "active",
                "unlocked": True,
                "level": 1,
                "category": "basic"
            },
            "quick_dodge": {
                "name": "Quick Dodge",
                "effect": lambda: min(0.5, self.agility * 0.05),  # Dodge chance
                "type": "passive",
                "unlocked": True,
                "level": 1,
                "category": "basic"
            },
            
            # Combat skills (initially locked)
            "critical_attack": {
                "name": "Critical Attack",
                "damage": lambda: self.strength * 2.0 * (1.0 + 0.1 * self.skill_levels.get("critical_attack", 1)),
                "mana_cost": 70,
                "cooldown": 5,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "evolved_form": "mutilation",
                "evolution_level": 5,  # Level at which it evolves
                "requirements": {"level": 5, "agility": 15},
                "category": "combat"
            },
            "mutilation": {
                "name": "Mutilation",
                "damage": lambda: self.strength * 3.5 * (1.0 + 0.15 * self.skill_levels.get("mutilation", 1)),
                "mana_cost": 100,
                "cooldown": 8,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"critical_attack_level": 5},
                "category": "combat"
            },
            "bloodlust": {
                "name": "Bloodlust",
                "effect": lambda: 0.5 + 0.05 * self.skill_levels.get("bloodlust", 1),  # % stat reduction
                "duration": lambda: 60 + 10 * self.skill_levels.get("bloodlust", 1),  # seconds
                "mana_cost": 100,
                "cooldown": 120,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"level": 10, "strength": 20},
                "category": "combat"
            },
            "stealth": {
                "name": "Stealth",
                "activation_cost": 200,
                "maintenance_cost": 10,  # per second
                "duration": lambda: 30 + 5 * self.skill_levels.get("stealth", 1),  # seconds
                "cooldown": 180,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"level": 15, "agility": 25},
                "category": "combat"
            },
            "sprint": {
                "name": "Sprint",
                "speed_boost": lambda: 0.2 + 0.05 * self.skill_levels.get("sprint", 1),  # % speed increase
                "mana_cost": 5,
                "duration": lambda: 20 + 5 * self.skill_levels.get("sprint", 1),  # seconds
                "cooldown": 30,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "evolved_form": "quicksilver",
                "evolution_level": 5,
                "requirements": {"level": 8, "agility": 18},
                "category": "combat"
            },
            "quicksilver": {
                "name": "Quicksilver",
                "speed_boost": lambda: 0.3 + 0.07 * self.skill_levels.get("quicksilver", 1),  # % speed increase
                "mana_cost": 10,
                "duration": lambda: 30 + 8 * self.skill_levels.get("quicksilver", 1),  # seconds
                "cooldown": 45,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"sprint_level": 5},
                "category": "combat"
            },
            "dagger_throw": {
                "name": "Dagger Throw",
                "damage": lambda: self.strength * 1.5 * (1.0 + 0.1 * self.skill_levels.get("dagger_throw", 1)),
                "accuracy": lambda: 0.7 + 0.03 * self.skill_levels.get("dagger_throw", 1),  # % hit chance
                "mana_cost": 30,
                "cooldown": 10,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "evolved_form": "dagger_rush",
                "evolution_level": 6,
                "requirements": {"level": 12, "agility": 20},
                "category": "combat"
            },
            "dagger_rush": {
                "name": "Dagger Rush",
                "damage": lambda: self.strength * 1.2 * (1.0 + 0.08 * self.skill_levels.get("dagger_rush", 1)),
                "daggers": lambda: 3 + min(7, self.skill_levels.get("dagger_rush", 1)),  # number of daggers
                "accuracy": lambda: 0.8 + 0.02 * self.skill_levels.get("dagger_rush", 1),  # % hit chance
                "mana_cost": 60,
                "cooldown": 20,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"dagger_throw_level": 6},
                "category": "combat"
            },
            
            # Job skill
            "shadow_extraction": {
                "name": "Shadow Extraction",
                "success_rate": lambda target_level: min(0.9, 0.6 + 0.1 * self.skill_levels.get("shadow_extraction", 1) - 0.1 * max(0, target_level - self.level) / 10),
                "max_attempts": 3,
                "cooldown": 60,
                "type": "active",
                "unlocked": False,
                "level": 0,
                "requirements": {"level": 20},
                "category": "job"
            }
        }
        
        # Initialize skill experience and levels
        for skill_name in self.skills:
            self.skill_exp[skill_name] = 0
            self.skill_levels[skill_name] = 1 if self.skills[skill_name]["unlocked"] else 0
        
        # Register starting skills
        self.active_skills.append("basic_attack")
        self.passive_skills.append("quick_dodge")
    
    def _calculate_exp_required(self, level: int) -> int:
        """
        Calculate experience points required for the next level.
        
        Args:
            level: Current level
            
        Returns:
            Experience points required to level up
        """
        return int(100 * (level ** 1.5))
    
    def _calculate_skill_exp_required(self, skill_name: str, level: int) -> int:
        """
        Calculate experience points required for the next skill level.
        
        Args:
            skill_name: Name of the skill
            level: Current skill level
            
        Returns:
            Experience points required to level up the skill
        """
        base_value = 50  # Base experience needed
        
        # Different categories might have different scaling
        if skill_name in self.skills:
            category = self.skills[skill_name]["category"]
            if category == "combat":
                return int(base_value * (level ** 1.3))
            elif category == "job":
                return int(base_value * (level ** 1.7))  # Job skills need more exp
        
        # Default
        return int(base_value * (level ** 1.4))
    
    def add_experience(self, exp_points: int) -> Tuple[bool, int]:
        """
        Add experience points to the agent and handle level ups.
        
        Args:
            exp_points: Amount of experience to add
            
        Returns:
            Tuple containing (leveled_up, levels_gained)
        """
        self.exp += exp_points
        levels_gained = 0
        leveled_up = False
        
        # Check for level ups
        while self.exp >= self.exp_to_next_level:
            self.exp -= self.exp_to_next_level
            self.level += 1
            levels_gained += 1
            leveled_up = True
            
            # Increase stats for level up
            self.max_health += 10.0
            self.health = self.max_health  # Heal to full on level up
            self.max_mana += 5.0
            self.mana = self.max_mana      # Restore full mana on level up
            self.strength += 1.0
            self.agility += 1.0
            self.intelligence += 1.0
            self.skill_points += 3         # Gain 3 skill points per level
            
            # Calculate exp needed for next level
            self.exp_to_next_level = self._calculate_exp_required(self.level)
            
            # Check if shadow army capacity increases at milestone levels
            if self.level in [20, 40, 60, 80, 100]:
                self.max_shadow_soldiers += 1
                
            # Check for skill unlocks based on level requirements
            self._check_skill_unlocks()
        
        return leveled_up, levels_gained
    
    def _check_skill_unlocks(self) -> List[str]:
        """
        Check if any new skills can be unlocked based on current stats.
        
        Returns:
            List of newly unlockable skill names
        """
        unlockable = []
        
        for skill_name, skill in self.skills.items():
            if not skill["unlocked"] and "requirements" in skill:
                req = skill["requirements"]
                can_unlock = True
                
                # Check level requirement
                if "level" in req and self.level < req["level"]:
                    can_unlock = False
                
                # Check stat requirements
                if "strength" in req and self.strength < req["strength"]:
                    can_unlock = False
                if "agility" in req and self.agility < req["agility"]:
                    can_unlock = False
                if "intelligence" in req and self.intelligence < req["intelligence"]:
                    can_unlock = False
                
                # Check skill level requirements (for evolved forms)
                for req_key, req_value in req.items():
                    if req_key.endswith("_level"):
                        base_skill = req_key.replace("_level", "")
                        if base_skill in self.skill_levels and self.skill_levels[base_skill] < req_value:
                            can_unlock = False
                
                if can_unlock:
                    unlockable.append(skill_name)
        
        return unlockable
    
    def add_skill_experience(self, skill_name: str, exp_points: int) -> Tuple[bool, int]:
        """
        Add experience to a specific skill and handle level ups.
        
        Args:
            skill_name: Name of the skill to add experience to
            exp_points: Amount of experience to add
            
        Returns:
            Tuple containing (leveled_up, levels_gained)
        """
        if skill_name not in self.skills or not self.skills[skill_name]["unlocked"]:
            return False, 0
        
        self.skill_exp[skill_name] += exp_points
        levels_gained = 0
        leveled_up = False
        current_level = self.skill_levels[skill_name]
        
        # Calculate exp needed for next level
        exp_required = self._calculate_skill_exp_required(skill_name, current_level)
        
        # Check for level ups
        while self.skill_exp[skill_name] >= exp_required:
            self.skill_exp[skill_name] -= exp_required
            self.skill_levels[skill_name] += 1
            levels_gained += 1
            leveled_up = True
            
            # Update required exp for next level
            current_level = self.skill_levels[skill_name]
            exp_required = self._calculate_skill_exp_required(skill_name, current_level)
            
            # Check if skill evolves
            if "evolved_form" in self.skills[skill_name] and current_level >= self.skills[skill_name]["evolution_level"]:
                evolved_skill = self.skills[skill_name]["evolved_form"]
                if evolved_skill in self.skills and not self.skills[evolved_skill]["unlocked"]:
                    # Make evolved form available for unlocking
                    self.skills[evolved_skill]["requirements"] = {
                        f"{skill_name}_level": self.skills[skill_name]["evolution_level"]
                    }
        
        return leveled_up, levels_gained
    
    def take_damage(self, damage: float) -> float:
        """
        Apply damage to the agent, considering dodge chance.
        
        Args:
            damage: Raw damage amount
            
        Returns:
            Actual damage taken after calculations
        """
        # Check for dodge (if quick_dodge is unlocked)
        dodge_chance = 0.0
        if "quick_dodge" in self.passive_skills:
            dodge_skill = self.skills["quick_dodge"]
            dodge_chance = dodge_skill["effect"]()
        
        # Roll for dodge
        if np.random.random() < dodge_chance:
            return 0.0  # Dodged the attack
        
        # Calculate damage reduction from agility (small amount)
        damage_reduction = min(0.3, self.agility * 0.01)  # Max 30% reduction
        actual_damage = damage * (1.0 - damage_reduction)
        
        # Apply damage
        self.health = max(0.0, self.health - actual_damage)
        
        return actual_damage
    
    def use_skill(self, skill_name: str, target: Optional[Dict] = None) -> Tuple[bool, Any, str]:
        """
        Use a skill if available and return the result.
        
        Args:
            skill_name: Name of the skill to use
            target: Optional target information for skills that need it
            
        Returns:
            Tuple of (success, effect_value, message)
        """
        if skill_name not in self.skills or skill_name not in self.active_skills:
            return False, None, f"Skill {skill_name} not available"
        
        skill = self.skills[skill_name]
        
        # Check if unlocked
        if not skill["unlocked"]:
            return False, None, f"Skill {skill_name} not unlocked"
        
        # Handle different skill types
        if skill_name == "shadow_extraction":
            return self._use_shadow_extraction(target)
        elif skill_name in ["critical_attack", "mutilation", "dagger_throw", "dagger_rush"]:
            return self._use_damage_skill(skill_name)
        elif skill_name in ["bloodlust", "stealth", "sprint", "quicksilver"]:
            return self._use_buff_skill(skill_name)
        else:
            # Generic skill usage
            # Check mana cost
            if self.mana < skill["mana_cost"]:
                return False, None, f"Not enough mana to use {skill_name}"
            
            # Use the skill
            self.mana -= skill["mana_cost"]
            
            # Calculate effect (e.g., damage)
            effect_value = skill["damage"]() if "damage" in skill else None
            
            # Add some experience to the skill
            self.add_skill_experience(skill_name, 5)
            
            return True, effect_value, f"Used {skill['name']} successfully"
    
    def _use_damage_skill(self, skill_name: str) -> Tuple[bool, float, str]:
        """
        Use a damage-dealing skill.
        
        Args:
            skill_name: Name of the skill to use
            
        Returns:
            Tuple of (success, damage_value, message)
        """
        skill = self.skills[skill_name]
        
        # Check mana cost
        if self.mana < skill["mana_cost"]:
            return False, 0.0, f"Not enough mana to use {skill_name}"
        
        # Use the skill
        self.mana -= skill["mana_cost"]
        
        # Calculate damage
        damage = skill["damage"]()
        
        # For skills like dagger rush that hit multiple times
        if skill_name == "dagger_rush":
            num_daggers = skill["daggers"]()
            accuracy = skill["accuracy"]()
            
            # Calculate hits based on accuracy
            hits = sum(1 for _ in range(int(num_daggers)) if np.random.random() < accuracy)
            total_damage = damage * hits
            
            # Add skill experience (more for successful hits)
            self.add_skill_experience(skill_name, 5 + hits * 2)
            
            return True, total_damage, f"Used {skill['name']} with {hits}/{int(num_daggers)} hits for {total_damage:.1f} damage"
        elif skill_name == "dagger_throw":
            # Single hit with accuracy check
            accuracy = skill["accuracy"]()
            if np.random.random() < accuracy:
                self.add_skill_experience(skill_name, 8)  # More exp for accurate throw
                return True, damage, f"Used {skill['name']} successfully for {damage:.1f} damage"
            else:
                self.add_skill_experience(skill_name, 3)  # Less exp for miss
                return True, 0.0, f"Used {skill['name']} but missed"
        else:
            # Normal damage skill
            self.add_skill_experience(skill_name, 10)
            return True, damage, f"Used {skill['name']} successfully for {damage:.1f} damage"
    
    def _use_buff_skill(self, skill_name: str) -> Tuple[bool, Dict, str]:
        """
        Use a buff/utility skill.
        
        Args:
            skill_name: Name of the skill to use
            
        Returns:
            Tuple of (success, buff_effect, message)
        """
        skill = self.skills[skill_name]
        
        # Check mana cost (for stealth, this is the activation cost)
        mana_cost = skill["activation_cost"] if "activation_cost" in skill else skill["mana_cost"]
        if self.mana < mana_cost:
            return False, {}, f"Not enough mana to use {skill_name}"
        
        # Use the skill
        self.mana -= mana_cost
        
        # Calculate duration
        duration = skill["duration"]() if callable(skill["duration"]) else skill["duration"]
        
        # Create buff effect dictionary
        buff_effect = {
            "name": skill["name"],
            "duration": duration,
            "remaining": duration
        }
        
        # Add skill-specific effects
        if skill_name == "bloodlust":
            buff_effect["stat_reduction"] = skill["effect"]()
        elif skill_name in ["sprint", "quicksilver"]:
            buff_effect["speed_boost"] = skill["speed_boost"]()
        elif skill_name == "stealth":
            buff_effect["maintenance_cost"] = skill["maintenance_cost"]
            buff_effect["is_stealthed"] = True
        
        # Add skill experience
        self.add_skill_experience(skill_name, 8)
        
        return True, buff_effect, f"Activated {skill['name']} for {duration} seconds"
    
    def _use_shadow_extraction(self, target: Optional[Dict]) -> Tuple[bool, Optional[Dict], str]:
        """
        Use the Shadow Extraction job skill.
        
        Args:
            target: Target information dict containing at least 'level' and 'type'
            
        Returns:
            Tuple of (success, extracted_shadow, message)
        """
        skill = self.skills["shadow_extraction"]
        
        # Check if we have a valid target
        if not target:
            return False, None, "No target for Shadow Extraction"
        
        # Check invalid target types
        invalid_types = ["demon", "ruler", "monarch"]
        if "type" in target and target["type"].lower() in invalid_types:
            return False, None, f"Cannot extract shadow from {target['type']}"
        
        # Check if we have space for more shadows
        if len(self.shadow_soldiers) >= self.max_shadow_soldiers:
            return False, None, f"Shadow army full ({len(self.shadow_soldiers)}/{self.max_shadow_soldiers})"
        
        # Calculate success chance
        target_level = target.get("level", 1)
        success_rate = skill["success_rate"](target_level)
        
        # Track number of attempts on this target
        target_id = target.get("id", "unknown")
        target_key = f"shadow_extraction_attempts_{target_id}"
        if not hasattr(self, target_key):
            setattr(self, target_key, 0)
        
        attempts = getattr(self, target_key)
        if attempts >= skill["max_attempts"]:
            return False, None, f"Maximum extraction attempts reached for this target"
        
        # Increment attempts
        setattr(self, target_key, attempts + 1)
        
        # Roll for success
        if np.random.random() < success_rate:
            # Create shadow soldier
            shadow = {
                "id": f"shadow_{len(self.shadow_soldiers) + 1}",
                "name": f"Shadow {target.get('name', 'Soldier')}",
                "level": target_level,
                "health": target.get("health", 50.0) * 0.8,
                "damage": target.get("damage", 10.0) * 0.7,
                "abilities": target.get("abilities", []),
                "original_type": target.get("type", "unknown")
            }
            
            self.shadow_soldiers.append(shadow)
            self.add_skill_experience("shadow_extraction", 20 + target_level)
            
            return True, shadow, f"Successfully extracted shadow from {target.get('name', 'target')}"
        else:
            self.add_skill_experience("shadow_extraction", 5)
            return False, None, f"Failed to extract shadow ({attempts + 1}/{skill['max_attempts']} attempts)"
    
    def spend_skill_point(self, skill_name: str) -> Tuple[bool, str]:
        """
        Spend a skill point to upgrade or unlock a skill.
        
        Args:
            skill_name: Name of the skill to upgrade
            
        Returns:
            Tuple of (success, message)
        """
        if self.skill_points <= 0:
            return False, "No skill points available"
        
        if skill_name not in self.skills:
            return False, f"Skill {skill_name} does not exist"
        
        skill = self.skills[skill_name]
        
        # Unlock new skill
        if not skill["unlocked"]:
            # Check requirements
            if "requirements" in skill:
                req = skill["requirements"]
                
                # Check level requirement
                if "level" in req and self.level < req["level"]:
                    return False, f"Level {req['level']} required to unlock {skill['name']}"
                
                # Check stat requirements
                if "strength" in req and self.strength < req["strength"]:
                    return False, f"Strength {req['strength']} required to unlock {skill['name']}"
                
                if "agility" in req and self.agility < req["agility"]:
                    return False, f"Agility {req['agility']} required to unlock {skill['name']}"
                
                if "intelligence" in req and self.intelligence < req["intelligence"]:
                    return False, f"Intelligence {req['intelligence']} required to unlock {skill['name']}"
                
                # Check skill level requirements (for evolved forms)
                for req_key, req_value in req.items():
                    if req_key.endswith("_level"):
                        base_skill = req_key.replace("_level", "")
                        if base_skill in self.skill_levels and self.skill_levels[base_skill] < req_value:
                            return False, f"{self.skills[base_skill]['name']} level {req_value} required"
            
            # All requirements met, unlock the skill
            skill["unlocked"] = True
            self.skill_points -= 1
            self.skill_levels[skill_name] = 1  # Start at level 1
            
            # Register in appropriate list
            if skill["type"] == "active":
                self.active_skills.append(skill_name)
            else:
                self.passive_skills.append(skill_name)
                
            return True, f"Unlocked {skill['name']}"
        
        # Upgrade existing skill (increases base stats but not level)
        # Skill leveling is handled through experience, this just boosts stats
        skill_boost_msg = "Unknown effect"
        
        if skill_name in ["critical_attack", "mutilation", "dagger_throw", "dagger_rush"]:
            # For damage skills, boost base damage
            old_dmg = skill["damage"]()
            skill["damage"] = lambda s=skill: s["damage"]() * 1.1  # 10% boost
            new_dmg = skill["damage"]()
            skill_boost_msg = f"Damage increased from {old_dmg:.1f} to {new_dmg:.1f}"
        
        elif skill_name in ["sprint", "quicksilver"]:
            # For speed skills, boost duration
            old_dur = skill["duration"]() if callable(skill["duration"]) else skill["duration"]
            if callable(skill["duration"]):
                old_func = skill["duration"]
                skill["duration"] = lambda s=old_func: s() * 1.2  # 20% boost
            else:
                skill["duration"] *= 1.2
            new_dur = skill["duration"]() if callable(skill["duration"]) else skill["duration"]
            skill_boost_msg = f"Duration increased from {old_dur:.1f} to {new_dur:.1f} seconds"
        
        elif skill_name == "shadow_extraction":
            # For shadow extraction, improve success rate
            old_func = skill["success_rate"]
            skill["success_rate"] = lambda tl, f=old_func: min(0.95, f(tl) + 0.05)  # +5% success rate
            skill_boost_msg = "Extraction success rate improved by 5%"
        
        self.skill_points -= 1
        return True, f"Enhanced {skill['name']}: {skill_boost_msg}"
    
    def heal(self, amount: float) -> float:
        """
        Heal the agent for the specified amount.
        
        Args:
            amount: Amount to heal
            
        Returns:
            Actual amount healed
        """
        old_health = self.health
        self.health = min(self.max_health, self.health + amount)
        return self.health - old_health
    
    def restore_mana(self, amount: float) -> float:
        """
        Restore mana for the specified amount.
        
        Args:
            amount: Amount of mana to restore
            
        Returns:
            Actual amount restored
        """
        old_mana = self.mana
        self.mana = min(self.max_mana, self.mana + amount)
        return self.mana - old_mana
    
    def update_stealth(self) -> Tuple[bool, str]:
        """
        Update the stealth status, consuming mana for maintenance.
        
        Returns:
            Tuple of (still_stealthed, message)
        """
        for buff in getattr(self, "active_buffs", []):
            if buff.get("name") == "Stealth" and buff.get("is_stealthed", False):
                if self.mana >= buff["maintenance_cost"]:
                    self.mana -= buff["maintenance_cost"]
                    return True, "Maintaining stealth"
                else:
                    buff["is_stealthed"] = False
                    return False, "Stealth broken: not enough mana"
        
        return False, "Not in stealth mode"
    
    def get_shadow_soldiers_stats(self) -> List[Dict]:
        """
        Get stats for all shadow soldiers.
        
        Returns:
            List of shadow soldier stat dictionaries
        """
        return [soldier.copy() for soldier in self.shadow_soldiers]
    
    def get_stats_dict(self) -> Dict:
        """
        Get a dictionary of the agent's current stats.
        
        Returns:
            Dictionary of agent stats
        """
        return {
            'health': self.health,
            'max_health': self.max_health,
            'mana': self.mana,
            'max_mana': self.max_mana,
            'strength': self.strength,
            'agility': self.agility,
            'intelligence': self.intelligence,
            'level': self.level,
            'experience': self.exp,
            'exp_to_next': self.exp_to_next_level,
            'skill_points': self.skill_points,
            'active_skills': self.active_skills.copy(),
            'shadow_soldiers': len(self.shadow_soldiers),
            'max_shadow_soldiers': self.max_shadow_soldiers
        }
    
    def get_stats_array(self) -> np.ndarray:
        """
        Get agent stats as a numpy array (for RL observation).
        
        Returns:
            Numpy array containing key stats
        """
        return np.array([
            self.health, 
            self.max_health,
            self.mana,
            self.max_mana,
            self.strength,
            self.agility,
            self.intelligence,
            self.level,
            self.skill_points,
            len(self.shadow_soldiers)
        ], dtype=np.float32)
    
    def reset(self) -> None:
        """Reset the agent to initial state (for environment reset)."""
        # Restore health and mana to maximum
        self.health = self.max_health
        self.mana = self.max_mana
        
        # Clear any active buffs or status effects
        self.active_buffs = []
        
        # Reset cooldowns for all skills
        for skill_name in self.skills:
            if "current_cooldown" in self.skills[skill_name]:
                self.skills[skill_name]["current_cooldown"] = 0
        
        # Clear any temporary target-specific data
        for attr_name in dir(self):
            if attr_name.startswith("shadow_extraction_attempts_"):
                delattr(self, attr_name)
                
        # Reset any active skill states
        for skill_name in self.active_skills:
            if skill_name == "stealth" and hasattr(self, "is_stealthed"):
                self.is_stealthed = False
                
        # Heal shadow soldiers if they exist
        for soldier in self.shadow_soldiers:
            if "health" in soldier and "max_health" in soldier:
                soldier["health"] = soldier["max_health"]
            elif "health" in soldier:
                # If no max_health defined, restore to original health
                soldier["health"] = soldier.get("original_health", soldier["health"])