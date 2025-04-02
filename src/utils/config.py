"""
Configuration utility for the VRP Optimizer.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class for VRP Optimizer.
    Handles loading and accessing configuration parameters.
    """
    
    def __init__(self, config_path):
        """
        Initialize the configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Default ALNS parameters
        self.alns_defaults = {
            "iterations": 1000,
            "segment_size": 100,
            "cooling_rate": 0.99,
            "initial_temperature": 100,
            "destroy_weight": 0.5,
            "time_weight": 0.3,
            "capacity_weight": 0.2,
            "min_destroy_percentage": 0.1,
            "max_destroy_percentage": 0.5,
            "noise_parameter": 0.1,
            "reaction_factor": 0.1,
            "weights_decay": 0.8
        }
        
        # Set default values if not in config
        for key, value in self.alns_defaults.items():
            if "alns" not in self.config:
                self.config["alns"] = {}
            if key not in self.config["alns"]:
                self.config["alns"][key] = value
                
    def _load_config(self):
        """
        Load configuration from JSON file.
        
        Returns:
            dict: Configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file {self.config_path} not found. Using default values.")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error parsing configuration file {self.config_path}. Using default values.")
            return {}
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key, use dot notation for nested keys (e.g. 'alns.iterations').
            default: Default value if key is not found.
            
        Returns:
            Configuration value for the key or default if not found.
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def save(self, config_path=None):
        """
        Save current configuration to a file.
        
        Args:
            config_path (str, optional): Path to save configuration. Defaults to self.config_path.
        """
        if config_path is None:
            config_path = self.config_path
            
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        
    def __str__(self):
        """Return string representation of the configuration."""
        return json.dumps(self.config, indent=2) 