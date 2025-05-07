#!/usr/bin/env python
"""
Base model class for the traffic management system.
All models should inherit from this class.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the traffic management system.
    Provides common functionality for saving, loading, and evaluating models.
    """
    
    def __init__(self, config_path, model_name):
        """
        Initialize the base model.
        
        Args:
            config_path (str): Path to the configuration file
            model_name (str): Name of the model
        """
        self.model_name = model_name
        self.config = self.load_config(config_path)
        self.model_config = self.config.get('models', {}).get(model_name, {})
        self.save_path = self.model_config.get('save_path', f'models/{model_name}')
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        logger.info(f"Initialized {self.model_name} model")
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}
    
    def save_model(self, model, custom_path=None):
        """
        Save the model to disk.
        
        Args:
            model: The model to save
            custom_path (str, optional): Custom path to save the model to
        """
        save_path = custom_path or self.save_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{timestamp}"
        
        try:
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
            self._save_model_impl(model, full_path)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'timestamp': timestamp,
                'config': self.model_config
            }
            with open(f"{full_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model saved to {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return None
    
    def load_model(self, path=None):
        """
        Load a model from disk.
        
        Args:
            path (str, optional): Path to load the model from
            
        Returns:
            The loaded model
        """
        load_path = path or self._get_latest_model_path()
        
        if not load_path:
            logger.warning(f"No model found at {self.save_path}")
            return None
        
        try:
            model = self._load_model_impl(load_path)
            logger.info(f"Model loaded from {load_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {load_path}: {str(e)}")
            return None
    
    def _get_latest_model_path(self):
        """
        Get the path of the latest model.
        
        Returns:
            str: Path to the latest model
        """
        if not os.path.exists(self.save_path):
            return None
            
        files = [f for f in os.listdir(self.save_path) if f.startswith(self.model_name) and not f.endswith('_metadata.json')]
        if not files:
            return None
            
        files.sort(reverse=True)  # Sort by timestamp (newest first)
        return os.path.join(self.save_path, files[0])
    
    @abstractmethod
    def _save_model_impl(self, model, path):
        """
        Implementation-specific method to save a model.
        
        Args:
            model: The model to save
            path (str): Path to save the model to
        """
        pass
    
    @abstractmethod
    def _load_model_impl(self, path):
        """
        Implementation-specific method to load a model.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            The loaded model
        """
        pass
    
    @abstractmethod
    def train(self, data):
        """
        Train the model.
        
        Args:
            data: Training data
            
        Returns:
            dict: Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Make predictions with the model.
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, data, targets):
        """
        Evaluate the model.
        
        Args:
            data: Input data
            targets: Ground truth
            
        Returns:
            dict: Evaluation metrics
        """
        pass 