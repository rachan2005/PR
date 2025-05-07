#!/usr/bin/env python
"""
Reinforcement learning model for adaptive traffic signal control.
Uses Deep Q-Network (DQN) to optimize traffic signal timings.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import logging
import pickle

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Memory buffer for experience replay in DQN."""
    
    def __init__(self, capacity):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from buffer.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class SignalControlModel(BaseModel):
    """
    Reinforcement learning model for adaptive traffic signal control.
    Uses Deep Q-Network (DQN) to optimize signal timings.
    """
    
    def __init__(self, config_path):
        """
        Initialize the signal control model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        super().__init__(config_path, "signal_control")
        
        # Model parameters from config
        self.learning_rate = self.model_config.get("learning_rate", 0.0001)
        self.discount_factor = self.model_config.get("discount_factor", 0.99)
        self.exploration_rate = self.model_config.get("exploration_rate", 0.1)
        self.exploration_decay = self.model_config.get("exploration_decay", 0.995)
        self.min_exploration_rate = self.model_config.get("min_exploration_rate", 0.01)
        self.batch_size = self.model_config.get("batch_size", 64)
        self.memory_size = self.model_config.get("memory_size", 10000)
        self.target_update_frequency = self.model_config.get("target_update_frequency", 1000)
        
        # Reward function weights
        reward_function = self.model_config.get("reward_function", {})
        self.queue_length_weight = reward_function.get("queue_length_weight", -0.5)
        self.waiting_time_weight = reward_function.get("waiting_time_weight", -0.3)
        self.throughput_weight = reward_function.get("throughput_weight", 0.2)
        
        # Initialize models and memory
        self.policy_model = None
        self.target_model = None
        self.memory = ReplayBuffer(self.memory_size)
        self.update_counter = 0
        
        logger.info("Signal Control Model initialized")
    
    def _build_model(self, state_size, action_size):
        """
        Build the DQN model architecture.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            
        Returns:
            tf.keras.Model: Compiled DQN model
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(state_size,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        logger.info(f"DQN model built with state size {state_size} and action size {action_size}")
        return model
    
    def initialize(self, state_size, action_size):
        """
        Initialize the DQN models.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
        """
        self.policy_model = self._build_model(state_size, action_size)
        self.target_model = self._build_model(state_size, action_size)
        self._update_target_model()
        
        logger.info("Policy and target models initialized")
    
    def _update_target_model(self):
        """Update the target model with weights from the policy model."""
        self.target_model.set_weights(self.policy_model.get_weights())
        logger.debug("Target model weights updated")
    
    def select_action(self, state, explore=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (numpy.ndarray): Current state
            explore (bool): Whether to use exploration
            
        Returns:
            int: Selected action
        """
        if self.policy_model is None:
            logger.error("Model not initialized")
            return None
        
        # Reshape state for model input
        state = np.reshape(state, [1, -1])
        
        if explore and np.random.rand() < self.exploration_rate:
            # Exploration: random action
            action_size = self.policy_model.output_shape[-1]
            return np.random.randint(0, action_size)
        else:
            # Exploitation: best action
            q_values = self.policy_model.predict(state, verbose=0)[0]
            return np.argmax(q_values)
    
    def train(self, data):
        """
        Train the DQN model through simulated episodes.
        In practice, this would be connected to a traffic simulator.
        
        Args:
            data: Simulation environment or historical data
            
        Returns:
            dict: Training metrics
        """
        # In a real implementation, this would interact with a traffic simulator
        # For now, we'll implement a stub that assumes data is an environment
        env = data
        episodes = 1000
        max_steps = 500
        
        total_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select action
                action = self.select_action(state, explore=True)
                
                # Take action and observe result
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                self.memory.add(state, action, reward, next_state, done)
                
                # Train network
                if len(self.memory) > self.batch_size:
                    self._replay()
                
                # Update target network
                self.update_counter += 1
                if self.update_counter % self.target_update_frequency == 0:
                    self._update_target_model()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * self.exploration_decay
            )
            
            total_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, Exploration: {self.exploration_rate:.2f}")
        
        # Save the model
        self.save_model(self.policy_model)
        
        logger.info("Signal control model training completed")
        
        # Return training metrics
        return {
            'rewards': total_rewards,
            'final_exploration_rate': self.exploration_rate
        }
    
    def _replay(self):
        """Train the model on a batch of experiences."""
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calculate Q-values
        current_q = self.policy_model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i, actions[i]] = rewards[i]
            else:
                current_q[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q[i])
        
        # Train the policy model
        self.policy_model.fit(states, current_q, epochs=1, verbose=0)
    
    def predict(self, state):
        """
        Predict optimal signal timing from the current state.
        
        Args:
            state (numpy.ndarray): Current traffic state
            
        Returns:
            int: Optimal action
        """
        if self.policy_model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Use the policy model to choose the best action (no exploration)
        return self.select_action(state, explore=False)
    
    def evaluate(self, env, num_episodes=10):
        """
        Evaluate the DQN model on the environment.
        
        Args:
            env: Evaluation environment
            num_episodes (int): Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if self.policy_model is None:
            logger.error("Model not trained or loaded")
            return None
        
        total_rewards = []
        waiting_times = []
        throughputs = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                action = self.select_action(state, explore=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Collect metrics from environment info
                if 'waiting_time' in info:
                    waiting_times.append(info['waiting_time'])
                if 'throughput' in info:
                    throughputs.append(info['throughput'])
            
            total_rewards.append(episode_reward)
            
            logger.info(f"Evaluation Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        avg_waiting_time = np.mean(waiting_times) if waiting_times else None
        avg_throughput = np.mean(throughputs) if throughputs else None
        
        metrics = {
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting_time,
            'avg_throughput': avg_throughput
        }
        
        logger.info(f"Model evaluation: Avg Reward={avg_reward:.2f}")
        
        return metrics
    
    def calculate_reward(self, queue_length, waiting_time, throughput):
        """
        Calculate reward based on traffic metrics.
        
        Args:
            queue_length (float): Length of vehicle queue
            waiting_time (float): Average waiting time
            throughput (float): Number of vehicles passing through
            
        Returns:
            float: Calculated reward
        """
        reward = (
            self.queue_length_weight * queue_length +
            self.waiting_time_weight * waiting_time +
            self.throughput_weight * throughput
        )
        
        return reward
    
    def _save_model_impl(self, model, path):
        """
        Implementation to save a DQN model.
        
        Args:
            model (tf.keras.Model): The model to save
            path (str): Path to save the model to
        """
        # Save policy model
        model.save(f"{path}_policy.h5")
        
        # Save target model if it exists
        if self.target_model is not None:
            self.target_model.save(f"{path}_target.h5")
        
        # Save exploration rate and memory
        metadata = {
            'exploration_rate': self.exploration_rate,
            'update_counter': self.update_counter
        }
        
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save a sample of the memory buffer
        if len(self.memory) > 0:
            memory_sample = list(self.memory.buffer)[:min(1000, len(self.memory))]
            with open(f"{path}_memory_sample.pkl", 'wb') as f:
                pickle.dump(memory_sample, f)
    
    def _load_model_impl(self, path):
        """
        Implementation to load a DQN model.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            tf.keras.Model: The loaded policy model
        """
        # Load policy model
        self.policy_model = load_model(f"{path}_policy.h5")
        
        # Load target model if file exists
        target_path = f"{path}_target.h5"
        if os.path.exists(target_path):
            self.target_model = load_model(target_path)
        else:
            # Initialize target model with same architecture
            state_size = self.policy_model.input_shape[1]
            action_size = self.policy_model.output_shape[1]
            self.target_model = self._build_model(state_size, action_size)
            self._update_target_model()
        
        # Load metadata if file exists
        metadata_path = f"{path}_metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.exploration_rate = metadata.get('exploration_rate', self.exploration_rate)
                self.update_counter = metadata.get('update_counter', 0)
        
        # Load memory sample if file exists
        memory_path = f"{path}_memory_sample.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory_sample = pickle.load(f)
                for experience in memory_sample:
                    self.memory.add(*experience)
        
        return self.policy_model 