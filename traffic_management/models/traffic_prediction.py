#!/usr/bin/env python
"""
Traffic prediction model implementation using LSTM.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import logging

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class TrafficPredictionModel(BaseModel):
    """
    Traffic prediction model using LSTM neural network.
    Predicts traffic metrics (volume, speed, occupancy) based on historical data.
    """
    
    def __init__(self, config_path):
        """
        Initialize the traffic prediction model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        super().__init__(config_path, "traffic_prediction")
        
        # Model parameters from config
        self.sequence_length = self.model_config.get("sequence_length", 24)
        self.hidden_units = self.model_config.get("hidden_units", 128)
        self.dropout_rate = self.model_config.get("dropout_rate", 0.2)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.epochs = self.model_config.get("epochs", 100)
        
        # Initialize model and scalers
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        logger.info("Traffic Prediction Model initialized")
    
    def _build_model(self, input_shape, output_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            output_shape (int): Number of output features
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(units=self.hidden_units, return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(units=self.hidden_units // 2),
            Dropout(self.dropout_rate),
            Dense(units=output_shape)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"LSTM model built with input shape {input_shape} and output shape {output_shape}")
        return model
    
    def _prepare_sequences(self, data):
        """
        Prepare input sequences and targets for LSTM.
        
        Args:
            data (pandas.DataFrame): Input time-series data
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the targets
        """
        # Scale the data
        scaled_data = self.feature_scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data):
        """
        Train the LSTM model.
        
        Args:
            data (pandas.DataFrame): Input time-series data
            
        Returns:
            dict: Training history
        """
        logger.info("Starting training of traffic prediction model")
        
        # Prepare sequences
        X, y = self._prepare_sequences(data)
        
        # Set target scaler
        self.target_scaler.fit(y)
        
        # Split into train and validation sets
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_shape = y_train.shape[1]
            self.model = self._build_model(input_shape, output_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.save_path, "checkpoint"),
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        self.save_model(self.model)
        
        logger.info("Traffic prediction model training completed")
        
        # Return training history
        return history.history
    
    def predict(self, data):
        """
        Make predictions with the LSTM model.
        
        Args:
            data (pandas.DataFrame): Input data for prediction
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Scale input data
        scaled_data = self.feature_scaler.transform(data)
        
        # Prepare sequences
        sequences = []
        for i in range(max(0, len(scaled_data) - self.sequence_length) + 1):
            sequences.append(scaled_data[i:i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        # Make predictions
        predictions = self.model.predict(sequences)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def evaluate(self, data, targets):
        """
        Evaluate the LSTM model.
        
        Args:
            data (pandas.DataFrame): Input data
            targets (pandas.DataFrame): Ground truth values
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Make predictions
        predictions = self.predict(data)
        
        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mape = mean_absolute_percentage_error(targets, predictions)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Model evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}")
        
        return metrics
    
    def _save_model_impl(self, model, path):
        """
        Implementation to save a Keras model.
        
        Args:
            model (tf.keras.Model): The model to save
            path (str): Path to save the model to
        """
        model.save(f"{path}.h5")
        
        # Save scalers
        np.save(f"{path}_feature_scaler.npy", self.feature_scaler.data_range_)
        np.save(f"{path}_feature_min.npy", self.feature_scaler.data_min_)
        np.save(f"{path}_target_scaler.npy", self.target_scaler.data_range_)
        np.save(f"{path}_target_min.npy", self.target_scaler.data_min_)
    
    def _load_model_impl(self, path):
        """
        Implementation to load a Keras model.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            tf.keras.Model: The loaded model
        """
        # Load model
        model = load_model(f"{path}.h5")
        self.model = model
        
        # Load scalers
        data_range = np.load(f"{path}_feature_scaler.npy")
        data_min = np.load(f"{path}_feature_min.npy")
        self.feature_scaler.data_range_ = data_range
        self.feature_scaler.data_min_ = data_min
        self.feature_scaler.scale_ = 1.0 / data_range
        
        data_range = np.load(f"{path}_target_scaler.npy")
        data_min = np.load(f"{path}_target_min.npy")
        self.target_scaler.data_range_ = data_range
        self.target_scaler.data_min_ = data_min
        self.target_scaler.scale_ = 1.0 / data_range
        
        return model 