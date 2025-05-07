#!/usr/bin/env python
"""
Data processing utilities for the traffic management system.
Provides functions for loading, cleaning, and transforming data.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Union, Optional
import pickle

logger = logging.getLogger(__name__)

def load_csv_data(file_path: str, date_column: Optional[str] = None, 
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
    """
    Load and process CSV data.
    
    Args:
        file_path (str): Path to the CSV file
        date_column (str, optional): Name of the date/timestamp column
        timestamp_format (str): Format of the timestamp
        
    Returns:
        pd.DataFrame: Loaded and processed dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse date column if specified
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format=timestamp_format)
            df = df.sort_values(by=date_column)
        
        logger.info(f"Loaded CSV data from {file_path}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data from {file_path}: {str(e)}")
        return pd.DataFrame()

def load_json_data(file_path: str) -> Dict:
    """
    Load and process JSON data.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded and processed data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {str(e)}")
        return {}

def save_json_data(data: Dict, file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str): Path to save the JSON file
        
    Returns:
        bool: Whether the data was saved successfully
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved JSON data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {str(e)}")
        return False

def save_dataframe(df: pd.DataFrame, file_path: str, file_format: str = 'csv') -> bool:
    """
    Save a dataframe to a file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        file_path (str): Path to save the file
        file_format (str): Format to save the dataframe (csv, parquet, pickle)
        
    Returns:
        bool: Whether the dataframe was saved successfully
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        elif file_format.lower() == 'pickle':
            df.to_pickle(file_path)
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return False
        
        logger.info(f"Saved dataframe to {file_path} in {file_format} format")
        return True
    except Exception as e:
        logger.error(f"Error saving dataframe to {file_path}: {str(e)}")
        return False

def clean_traffic_data(df: pd.DataFrame, volume_col: str = 'volume', 
                      speed_col: str = 'speed', occupancy_col: str = 'occupancy') -> pd.DataFrame:
    """
    Clean traffic sensor data.
    
    Args:
        df (pd.DataFrame): Dataframe with traffic data
        volume_col (str): Name of the volume column
        speed_col (str): Name of the speed column
        occupancy_col (str): Name of the occupancy column
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Remove rows with missing values in key columns
        if volume_col in df_clean.columns:
            df_clean = df_clean[df_clean[volume_col].notna()]
        if speed_col in df_clean.columns:
            df_clean = df_clean[df_clean[speed_col].notna()]
        if occupancy_col in df_clean.columns:
            df_clean = df_clean[df_clean[occupancy_col].notna()]
        
        # Remove outliers in volume (negative or extremely high values)
        if volume_col in df_clean.columns:
            q_hi = df_clean[volume_col].quantile(0.99)
            df_clean = df_clean[(df_clean[volume_col] >= 0) & (df_clean[volume_col] <= q_hi)]
        
        # Remove outliers in speed (negative or extremely high values)
        if speed_col in df_clean.columns:
            df_clean = df_clean[(df_clean[speed_col] >= 0) & (df_clean[speed_col] <= 120)]
        
        # Remove outliers in occupancy (outside 0-100%)
        if occupancy_col in df_clean.columns:
            df_clean = df_clean[(df_clean[occupancy_col] >= 0) & (df_clean[occupancy_col] <= 100)]
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        logger.info(f"Cleaned traffic data: {len(df)} -> {len(df_clean)} rows")
        return df_clean
    except Exception as e:
        logger.error(f"Error cleaning traffic data: {str(e)}")
        return df

def resample_traffic_data(df: pd.DataFrame, date_col: str, 
                         freq: str = '5min', agg_dict: Optional[Dict] = None) -> pd.DataFrame:
    """
    Resample time series traffic data to a different frequency.
    
    Args:
        df (pd.DataFrame): Dataframe with traffic data
        date_col (str): Name of the datetime column
        freq (str): Resampling frequency (e.g., '5min', '1h')
        agg_dict (dict, optional): Dictionary mapping columns to aggregation functions
        
    Returns:
        pd.DataFrame: Resampled dataframe
    """
    try:
        # Make sure date column is datetime
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Set date column as index
        df_resampled = df.set_index(date_col)
        
        # Default aggregation functions if none provided
        if agg_dict is None:
            agg_dict = {
                'volume': 'sum',
                'speed': 'mean',
                'occupancy': 'mean'
            }
        
        # Only use columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_resampled.columns}
        
        # Resample
        df_resampled = df_resampled.resample(freq).agg(agg_dict)
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        logger.info(f"Resampled traffic data to {freq} frequency: {len(df)} -> {len(df_resampled)} rows")
        return df_resampled
    except Exception as e:
        logger.error(f"Error resampling traffic data: {str(e)}")
        return df

def calculate_congestion(df: pd.DataFrame, speed_col: str = 'speed', 
                        free_flow_speed: float = 60.0) -> pd.DataFrame:
    """
    Calculate congestion level based on speed.
    
    Args:
        df (pd.DataFrame): Dataframe with traffic data
        speed_col (str): Name of the speed column
        free_flow_speed (float): Free flow speed (km/h)
        
    Returns:
        pd.DataFrame: Dataframe with added congestion column
    """
    try:
        df_result = df.copy()
        
        # Calculate congestion (0 = free flow, 1 = full congestion)
        if speed_col in df_result.columns:
            # Ensure speed is not zero to avoid division by zero
            df_result[speed_col] = df_result[speed_col].clip(lower=1.0)
            
            # Calculate congestion as 1 - (speed / free_flow_speed), clipped to [0, 1]
            df_result['congestion'] = (1 - (df_result[speed_col] / free_flow_speed)).clip(0, 1)
        
        logger.info(f"Calculated congestion levels for {len(df)} rows")
        return df_result
    except Exception as e:
        logger.error(f"Error calculating congestion: {str(e)}")
        return df

def prepare_lstm_sequences(df: pd.DataFrame, feature_cols: List[str], 
                         target_cols: List[str], sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM model.
    
    Args:
        df (pd.DataFrame): Dataframe with time series data
        feature_cols (list): List of feature column names
        target_cols (list): List of target column names
        sequence_length (int): Length of input sequences
        
    Returns:
        tuple: (X, y) where X is input sequences and y is targets
    """
    try:
        # Extract features and targets
        data = df[feature_cols].values
        targets = df[target_cols].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        
        logger.info(f"Prepared {len(X)} sequences with length {sequence_length}")
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error preparing LSTM sequences: {str(e)}")
        return np.array([]), np.array([])

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx = int(n * (1 - test_size - val_size))
        
        # Split data
        X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
        y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]
        
        logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

def normalize_data(train_data: np.ndarray, val_data: Optional[np.ndarray] = None, 
                 test_data: Optional[np.ndarray] = None) -> Tuple:
    """
    Normalize data using Min-Max scaling.
    
    Args:
        train_data (numpy.ndarray): Training data
        val_data (numpy.ndarray, optional): Validation data
        test_data (numpy.ndarray, optional): Test data
        
    Returns:
        tuple: (normalized_train, normalized_val, normalized_test, min_vals, max_vals)
    """
    try:
        # Calculate min and max from training data
        min_vals = np.min(train_data, axis=0)
        max_vals = np.max(train_data, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        # Normalize data
        normalized_train = (train_data - min_vals) / range_vals
        
        normalized_val = None
        if val_data is not None:
            normalized_val = (val_data - min_vals) / range_vals
        
        normalized_test = None
        if test_data is not None:
            normalized_test = (test_data - min_vals) / range_vals
        
        logger.info(f"Normalized data with min={min_vals} and max={max_vals}")
        return normalized_train, normalized_val, normalized_test, min_vals, max_vals
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return None, None, None, None, None

def denormalize_data(normalized_data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Denormalize data (inverse of Min-Max scaling).
    
    Args:
        normalized_data (numpy.ndarray): Normalized data
        min_vals (numpy.ndarray): Minimum values used for normalization
        max_vals (numpy.ndarray): Maximum values used for normalization
        
    Returns:
        numpy.ndarray: Denormalized data
    """
    try:
        range_vals = max_vals - min_vals
        denormalized_data = normalized_data * range_vals + min_vals
        return denormalized_data
    except Exception as e:
        logger.error(f"Error denormalizing data: {str(e)}")
        return normalized_data

def save_preprocessor(min_vals: np.ndarray, max_vals: np.ndarray, file_path: str) -> bool:
    """
    Save preprocessing parameters.
    
    Args:
        min_vals (numpy.ndarray): Minimum values
        max_vals (numpy.ndarray): Maximum values
        file_path (str): Path to save the preprocessor
        
    Returns:
        bool: Whether the preprocessor was saved successfully
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({'min_vals': min_vals, 'max_vals': max_vals}, f)
        
        logger.info(f"Saved preprocessor to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving preprocessor: {str(e)}")
        return False

def load_preprocessor(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessing parameters.
    
    Args:
        file_path (str): Path to the saved preprocessor
        
    Returns:
        tuple: (min_vals, max_vals)
    """
    try:
        with open(file_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        min_vals = preprocessor.get('min_vals')
        max_vals = preprocessor.get('max_vals')
        
        logger.info(f"Loaded preprocessor from {file_path}")
        return min_vals, max_vals
    except Exception as e:
        logger.error(f"Error loading preprocessor from {file_path}: {str(e)}")
        return None, None

def generate_synthetic_traffic_data(num_segments: int = 10, days: int = 7, 
                                  interval_minutes: int = 5) -> pd.DataFrame:
    """
    Generate synthetic traffic data for testing.
    
    Args:
        num_segments (int): Number of road segments
        days (int): Number of days of data
        interval_minutes (int): Interval between measurements in minutes
        
    Returns:
        pd.DataFrame: Synthetic traffic data
    """
    try:
        # Calculate number of intervals
        intervals_per_day = 24 * 60 // interval_minutes
        total_intervals = intervals_per_day * days
        
        # Create timestamps
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
        timestamps = [start_date + timedelta(minutes=i * interval_minutes) for i in range(total_intervals)]
        
        # Create road segment IDs
        segments = [f"segment_{i}" for i in range(num_segments)]
        
        # Create dataframe
        data = []
        
        for segment in segments:
            # Base parameters for this segment
            base_volume = np.random.randint(50, 200)
            base_speed = np.random.randint(40, 70)
            
            for ts in timestamps:
                # Time-based patterns
                hour = ts.hour
                day_of_week = ts.weekday()
                
                # Morning rush hour pattern (7-9 AM)
                morning_rush = (hour >= 7 and hour < 9)
                
                # Evening rush hour pattern (4-6 PM)
                evening_rush = (hour >= 16 and hour < 18)
                
                # Weekend pattern
                is_weekend = (day_of_week >= 5)  # Saturday or Sunday
                
                # Calculate volume
                volume_factor = 1.0
                if morning_rush:
                    volume_factor = 2.0 if not is_weekend else 1.2
                elif evening_rush:
                    volume_factor = 1.8 if not is_weekend else 1.3
                elif is_weekend:
                    volume_factor = 0.7
                
                # Add some randomness
                volume_factor *= np.random.uniform(0.9, 1.1)
                
                volume = int(base_volume * volume_factor)
                
                # Calculate speed (inversely related to volume)
                speed_factor = 2 - volume_factor  # Speeds decrease as volume increases
                speed_factor = max(0.5, min(1.2, speed_factor))  # Clamp between 0.5 and 1.2
                
                # Add some randomness
                speed_factor *= np.random.uniform(0.95, 1.05)
                
                speed = base_speed * speed_factor
                
                # Calculate occupancy (roughly proportional to volume/speed)
                occupancy = min(100, max(0, (volume / speed) * 2))
                
                # Add some randomness
                occupancy *= np.random.uniform(0.9, 1.1)
                
                # Add to data
                data.append({
                    'timestamp': ts,
                    'segment_id': segment,
                    'volume': volume,
                    'speed': speed,
                    'occupancy': occupancy
                })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Calculate congestion
        df = calculate_congestion(df, speed_col='speed', free_flow_speed=70.0)
        
        logger.info(f"Generated synthetic traffic data with {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        return pd.DataFrame() 