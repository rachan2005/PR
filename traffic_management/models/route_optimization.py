#!/usr/bin/env python
"""
Route optimization model for traffic management.
Implements various algorithms for finding optimal routes.
"""

import os
import numpy as np
import pickle
import networkx as nx
import logging
import heapq
import json
from collections import defaultdict

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class RouteOptimizationModel(BaseModel):
    """
    Route optimization model for traffic management.
    Uses graph algorithms to find optimal routes based on current traffic conditions.
    """
    
    def __init__(self, config_path):
        """
        Initialize the route optimization model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        super().__init__(config_path, "route_optimization")
        
        # Model parameters from config
        self.algorithm = self.model_config.get("algorithm", "dijkstra")
        self.update_frequency = self.model_config.get("update_frequency", 60)
        self.max_alternatives = self.model_config.get("max_alternatives", 3)
        self.congestion_threshold = self.model_config.get("congestion_threshold", 0.7)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.last_update_time = 0
        
        logger.info(f"Route Optimization Model initialized with {self.algorithm} algorithm")
    
    def load_map(self, map_file):
        """
        Load map data from file and create graph.
        
        Args:
            map_file (str): Path to the map file
            
        Returns:
            bool: Whether the map was loaded successfully
        """
        try:
            # For now, assume the map file is a JSON with nodes and edges
            with open(map_file, 'r') as f:
                map_data = json.load(f)
            
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes
            for node in map_data.get('nodes', []):
                node_id = node['id']
                self.graph.add_node(
                    node_id,
                    pos=(node.get('lat', 0), node.get('lon', 0)),
                    is_intersection=node.get('is_intersection', False),
                    has_signals=node.get('has_signals', False)
                )
            
            # Add edges
            for edge in map_data.get('edges', []):
                source = edge['source']
                target = edge['target']
                length = edge.get('length', 1.0)
                max_speed = edge.get('max_speed', 50.0)
                lanes = edge.get('lanes', 1)
                
                # Calculate default travel time based on length and max speed
                default_time = length / max_speed
                
                self.graph.add_edge(
                    source, target,
                    length=length,
                    max_speed=max_speed,
                    lanes=lanes,
                    default_time=default_time,
                    current_time=default_time,
                    congestion=0.0
                )
            
            logger.info(f"Loaded map with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading map from {map_file}: {str(e)}")
            return False
    
    def update_traffic_data(self, traffic_data, timestamp):
        """
        Update the graph with current traffic data.
        
        Args:
            traffic_data (dict): Dictionary mapping edge IDs to traffic metrics
            timestamp (float): Current timestamp
            
        Returns:
            bool: Whether the graph was updated successfully
        """
        try:
            # Check if update is needed based on frequency
            if timestamp - self.last_update_time < self.update_frequency:
                logger.debug("Skipping update due to frequency limit")
                return False
            
            # Update edge weights based on traffic data
            for edge_id, data in traffic_data.items():
                if isinstance(edge_id, tuple) and len(edge_id) == 2:
                    source, target = edge_id
                    if self.graph.has_edge(source, target):
                        # Get current edge data
                        edge_data = self.graph.get_edge_data(source, target)
                        
                        # Update congestion level (0 to 1, where 1 is fully congested)
                        congestion = data.get('congestion', 0.0)
                        
                        # Update travel time based on congestion
                        default_time = edge_data.get('default_time', 1.0)
                        # Congestion factor: 1.0 means default time, 5.0 means 5x slower at max congestion
                        congestion_factor = 1.0 + 4.0 * congestion
                        current_time = default_time * congestion_factor
                        
                        # Update edge attributes
                        self.graph[source][target].update({
                            'current_time': current_time,
                            'congestion': congestion,
                            'volume': data.get('volume', 0),
                            'speed': data.get('speed', edge_data.get('max_speed', 50.0)),
                            'last_updated': timestamp
                        })
            
            self.last_update_time = timestamp
            logger.info(f"Updated graph with traffic data at timestamp {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating traffic data: {str(e)}")
            return False
    
    def find_shortest_path(self, origin, destination, weight='current_time'):
        """
        Find the shortest path between origin and destination.
        
        Args:
            origin (str): ID of the origin node
            destination (str): ID of the destination node
            weight (str): Edge attribute to use as weight
            
        Returns:
            tuple: (path, total_weight) where path is a list of node IDs
        """
        try:
            # Check if nodes exist
            if not self.graph.has_node(origin) or not self.graph.has_node(destination):
                logger.error(f"Origin or destination node not found: {origin} -> {destination}")
                return None, None
            
            # Find shortest path
            if self.algorithm == 'dijkstra':
                path = nx.dijkstra_path(self.graph, origin, destination, weight=weight)
                length = nx.dijkstra_path_length(self.graph, origin, destination, weight=weight)
            elif self.algorithm == 'astar':
                path = nx.astar_path(self.graph, origin, destination, weight=weight)
                length = nx.astar_path_length(self.graph, origin, destination, weight=weight)
            else:
                # Default to Dijkstra
                path = nx.dijkstra_path(self.graph, origin, destination, weight=weight)
                length = nx.dijkstra_path_length(self.graph, origin, destination, weight=weight)
            
            logger.info(f"Found path from {origin} to {destination} with length {length:.2f}")
            return path, length
            
        except Exception as e:
            logger.error(f"Error finding path from {origin} to {destination}: {str(e)}")
            return None, None
    
    def find_alternative_paths(self, origin, destination, num_paths=3, weight='current_time'):
        """
        Find multiple alternative paths between origin and destination.
        
        Args:
            origin (str): ID of the origin node
            destination (str): ID of the destination node
            num_paths (int): Number of alternative paths to find
            weight (str): Edge attribute to use as weight
            
        Returns:
            list: List of (path, total_weight) tuples
        """
        try:
            # First find the shortest path
            shortest_path, shortest_length = self.find_shortest_path(origin, destination, weight)
            if shortest_path is None:
                return []
            
            paths = [(shortest_path, shortest_length)]
            
            # Make a copy of the graph to modify for finding alternative paths
            temp_graph = self.graph.copy()
            
            # Find additional paths by penalizing edges in previous paths
            for i in range(1, num_paths):
                # Penalize edges in previous paths
                for path, _ in paths:
                    for j in range(len(path) - 1):
                        source, target = path[j], path[j + 1]
                        if temp_graph.has_edge(source, target):
                            # Increase the weight of this edge
                            current_weight = temp_graph.get_edge_data(source, target).get(weight, 1.0)
                            temp_graph[source][target][weight] = current_weight * 1.5
                
                # Find the next best path in the modified graph
                try:
                    if self.algorithm == 'dijkstra':
                        path = nx.dijkstra_path(temp_graph, origin, destination, weight=weight)
                        length = nx.dijkstra_path_length(temp_graph, origin, destination, weight=weight)
                    elif self.algorithm == 'astar':
                        path = nx.astar_path(temp_graph, origin, destination, weight=weight)
                        length = nx.astar_path_length(temp_graph, origin, destination, weight=weight)
                    else:
                        path = nx.dijkstra_path(temp_graph, origin, destination, weight=weight)
                        length = nx.dijkstra_path_length(temp_graph, origin, destination, weight=weight)
                    
                    # Check if this path is different enough from previous paths
                    is_different = True
                    for prev_path, _ in paths:
                        # Calculate path similarity (Jaccard index of edges)
                        prev_edges = set(zip(prev_path[:-1], prev_path[1:]))
                        current_edges = set(zip(path[:-1], path[1:]))
                        
                        if len(prev_edges) == 0 and len(current_edges) == 0:
                            similarity = 1.0
                        else:
                            intersection = len(prev_edges.intersection(current_edges))
                            union = len(prev_edges.union(current_edges))
                            similarity = intersection / union
                        
                        if similarity > 0.7:  # If more than 70% similar, consider it too similar
                            is_different = False
                            break
                    
                    if is_different:
                        paths.append((path, length))
                
                except:
                    # If we can't find another path, stop
                    break
            
            # Sort paths by length
            paths.sort(key=lambda x: x[1])
            
            logger.info(f"Found {len(paths)} alternative paths from {origin} to {destination}")
            return paths[:num_paths]
            
        except Exception as e:
            logger.error(f"Error finding alternative paths from {origin} to {destination}: {str(e)}")
            return []
    
    def train(self, data):
        """
        'Train' the route optimization model. 
        For graph-based models, this could involve learning traffic patterns.
        
        Args:
            data: Training data
            
        Returns:
            dict: Training metrics or empty dict if not applicable
        """
        # For simple graph algorithms, training might not be applicable
        # This method could be expanded for learning-based approaches
        logger.info("Route optimization model does not require traditional training")
        return {}
    
    def predict(self, data):
        """
        Generate route predictions for the given data.
        
        Args:
            data (dict): Dictionary with 'origin' and 'destination' keys
            
        Returns:
            list: List of recommended routes
        """
        origin = data.get('origin')
        destination = data.get('destination')
        
        if not origin or not destination:
            logger.error("Origin and destination must be provided")
            return []
        
        # Find alternative paths
        paths = self.find_alternative_paths(
            origin, 
            destination, 
            num_paths=self.max_alternatives
        )
        
        # Format the result
        routes = []
        for path, length in paths:
            # Calculate additional metrics for this route
            segments = []
            total_distance = 0
            total_congestion = 0
            
            for i in range(len(path) - 1):
                source, target = path[i], path[i + 1]
                edge_data = self.graph.get_edge_data(source, target)
                distance = edge_data.get('length', 0)
                time = edge_data.get('current_time', 0)
                congestion = edge_data.get('congestion', 0)
                
                segments.append({
                    'from': source,
                    'to': target,
                    'distance': distance,
                    'time': time,
                    'congestion': congestion
                })
                
                total_distance += distance
                total_congestion += congestion * distance  # Weight by segment length
            
            # Calculate average congestion weighted by segment length
            avg_congestion = total_congestion / total_distance if total_distance > 0 else 0
            
            routes.append({
                'path': path,
                'segments': segments,
                'total_time': length,
                'total_distance': total_distance,
                'avg_congestion': avg_congestion
            })
        
        logger.info(f"Generated {len(routes)} route options from {origin} to {destination}")
        return routes
    
    def evaluate(self, data, targets):
        """
        Evaluate the route optimization model.
        
        Args:
            data: Input data
            targets: Ground truth
            
        Returns:
            dict: Evaluation metrics
        """
        # For route optimization, evaluation might involve comparing against ground truth routes
        # or historical travel times
        if not data or not targets:
            return {}
        
        times_pred = []
        times_actual = []
        
        for query, actual in zip(data, targets):
            origin = query.get('origin')
            destination = query.get('destination')
            
            # Get predicted best route
            routes = self.predict({'origin': origin, 'destination': destination})
            if routes:
                best_route = routes[0]
                times_pred.append(best_route['total_time'])
                
                # Get actual travel time from ground truth
                times_actual.append(actual.get('total_time', 0))
        
        if not times_pred or not times_actual:
            return {}
        
        # Calculate error metrics
        errors = np.array(times_pred) - np.array(times_actual)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        mape = np.mean(np.abs(errors / np.array(times_actual))) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Route optimization evaluation: MAE={mae:.2f}s, RMSE={rmse:.2f}s, MAPE={mape:.2f}%")
        return metrics
    
    def _save_model_impl(self, model, path):
        """
        Implementation to save the route optimization model.
        Mostly saves the graph and configuration.
        
        Args:
            model: The model to save (not used)
            path (str): Path to save the model to
        """
        # Save graph as pickle
        with open(f"{path}_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'update_frequency': self.update_frequency,
            'max_alternatives': self.max_alternatives,
            'congestion_threshold': self.congestion_threshold,
            'last_update_time': self.last_update_time
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_model_impl(self, path):
        """
        Implementation to load a route optimization model.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            The loaded model (self in this case)
        """
        # Load graph
        graph_path = f"{path}_graph.pkl"
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
        
        # Load metadata
        metadata_path = f"{path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.algorithm = metadata.get('algorithm', self.algorithm)
                self.update_frequency = metadata.get('update_frequency', self.update_frequency)
                self.max_alternatives = metadata.get('max_alternatives', self.max_alternatives)
                self.congestion_threshold = metadata.get('congestion_threshold', self.congestion_threshold)
                self.last_update_time = metadata.get('last_update_time', 0)
        
        return self 