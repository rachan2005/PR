#!/usr/bin/env python
"""
Traffic simulation module for testing and evaluating the traffic management system.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from models.traffic_prediction import TrafficPredictionModel
from models.signal_control import SignalControlModel
from models.route_optimization import RouteOptimizationModel

logger = logging.getLogger(__name__)

class TrafficSimulation:
    """
    Traffic simulation environment for testing and evaluating the traffic management system.
    Integrates traffic prediction, signal control, and route optimization models.
    """
    
    def __init__(self, config_file):
        """
        Initialize the traffic simulation.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Simulation parameters
        sim_config = self.config.get('simulation', {})
        self.duration = sim_config.get('duration', 3600)  # seconds
        self.time_step = sim_config.get('time_step', 1.0)  # seconds
        self.map_file = sim_config.get('map_file', 'data/maps/city_network.osm')
        self.vehicle_count = sim_config.get('vehicle_count', 1000)
        self.random_seed = sim_config.get('random_seed', 42)
        
        # Intersection parameters
        intersections_config = sim_config.get('intersections', {})
        self.intersection_count = intersections_config.get('count', 20)
        self.intersections_file = intersections_config.get('file', 'data/simulation/intersections.json')
        
        # Visualization parameters
        viz_config = sim_config.get('visualization', {})
        self.viz_enabled = viz_config.get('enabled', True)
        self.viz_update_freq = viz_config.get('update_frequency', 1.0)
        self.viz_save_video = viz_config.get('save_video', True)
        self.viz_video_path = viz_config.get('video_path', 'visualization/simulation_video.mp4')
        
        # Initialize models
        self.prediction_model = None
        self.signal_control_model = None
        self.route_optimization_model = None
        
        # Simulation state
        self.current_time = 0
        self.vehicles = {}
        self.traffic_data = {}
        self.intersections = {}
        self.signals = {}
        
        # Statistics
        self.stats = {
            'avg_travel_time': [],
            'avg_waiting_time': [],
            'throughput': [],
            'congestion': []
        }
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        logger.info("Traffic Simulation initialized")
    
    def _load_config(self):
        """
        Load configuration from a JSON file.
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_file}: {str(e)}")
            return {}
    
    def initialize(self):
        """
        Initialize the simulation environment, models, and data.
        
        Returns:
            bool: Whether initialization was successful
        """
        logger.info("Initializing simulation environment")
        
        try:
            # Create necessary directories
            os.makedirs('data/maps', exist_ok=True)
            os.makedirs('data/simulation', exist_ok=True)
            os.makedirs('visualization', exist_ok=True)
            
            # Initialize prediction model
            self.prediction_model = TrafficPredictionModel(self.config_file)
            
            # Initialize signal control model
            self.signal_control_model = SignalControlModel(self.config_file)
            
            # Initialize route optimization model
            self.route_optimization_model = RouteOptimizationModel(self.config_file)
            
            # Load or generate map if not exists
            if not os.path.exists(self.map_file):
                self._generate_sample_map()
            
            # Load map into route optimization model
            self.route_optimization_model.load_map(self.map_file)
            
            # Load or generate intersections
            if not os.path.exists(self.intersections_file):
                self._generate_sample_intersections()
            
            self._load_intersections()
            
            # Initialize vehicles
            self._initialize_vehicles()
            
            # Initialize signals
            self._initialize_signals()
            
            logger.info("Simulation environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing simulation environment: {str(e)}")
            return False
    
    def _generate_sample_map(self):
        """Generate a sample map for simulation."""
        logger.info("Generating sample map")
        
        # Create a simple grid network
        grid_size = 5  # 5x5 grid
        nodes = []
        edges = []
        
        # Create nodes
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = f"n_{i}_{j}"
                nodes.append({
                    'id': node_id,
                    'lat': i,
                    'lon': j,
                    'is_intersection': True,
                    'has_signals': (i > 0 and i < grid_size - 1 and j > 0 and j < grid_size - 1)
                })
        
        # Create edges (roads)
        for i in range(grid_size):
            for j in range(grid_size):
                source_id = f"n_{i}_{j}"
                
                # Add edge to the right
                if j < grid_size - 1:
                    target_id = f"n_{i}_{j+1}"
                    edges.append({
                        'source': source_id,
                        'target': target_id,
                        'length': 1.0,
                        'max_speed': 50.0,
                        'lanes': 2
                    })
                    # Add reverse direction
                    edges.append({
                        'source': target_id,
                        'target': source_id,
                        'length': 1.0,
                        'max_speed': 50.0,
                        'lanes': 2
                    })
                
                # Add edge down
                if i < grid_size - 1:
                    target_id = f"n_{i+1}_{j}"
                    edges.append({
                        'source': source_id,
                        'target': target_id,
                        'length': 1.0,
                        'max_speed': 50.0,
                        'lanes': 2
                    })
                    # Add reverse direction
                    edges.append({
                        'source': target_id,
                        'target': source_id,
                        'length': 1.0,
                        'max_speed': 50.0,
                        'lanes': 2
                    })
        
        # Save map to file
        map_data = {
            'nodes': nodes,
            'edges': edges
        }
        
        os.makedirs(os.path.dirname(self.map_file), exist_ok=True)
        with open(self.map_file, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        logger.info(f"Sample map generated and saved to {self.map_file}")
    
    def _generate_sample_intersections(self):
        """Generate sample intersections for simulation."""
        logger.info("Generating sample intersections")
        
        # Create a simple set of intersections
        intersections = []
        grid_size = 5  # 5x5 grid
        
        # Create intersections for inner nodes of the grid
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                intersection_id = f"n_{i}_{j}"
                
                # Define the incoming roads
                incoming_roads = [
                    {'from': f"n_{i-1}_{j}", 'to': intersection_id},
                    {'from': f"n_{i+1}_{j}", 'to': intersection_id},
                    {'from': f"n_{i}_{j-1}", 'to': intersection_id},
                    {'from': f"n_{i}_{j+1}", 'to': intersection_id}
                ]
                
                # Define the phases (signal timings)
                phases = [
                    {
                        'id': 0,
                        'roads': [0, 2],  # North-South
                        'default_time': 30,
                        'min_time': 10,
                        'max_time': 60
                    },
                    {
                        'id': 1,
                        'roads': [1, 3],  # East-West
                        'default_time': 30,
                        'min_time': 10,
                        'max_time': 60
                    }
                ]
                
                intersections.append({
                    'id': intersection_id,
                    'incoming_roads': incoming_roads,
                    'phases': phases,
                    'cycle_time': 60
                })
        
        # Save intersections to file
        os.makedirs(os.path.dirname(self.intersections_file), exist_ok=True)
        with open(self.intersections_file, 'w') as f:
            json.dump(intersections, f, indent=2)
        
        logger.info(f"Sample intersections generated and saved to {self.intersections_file}")
    
    def _load_intersections(self):
        """Load intersections from file."""
        try:
            with open(self.intersections_file, 'r') as f:
                intersection_data = json.load(f)
            
            # Process intersections
            for intersection in intersection_data:
                self.intersections[intersection['id']] = intersection
            
            logger.info(f"Loaded {len(self.intersections)} intersections")
            
        except Exception as e:
            logger.error(f"Error loading intersections: {str(e)}")
    
    def _initialize_vehicles(self):
        """Initialize vehicles for simulation."""
        logger.info(f"Initializing {self.vehicle_count} vehicles")
        
        # Get all nodes from the route optimization model's graph
        all_nodes = list(self.route_optimization_model.graph.nodes())
        
        # Generate random vehicles
        for i in range(self.vehicle_count):
            # Generate random origin and destination
            origin, destination = np.random.choice(all_nodes, size=2, replace=False)
            
            # Generate random departure time (within the first half of simulation)
            departure_time = np.random.uniform(0, self.duration / 2)
            
            # Find initial route
            paths = self.route_optimization_model.find_alternative_paths(origin, destination)
            if not paths:
                continue  # Skip if no path found
                
            path, _ = paths[0]  # Take the shortest path
            
            # Create vehicle
            vehicle = {
                'id': f"v_{i}",
                'origin': origin,
                'destination': destination,
                'departure_time': departure_time,
                'arrival_time': None,
                'current_position': origin,
                'current_edge': None,
                'path': path,
                'progress': 0.0,
                'status': 'waiting',  # waiting, moving, arrived
                'total_waiting_time': 0.0,
                'current_wait_start': None
            }
            
            self.vehicles[vehicle['id']] = vehicle
        
        logger.info(f"Initialized {len(self.vehicles)} vehicles successfully")
    
    def _initialize_signals(self):
        """Initialize traffic signals."""
        logger.info("Initializing traffic signals")
        
        # Process each intersection
        for intersection_id, intersection in self.intersections.items():
            # Initial signal state: first phase is active
            active_phase = 0
            phase_time = 0
            
            self.signals[intersection_id] = {
                'active_phase': active_phase,
                'phase_time': phase_time,
                'phases': intersection['phases'],
                'cycle_time': intersection.get('cycle_time', 60)
            }
        
        logger.info(f"Initialized signals for {len(self.signals)} intersections")
    
    def run(self):
        """
        Run the simulation for the specified duration.
        
        Returns:
            dict: Simulation results
        """
        logger.info(f"Starting simulation for {self.duration} seconds")
        
        # Initialize visualization if enabled
        if self.viz_enabled:
            self._init_visualization()
        
        # Main simulation loop
        sim_start_time = time.time()
        while self.current_time < self.duration:
            # Update all vehicles
            self._update_vehicles()
            
            # Update traffic data
            self._update_traffic_data()
            
            # Update signal control (using reinforcement learning model)
            if self.current_time % 5 == 0:  # Update every 5 seconds
                self._update_signals()
            
            # Update route optimization (reroute vehicles if needed)
            if self.current_time % 60 == 0:  # Update every 60 seconds
                self._update_routes()
            
            # Update visualization
            if self.viz_enabled and self.current_time % (self.viz_update_freq / self.time_step) == 0:
                self._update_visualization()
            
            # Collect statistics
            self._collect_statistics()
            
            # Increment time
            self.current_time += self.time_step
            
            # Print progress every 10% of simulation
            if self.current_time % (self.duration / 10) < self.time_step:
                progress = (self.current_time / self.duration) * 100
                elapsed = time.time() - sim_start_time
                estimated_total = elapsed / (progress / 100)
                remaining = estimated_total - elapsed
                
                logger.info(f"Simulation progress: {progress:.1f}% - Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
        
        # Finish visualization
        if self.viz_enabled:
            self._finish_visualization()
        
        # Compile and return results
        results = self._compile_results()
        
        logger.info("Simulation completed")
        
        return results
    
    def _update_vehicles(self):
        """Update the state of all vehicles."""
        active_vehicles = 0
        waiting_vehicles = 0
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Skip if not yet departed or already arrived
            if vehicle['status'] == 'waiting' and vehicle['departure_time'] > self.current_time:
                continue
            elif vehicle['status'] == 'arrived':
                continue
            
            # If vehicle is waiting to depart and it's time, change status to moving
            if vehicle['status'] == 'waiting' and vehicle['departure_time'] <= self.current_time:
                vehicle['status'] = 'moving'
                vehicle['current_edge'] = (vehicle['path'][0], vehicle['path'][1])
                vehicle['progress'] = 0.0
            
            # If vehicle is moving, update its position
            if vehicle['status'] == 'moving':
                active_vehicles += 1
                
                # Get current edge data
                source, target = vehicle['current_edge']
                edge_data = self.route_optimization_model.graph.get_edge_data(source, target)
                
                # Check if vehicle is stopped at a red light
                is_stopped = self._is_vehicle_stopped_at_signal(vehicle)
                
                if is_stopped:
                    waiting_vehicles += 1
                    
                    # Update waiting time
                    if vehicle['current_wait_start'] is None:
                        vehicle['current_wait_start'] = self.current_time
                else:
                    # If was waiting but now moving, update total waiting time
                    if vehicle['current_wait_start'] is not None:
                        vehicle['total_waiting_time'] += (self.current_time - vehicle['current_wait_start'])
                        vehicle['current_wait_start'] = None
                    
                    # Calculate progress increment based on speed and time step
                    speed = edge_data.get('speed', edge_data.get('max_speed', 50.0))
                    edge_length = edge_data.get('length', 1.0)
                    
                    # Progress is normalized (0 to 1) along the edge
                    progress_increment = (speed * self.time_step) / edge_length
                    vehicle['progress'] += progress_increment
                
                # If reached the end of current edge
                if vehicle['progress'] >= 1.0:
                    # Find current position in path
                    current_index = vehicle['path'].index(target)
                    
                    # If reached destination
                    if current_index == len(vehicle['path']) - 1:
                        vehicle['status'] = 'arrived'
                        vehicle['arrival_time'] = self.current_time
                        logger.debug(f"Vehicle {vehicle_id} arrived at destination")
                    else:
                        # Move to next edge
                        next_node = vehicle['path'][current_index + 1]
                        vehicle['current_position'] = target
                        vehicle['current_edge'] = (target, next_node)
                        vehicle['progress'] = 0.0
        
        logger.debug(f"Active vehicles: {active_vehicles}, Waiting at signals: {waiting_vehicles}")
    
    def _is_vehicle_stopped_at_signal(self, vehicle):
        """
        Check if a vehicle is stopped at a traffic signal.
        
        Args:
            vehicle (dict): Vehicle data
            
        Returns:
            bool: Whether the vehicle is stopped
        """
        if vehicle['progress'] < 0.9:  # Only vehicles at the end of an edge can be stopped at a signal
            return False
        
        source, target = vehicle['current_edge']
        
        # Check if target node is an intersection with signals
        if target in self.intersections and target in self.signals:
            intersection = self.intersections[target]
            signal = self.signals[target]
            
            # Check if the current road is allowed in the active phase
            active_phase = signal['active_phase']
            phase_data = next((p for p in signal['phases'] if p['id'] == active_phase), None)
            
            if phase_data:
                # Find the index of the current road in the intersection's incoming roads
                road_index = next((i for i, r in enumerate(intersection['incoming_roads']) 
                                  if r['from'] == source and r['to'] == target), None)
                
                # Check if the road is allowed in the current phase
                if road_index is not None and road_index not in phase_data['roads']:
                    return True  # Vehicle is stopped at a red light
        
        return False
    
    def _update_traffic_data(self):
        """Update traffic data based on current vehicle positions."""
        traffic_data = {}
        
        # Group vehicles by edge
        edge_vehicles = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'moving' and vehicle['current_edge']:
                edge = vehicle['current_edge']
                if edge not in edge_vehicles:
                    edge_vehicles[edge] = []
                edge_vehicles[edge].append(vehicle)
        
        # Calculate traffic metrics for each edge
        for edge, vehicles in edge_vehicles.items():
            source, target = edge
            
            # Get edge data
            edge_data = self.route_optimization_model.graph.get_edge_data(source, target)
            edge_length = edge_data.get('length', 1.0)
            max_speed = edge_data.get('max_speed', 50.0)
            
            # Calculate metrics
            volume = len(vehicles)
            waiting_count = sum(1 for v in vehicles if v['current_wait_start'] is not None)
            speed = np.mean([edge_data.get('speed', max_speed) for v in vehicles]) if vehicles else max_speed
            
            # Calculate congestion (simplified)
            # 0 means free flow, 1 means completely congested
            lanes = edge_data.get('lanes', 1)
            capacity = lanes * 10  # Simplified capacity calculation
            congestion = min(1.0, volume / capacity) if capacity > 0 else 0.0
            
            # Further increase congestion if vehicles are waiting
            if waiting_count > 0:
                congestion = min(1.0, congestion + 0.2 * (waiting_count / volume))
            
            # Store traffic data
            traffic_data[edge] = {
                'volume': volume,
                'speed': speed,
                'congestion': congestion,
                'waiting_count': waiting_count
            }
        
        # Update the traffic data
        self.traffic_data = traffic_data
        
        # Update the route optimization model with this data
        self.route_optimization_model.update_traffic_data(traffic_data, self.current_time)
    
    def _update_signals(self):
        """Update traffic signals using the signal control model."""
        for intersection_id, signal in self.signals.items():
            # Get intersection data
            intersection = self.intersections.get(intersection_id)
            if not intersection:
                continue
            
            # Get current signal phase and time
            signal['phase_time'] += self.time_step
            
            # Check if it's time to change the phase (either through fixed timing or RL)
            current_phase = signal['phases'][signal['active_phase']]
            cycle_time = signal['cycle_time']
            
            if signal['phase_time'] >= current_phase.get('default_time', cycle_time / len(signal['phases'])):
                # Use RL model to decide next phase if available
                # For now, just use round-robin
                signal['active_phase'] = (signal['active_phase'] + 1) % len(signal['phases'])
                signal['phase_time'] = 0
                
                logger.debug(f"Intersection {intersection_id} changed to phase {signal['active_phase']}")
    
    def _update_routes(self):
        """Update vehicle routes based on current traffic conditions."""
        # For each active vehicle, consider rerouting
        rerouted_count = 0
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] != 'moving':
                continue
            
            # Only reroute some vehicles to avoid all vehicles taking the same route
            if np.random.random() > 0.3:  # 30% chance of rerouting
                continue
            
            # Get current position and destination
            current_pos = vehicle['current_position']
            destination = vehicle['destination']
            
            # Find current position in path
            current_index = vehicle['path'].index(current_pos)
            
            # Get alternative paths
            paths = self.route_optimization_model.find_alternative_paths(current_pos, destination)
            if not paths:
                continue
            
            # Check if there's a significantly better route
            current_remaining_path = vehicle['path'][current_index:]
            
            # Calculate remaining travel time on current path
            current_time = 0
            for i in range(len(current_remaining_path) - 1):
                source, target = current_remaining_path[i], current_remaining_path[i + 1]
                edge_data = self.route_optimization_model.graph.get_edge_data(source, target)
                current_time += edge_data.get('current_time', 1.0)
            
            # Get best alternative path
            best_path, best_time = paths[0]
            
            # Reroute if alternative is significantly better (>10% improvement)
            if best_time < 0.9 * current_time:
                vehicle['path'] = vehicle['path'][:current_index] + best_path
                rerouted_count += 1
                logger.debug(f"Rerouted vehicle {vehicle_id} at {current_pos}, expected improvement: {(current_time - best_time) / current_time * 100:.1f}%")
        
        if rerouted_count > 0:
            logger.info(f"Rerouted {rerouted_count} vehicles based on current traffic conditions")
    
    def _collect_statistics(self):
        """Collect simulation statistics."""
        if self.current_time % 60 == 0:  # Collect every minute
            # Calculate average travel time for completed trips
            completed_trips = [v for v in self.vehicles.values() if v['status'] == 'arrived']
            if completed_trips:
                avg_travel_time = np.mean([v['arrival_time'] - v['departure_time'] for v in completed_trips])
                self.stats['avg_travel_time'].append((self.current_time, avg_travel_time))
            
            # Calculate average waiting time
            active_vehicles = [v for v in self.vehicles.values() if v['status'] == 'moving']
            if active_vehicles:
                # For active vehicles, add current waiting time if currently waiting
                waiting_times = []
                for v in active_vehicles:
                    wait_time = v['total_waiting_time']
                    if v['current_wait_start'] is not None:
                        wait_time += (self.current_time - v['current_wait_start'])
                    waiting_times.append(wait_time)
                
                avg_waiting_time = np.mean(waiting_times)
                self.stats['avg_waiting_time'].append((self.current_time, avg_waiting_time))
            
            # Calculate throughput (vehicles arrived per minute)
            arrived_last_minute = sum(1 for v in self.vehicles.values() 
                                     if v['status'] == 'arrived' and 
                                     v['arrival_time'] > self.current_time - 60 and 
                                     v['arrival_time'] <= self.current_time)
            self.stats['throughput'].append((self.current_time, arrived_last_minute))
            
            # Calculate average congestion
            if self.traffic_data:
                avg_congestion = np.mean([data['congestion'] for data in self.traffic_data.values()])
                self.stats['congestion'].append((self.current_time, avg_congestion))
    
    def _init_visualization(self):
        """Initialize visualization."""
        logger.info("Initializing visualization")
        # In a real implementation, this would set up the visualization
        # For example, using matplotlib or a more specialized visualization library
    
    def _update_visualization(self):
        """Update visualization with current state."""
        # In a real implementation, this would update the visualization
        pass
    
    def _finish_visualization(self):
        """Finish visualization, save video if enabled."""
        logger.info("Finishing visualization")
        if self.viz_save_video:
            logger.info(f"Saving simulation video to {self.viz_video_path}")
            # In a real implementation, this would save the video
    
    def _compile_results(self):
        """
        Compile simulation results.
        
        Returns:
            dict: Simulation results
        """
        # Calculate final statistics
        completed_trips = [v for v in self.vehicles.values() if v['status'] == 'arrived']
        active_vehicles = [v for v in self.vehicles.values() if v['status'] == 'moving']
        waiting_vehicles = [v for v in self.vehicles.values() if v['status'] == 'waiting']
        
        # Calculate average travel time
        avg_travel_time = np.mean([v['arrival_time'] - v['departure_time'] for v in completed_trips]) if completed_trips else 0
        
        # Calculate average waiting time
        waiting_times = []
        for v in completed_trips + active_vehicles:
            wait_time = v['total_waiting_time']
            if v['current_wait_start'] is not None:
                wait_time += (self.current_time - v['current_wait_start'])
            waiting_times.append(wait_time)
        
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        
        # Calculate total throughput
        throughput = len(completed_trips)
        
        # Create results dictionary
        results = {
            'simulation_time': self.current_time,
            'vehicle_count': self.vehicle_count,
            'completed_trips': len(completed_trips),
            'active_vehicles': len(active_vehicles),
            'waiting_vehicles': len(waiting_vehicles),
            'avg_travel_time': avg_travel_time,
            'avg_waiting_time': avg_waiting_time,
            'throughput': throughput,
            'stats': self.stats
        }
        
        logger.info(f"Simulation results: {len(completed_trips)}/{self.vehicle_count} trips completed, avg travel time: {avg_travel_time:.1f}s")
        
        # Save results to file
        results_file = f"data/simulation/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        logger.info(f"Simulation results saved to {results_file}")
        
        return results 