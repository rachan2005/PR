#!/usr/bin/env python
"""
Run script for the traffic management simulation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from models.simulation import TrafficSimulation
from visualization.visualizer import visualize_simulation_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run traffic management simulation')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Configuration file path')
    parser.add_argument('--duration', type=int, default=None,
                        help='Simulation duration in seconds (overrides config)')
    parser.add_argument('--vehicles', type=int, default=None,
                        help='Number of vehicles (overrides config)')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results after simulation')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    return parser.parse_args()

def setup_logging(log_level):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging
    log_filename = f"logs/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def run_simulation(config_file, duration=None, vehicles=None, random_seed=None, visualize=False):
    """
    Run the traffic management simulation.
    
    Args:
        config_file (str): Configuration file path
        duration (int, optional): Simulation duration in seconds
        vehicles (int, optional): Number of vehicles
        random_seed (int, optional): Random seed
        visualize (bool): Whether to visualize results
        
    Returns:
        dict: Simulation results
    """
    # Initialize simulation
    simulation = TrafficSimulation(config_file)
    
    # Override config parameters if provided
    if duration is not None:
        simulation.duration = duration
    if vehicles is not None:
        simulation.vehicle_count = vehicles
    if random_seed is not None:
        simulation.random_seed = random_seed
        
    logging.info(f"Starting simulation with {simulation.vehicle_count} vehicles for {simulation.duration} seconds")
    
    # Initialize and run simulation
    if simulation.initialize():
        results = simulation.run()
        
        # Visualize results if requested
        if visualize and results:
            try:
                # Create visualization directory if it doesn't exist
                os.makedirs('visualization/results', exist_ok=True)
                
                # Generate visualizations
                viz_file = f"visualization/results/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                visualize_simulation_results(results, save_path=viz_file)
                logging.info(f"Saved visualization to {viz_file}")
            except Exception as e:
                logging.error(f"Error visualizing results: {str(e)}")
        
        return results
    else:
        logging.error("Failed to initialize simulation")
        return None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Run simulation
    results = run_simulation(
        args.config,
        duration=args.duration,
        vehicles=args.vehicles,
        random_seed=args.random_seed,
        visualize=args.visualize
    )
    
    if results:
        logging.info("Simulation completed successfully")
        
        # Print summary results
        print("\nSimulation Results Summary:")
        print(f"Simulation time: {results['simulation_time']} seconds")
        print(f"Vehicle count: {results['vehicle_count']}")
        print(f"Completed trips: {results['completed_trips']}/{results['vehicle_count']} ({results['completed_trips']/results['vehicle_count']*100:.1f}%)")
        print(f"Average travel time: {results['avg_travel_time']:.1f} seconds")
        print(f"Average waiting time: {results['avg_waiting_time']:.1f} seconds")
        
        return 0
    else:
        logging.error("Simulation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 