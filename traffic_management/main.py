#!/usr/bin/env python
"""
Main entry point for the AI-Powered Urban Traffic Management System.
This script orchestrates the different components of the system.
"""

import os
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/traffic_management_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-Powered Urban Traffic Management System')
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'production', 'evaluation'],
                        help='Operation mode: simulation, production, or evaluation')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main function to run the traffic management system."""
    # Parse arguments
    args = parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting Traffic Management System in {args.mode} mode")
    logger.info(f"Using configuration file: {args.config}")
    
    try:
        # Import system components based on mode
        if args.mode == 'simulation':
            from models.simulation import TrafficSimulation
            system = TrafficSimulation(config_file=args.config)
        elif args.mode == 'production':
            from models.production import ProductionSystem
            system = ProductionSystem(config_file=args.config)
        elif args.mode == 'evaluation':
            from models.evaluation import EvaluationSystem
            system = EvaluationSystem(config_file=args.config)
        
        # Run the system
        logger.info("Initializing system...")
        system.initialize()
        
        logger.info("Starting system operation...")
        system.run()
        
        logger.info("System operation completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 