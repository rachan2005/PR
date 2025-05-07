#!/usr/bin/env python
"""
Run script for the traffic management API.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from api.api import start_api_server

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run traffic management API server')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Configuration file path')
    parser.add_argument('--host', type=str, default=None,
                        help='API host (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                        help='API port (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (enables auto-reload)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    return parser.parse_args()

def setup_logging(log_level):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging
    log_filename = f"logs/api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set configuration in environment variables
    if args.config:
        os.environ['API_CONFIG_PATH'] = args.config
    if args.host:
        os.environ['API_HOST'] = args.host
    if args.port:
        os.environ['API_PORT'] = str(args.port)
    if args.debug:
        os.environ['API_DEBUG'] = 'true'
    
    logging.info("Starting API server")
    
    try:
        # Run the API server
        start_api_server()
        return 0
    except Exception as e:
        logging.error(f"Error starting API server: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 