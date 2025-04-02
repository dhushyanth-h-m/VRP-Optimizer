#!/usr/bin/env python3
"""
VRP Optimizer: Main script for running the Vehicle Routing Problem optimization.
"""

import argparse
import os
import logging
import time
from datetime import datetime

from data.data_loader import DataLoader
from models.alns import ALNS
from utils.config import Config
from utils.solution import Solution
from visualization.visualizer import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vrp_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VRP Optimizer')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                        help='Path to data file')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations for the ALNS algorithm')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save results')
    return parser.parse_args()

def main():
    """Main function to run the VRP optimization."""
    # Start timing
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config(args.config)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data_loader = DataLoader(args.data)
    customers, vehicles, depot = data_loader.load()
    
    # Initialize solution
    logger.info("Initializing solution")
    initial_solution = Solution(customers, vehicles, depot)
    initial_solution.generate_initial_solution()
    
    # Run ALNS algorithm
    logger.info(f"Running ALNS algorithm for {args.iterations} iterations")
    alns = ALNS(initial_solution, config)
    best_solution = alns.solve(args.iterations)
    
    # Calculate metrics
    logger.info("Calculating metrics")
    total_distance = best_solution.calculate_total_distance()
    total_time = best_solution.calculate_total_time()
    fuel_consumption = best_solution.calculate_fuel_consumption()
    
    # Print results
    logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Total distance: {total_distance:.2f} km")
    logger.info(f"Total time: {total_time:.2f} hours")
    logger.info(f"Estimated fuel consumption: {fuel_consumption:.2f} liters")
    
    # Visualize results
    logger.info("Generating visualizations")
    visualizer = Visualizer(best_solution)
    visualizer.plot_routes(f"{args.output}/routes.html")
    visualizer.create_kpi_dashboard(f"{args.output}/kpi_dashboard.html")
    
    logger.info(f"Results saved to {args.output}/")

if __name__ == "__main__":
    main() 