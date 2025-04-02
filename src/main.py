#!/usr/bin/env python3
"""
VRP Optimizer: Main script for running the Vehicle Routing Problem optimization.
"""

import argparse
import os
import logging
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np

from data.data_loader import DataLoader
from models.alns import ALNS
from utils.config import Config
from utils.solution import Solution
from utils.analytics import Analytics
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
    parser.add_argument('--traffic', action='store_true',
                        help='Enable time-dependent travel speeds (traffic simulation)')
    parser.add_argument('--driver_hours', action='store_true',
                        help='Enable driver working hours constraints')
    parser.add_argument('--heterogeneous_fleet', action='store_true',
                        help='Use heterogeneous vehicle fleet with different characteristics')
    return parser.parse_args()

def configure_traffic_patterns(vehicles):
    """
    Configure time-dependent speed factors to simulate traffic patterns.
    
    Args:
        vehicles (list): List of vehicles to configure
    """
    # Define typical traffic patterns
    # Morning rush hour (7-9 AM): 60% of normal speed
    # Evening rush hour (4-7 PM): 50% of normal speed
    # Night (10 PM - 5 AM): 120% of normal speed (less traffic)
    # Rest of day: 80-100% of normal speed
    
    speed_factors = {
        0: 1.2, 1: 1.2, 2: 1.2, 3: 1.2, 4: 1.2, 5: 1.0,  # Night/early morning
        6: 0.8, 7: 0.6, 8: 0.6, 9: 0.7,  # Morning rush
        10: 0.8, 11: 0.9, 12: 0.8, 13: 0.9, 14: 0.9, 15: 0.8,  # Midday
        16: 0.5, 17: 0.5, 18: 0.5, 19: 0.7,  # Evening rush
        20: 0.8, 21: 0.9, 22: 1.1, 23: 1.2  # Evening/night
    }
    
    # Apply to all vehicles
    for vehicle in vehicles:
        vehicle.set_time_dependent_speed_factors(speed_factors)
        
    logger.info("Traffic patterns configured for all vehicles")

def configure_heterogeneous_fleet(vehicles):
    """
    Configure a heterogeneous fleet with different vehicle characteristics.
    
    Args:
        vehicles (list): List of vehicles to configure
        
    Returns:
        list: Updated list of vehicles
    """
    # Define different vehicle types
    vehicle_types = [
        {
            "name": "Small Van",
            "capacity": 50,
            "speed": 60,
            "fixed_cost": 80,
            "variable_cost": 0.3,
            "fuel_consumption": 0.08
        },
        {
            "name": "Medium Truck",
            "capacity": 100,
            "speed": 50,
            "fixed_cost": 120,
            "variable_cost": 0.4,
            "fuel_consumption": 0.12
        },
        {
            "name": "Large Truck",
            "capacity": 200,
            "speed": 45,
            "fixed_cost": 180,
            "variable_cost": 0.5,
            "fuel_consumption": 0.18
        },
        {
            "name": "Extra Large Truck",
            "capacity": 300,
            "speed": 40,
            "fixed_cost": 250,
            "variable_cost": 0.6,
            "fuel_consumption": 0.25
        }
    ]
    
    # Distribute vehicle types across the fleet
    if len(vehicles) <= 4:
        # If few vehicles, use one of each type
        new_vehicles = []
        for i in range(min(len(vehicles), len(vehicle_types))):
            v_type = vehicle_types[i]
            vehicles[i].capacity = v_type["capacity"]
            vehicles[i].speed = v_type["speed"]
            vehicles[i].fixed_cost = v_type["fixed_cost"]
            vehicles[i].variable_cost = v_type["variable_cost"]
            vehicles[i].fuel_consumption = v_type["fuel_consumption"]
            new_vehicles.append(vehicles[i])
    else:
        # For larger fleets, distribute types proportionally
        new_vehicles = []
        
        # Determine distribution (more smaller vehicles, fewer larger ones)
        distribution = [0.4, 0.3, 0.2, 0.1]  # 40% small vans, 30% medium trucks, etc.
        
        # Apply distribution
        remaining = len(vehicles)
        used = 0
        
        for i, pct in enumerate(distribution):
            count = int(len(vehicles) * pct)
            if i == len(distribution) - 1:
                # Last type gets all remaining vehicles
                count = remaining
                
            v_type = vehicle_types[i]
            
            for j in range(count):
                if used < len(vehicles):
                    vehicles[used].capacity = v_type["capacity"]
                    vehicles[used].speed = v_type["speed"]
                    vehicles[used].fixed_cost = v_type["fixed_cost"]
                    vehicles[used].variable_cost = v_type["variable_cost"]
                    vehicles[used].fuel_consumption = v_type["fuel_consumption"]
                    new_vehicles.append(vehicles[used])
                    used += 1
                    remaining -= 1
    
    logger.info(f"Configured heterogeneous fleet with {len(new_vehicles)} vehicles")
    return new_vehicles

def main():
    """Main function to run the VRP optimization."""
    # Start timing
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config(args.config)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data_loader = DataLoader(args.data)
    customers, vehicles, depot = data_loader.load()
    
    try:
        # Configure traffic patterns if enabled
        if args.traffic:
            logger.info("Configuring traffic patterns")
            configure_traffic_patterns(vehicles)
        
        # Configure heterogeneous fleet if enabled
        if args.heterogeneous_fleet:
            logger.info("Configuring heterogeneous fleet")
            vehicles = configure_heterogeneous_fleet(vehicles)
        
        # Initialize solution
        logger.info("Initializing solution")
        initial_solution = Solution(customers, vehicles, depot)
        initial_solution.generate_initial_solution(method="nearest_neighbor")  # Force nearest_neighbor method
        
        # Enable time-dependent routing if traffic simulation is enabled
        if args.traffic:
            logger.info("Enabling time-dependent routing")
            initial_solution.enable_time_dependent_routing(True)
        
        # Set driver constraints if enabled
        if args.driver_hours:
            logger.info("Setting driver working hour constraints")
            initial_solution.set_driver_constraints(
                max_route_duration=10.0,  # 10 hours max route duration
                max_continuous_driving=4.5,  # 4.5 hours max continuous driving (EU regulation)
                required_break_time=0.75  # 45 minutes break
            )
    except Exception as e:
        logger.error(f"Error initializing solution: {str(e)}")
        logger.error("Falling back to basic initialization")
        # Fallback to basic initialization without special options
        initial_solution = Solution(customers, vehicles, depot)
        initial_solution.generate_initial_solution(method="nearest_neighbor")
    
    # Run ALNS algorithm
    logger.info(f"Running ALNS algorithm for {args.iterations} iterations")
    try:
        alns = ALNS(initial_solution, config)
        best_solution = alns.solve(args.iterations)
    except Exception as e:
        logger.error(f"Error during ALNS algorithm execution: {str(e)}")
        logger.error("Using initial solution as best solution")
        best_solution = initial_solution
        # Create empty ALNS object to avoid errors in visualization
        alns = ALNS(initial_solution, config)
        # Initialize required fields for visualization
        alns.iter_objective_values = [initial_solution.evaluate()]
        alns.iter_best_values = [initial_solution.evaluate()]
        alns.iter_temperatures = [100.0]
        alns.iter_details = []
    
    # Generate progress plot
    logger.info("Generating progress visualization")
    alns.plot_progress(f"{args.output}/progress.png")
    
    # Generate detailed iteration report
    logger.info("Generating iteration report")
    alns.generate_iteration_report(f"{args.output}/iterations.html")
    
    # Calculate metrics
    logger.info("Calculating metrics")
    total_distance = best_solution.calculate_total_distance()
    total_time = best_solution.calculate_total_time()
    fuel_consumption = best_solution.calculate_fuel_consumption()
    
    # Calculate environmental impact
    env_impact = best_solution.calculate_environmental_impact()
    
    # Calculate driver metrics if driver hours enabled
    if args.driver_hours:
        try:
            driver_metrics = best_solution.calculate_driver_metrics()
        except Exception as e:
            logger.error(f"Error calculating driver metrics: {str(e)}")
            driver_metrics = {
                "total_driving_time": 0,
                "total_break_time": 0,
                "total_service_time": 0,
                "total_wait_time": 0,
                "total_working_time": 0,
                "driving_time_percentage": 0
            }
    
    # Print results
    logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Total distance: {total_distance:.2f} km")
    logger.info(f"Total time: {total_time:.2f} hours")
    logger.info(f"Estimated fuel consumption: {fuel_consumption:.2f} liters")
    logger.info(f"Estimated CO2 emissions: {env_impact['total_co2_emissions']:.2f} kg")
    
    if args.driver_hours:
        logger.info(f"Total driving time: {driver_metrics['total_driving_time']:.2f} hours")
        logger.info(f"Total break time: {driver_metrics['total_break_time']:.2f} hours")
    
    # Save detailed results to JSON
    results = {
        "summary": {
            "total_distance": total_distance,
            "total_time": total_time,
            "fuel_consumption": fuel_consumption,
            "co2_emissions": env_impact["total_co2_emissions"],
            "total_vehicles_used": len([r for r in best_solution.routes if r.customers]),
            "total_customers": len(best_solution.all_customers),
            "computation_time": time.time() - start_time
        },
        "environmental_impact": env_impact,
        "routes": []
    }
    
    # Add driver metrics if available
    if args.driver_hours:
        results["driver_metrics"] = driver_metrics
    
    # Add route details
    for i, route in enumerate(best_solution.routes):
        if not route.customers:
            continue
            
        route_details = {
            "route_id": route.id,
            "vehicle_id": route.vehicle.id,
            "vehicle_capacity": route.vehicle.capacity,
            "total_demand": route.total_demand,
            "distance": route.calculate_distance(),
            "travel_time": route.calculate_travel_time(),
            "co2_emissions": route.calculate_co2_emissions(),
            "customer_count": len(route.customers),
            "customer_ids": [c.id for c in route.customers]
        }
        
        results["routes"].append(route_details)
    
    # Save to JSON
    with open(f"{args.output}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    logger.info("Generating visualizations")
    try:
        visualizer = Visualizer(best_solution)
        
        # Generate route map
        try:
            visualizer.plot_routes(f"{args.output}/routes.html")
        except Exception as e:
            logger.error(f"Error generating routes visualization: {str(e)}")
        
        # Generate KPI dashboard
        try:
            visualizer.create_kpi_dashboard(f"{args.output}/kpi_dashboard.html")
        except Exception as e:
            logger.error(f"Error generating KPI dashboard: {str(e)}")
        
        # Add environmental impact visualization
        try:
            visualizer.plot_environmental_impact(f"{args.output}/environmental_impact.html")
        except Exception as e:
            logger.error(f"Error generating environmental impact visualization: {str(e)}")
        
        # Add traffic patterns visualization if enabled
        if args.traffic:
            try:
                visualizer.plot_traffic_patterns(f"{args.output}/traffic_patterns.html")
            except Exception as e:
                logger.error(f"Error generating traffic patterns visualization: {str(e)}")
        
        # Add heterogeneous fleet visualization if enabled
        if args.heterogeneous_fleet:
            try:
                visualizer.plot_fleet_composition(f"{args.output}/fleet_composition.html")
            except Exception as e:
                logger.error(f"Error generating fleet composition visualization: {str(e)}")
    except Exception as e:
        logger.error(f"Error in visualization process: {str(e)}")
    
    logger.info(f"Results saved to {args.output}/")

if __name__ == "__main__":
    main() 