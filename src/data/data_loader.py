"""
Data loader module for the VRP Optimizer.
Handles loading customer, vehicle, and depot data.
"""

import pandas as pd
import numpy as np
import logging
from geopy.distance import great_circle
from math import ceil

logger = logging.getLogger(__name__)

class Node:
    """Base class for nodes (depot and customers)."""
    
    def __init__(self, id, x, y, demand=0, ready_time=0, due_time=0, service_time=0):
        """
        Initialize a node.
        
        Args:
            id (int): Node identifier
            x (float): X coordinate (longitude)
            y (float): Y coordinate (latitude)
            demand (float): Demand amount
            ready_time (float): Earliest time for service
            due_time (float): Latest time for service
            service_time (float): Service duration
        """
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        
    def distance_to(self, other):
        """
        Calculate the distance to another node.
        
        Args:
            other (Node): Another node
            
        Returns:
            float: Distance in kilometers
        """
        return great_circle((self.y, self.x), (other.y, other.x)).kilometers
    
    def __str__(self):
        return f"Node(id={self.id}, coord=({self.x}, {self.y}), demand={self.demand})"

class Customer(Node):
    """Customer node class."""
    
    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        """Initialize a customer node."""
        super().__init__(id, x, y, demand, ready_time, due_time, service_time)

class Depot(Node):
    """Depot node class."""
    
    def __init__(self, id, x, y, open_time, close_time):
        """Initialize a depot node."""
        super().__init__(id, x, y, 0, open_time, close_time, 0)
        self.open_time = float(open_time)
        self.close_time = float(close_time)

class Vehicle:
    """Vehicle class."""
    
    def __init__(self, id, capacity, speed=1.0, fixed_cost=0, variable_cost=0, fuel_consumption=0):
        """
        Initialize a vehicle.
        
        Args:
            id (int): Vehicle identifier
            capacity (float): Maximum capacity
            speed (float): Speed in km/h
            fixed_cost (float): Fixed cost of using the vehicle
            variable_cost (float): Variable cost per km
            fuel_consumption (float): Fuel consumption per km
        """
        self.id = id
        self.capacity = capacity
        self.speed = speed
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.fuel_consumption = fuel_consumption
        # Initialize time-dependent speed factors (default: no variation)
        self.time_dependent_speed_factors = {h: 1.0 for h in range(24)}
        
    def set_time_dependent_speed_factors(self, factors):
        """
        Set time-dependent speed factors to simulate traffic.
        
        Args:
            factors (dict): Dictionary mapping hour of day (0-23) to speed factor (e.g., 0.7 for 70% of normal speed)
        """
        for hour, factor in factors.items():
            if 0 <= hour < 24 and 0 < factor <= 1.5:  # Sanity check on values
                self.time_dependent_speed_factors[hour] = factor
                
    def get_speed_at_time(self, time):
        """
        Get the adjusted speed at a specific time of day.
        
        Args:
            time (float): Time in hours (e.g., 14.5 for 2:30 PM)
            
        Returns:
            float: Adjusted speed in km/h
        """
        hour = int(time) % 24
        return self.speed * self.time_dependent_speed_factors[hour]
        
    def travel_time(self, node1, node2, departure_time=None):
        """
        Calculate travel time between two nodes.
        
        Args:
            node1 (Node): First node
            node2 (Node): Second node
            departure_time (float, optional): Time of departure in hours
            
        Returns:
            float: Travel time in hours
        """
        distance = node1.distance_to(node2)
        
        if departure_time is None:
            # Use average speed if no departure time provided
            return distance / self.speed
        
        # For time-dependent travel time, we need to simulate the journey
        # since the speed might change during the journey
        time = departure_time
        remaining_distance = distance
        total_time = 0
        
        while remaining_distance > 0:
            hour = int(time) % 24
            speed_factor = self.time_dependent_speed_factors[hour]
            current_speed = self.speed * speed_factor
            
            # Calculate how far we can go in the current hour
            hours_in_current_period = min(1.0, ceil(time) - time)
            distance_in_period = current_speed * hours_in_current_period
            
            if distance_in_period >= remaining_distance:
                # We'll reach the destination in this period
                total_time += remaining_distance / current_speed
                remaining_distance = 0
            else:
                # We'll continue in the next period
                total_time += hours_in_current_period
                remaining_distance -= distance_in_period
                time = (time + hours_in_current_period) % 24
        
        return total_time
    
    def __str__(self):
        return f"Vehicle(id={self.id}, capacity={self.capacity}, speed={self.speed})"

class DataLoader:
    """Data loader class for VRP Optimizer."""
    
    def __init__(self, data_path):
        """
        Initialize data loader.
        
        Args:
            data_path (str): Path to data file
        """
        self.data_path = data_path
        
    def load(self):
        """
        Load data from file.
        
        Returns:
            tuple: (customers, vehicles, depot)
        """
        # Check if file exists
        try:
            ext = self.data_path.split('.')[-1].lower()
            if ext == 'csv':
                return self._load_from_csv()
            elif ext == 'json':
                return self._load_from_json()
            else:
                logger.error(f"Unsupported file format: {ext}")
                return self._generate_sample_data()
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
            logger.warning(f"Error loading data: {e}. Generating sample data instead.")
            return self._generate_sample_data()
    
    def _load_from_csv(self):
        """Load data from CSV file."""
        df = pd.read_csv(self.data_path)
        
        # Extract depot data
        depot_rows = df[df['type'] == 'depot']
        if len(depot_rows) == 0:
            logger.warning("No depot found in data. Using default values.")
            depot = Depot(0, 0, 0, 8, 18)  # Default depot
        else:
            depot_row = depot_rows.iloc[0]
            depot = Depot(
                id=0,
                x=depot_row['longitude'],
                y=depot_row['latitude'],
                open_time=float(depot_row.get('open_time', 8)),
                close_time=float(depot_row.get('close_time', 18))
            )
        
        # Extract customer data
        customer_rows = df[df['type'] == 'customer']
        customers = []
        for i, row in customer_rows.iterrows():
            customer = Customer(
                id=row.get('id', i + 1),
                x=row['longitude'],
                y=row['latitude'],
                demand=float(row.get('demand', 1)),
                ready_time=float(row.get('ready_time', depot.open_time)),
                due_time=float(row.get('due_time', depot.close_time)),
                service_time=float(row.get('service_time', 0.5))
            )
            customers.append(customer)
        
        # Extract vehicle data
        vehicle_rows = df[df['type'] == 'vehicle'] if 'type' in df.columns else pd.DataFrame()
        vehicles = []
        if len(vehicle_rows) == 0:
            # Default vehicles if not specified
            for i in range(15):  # Default to 15 vehicles
                vehicle = Vehicle(
                    id=i,
                    capacity=100,
                    speed=50,
                    fixed_cost=100,
                    variable_cost=0.5,
                    fuel_consumption=0.1
                )
                vehicles.append(vehicle)
        else:
            for i, row in vehicle_rows.iterrows():
                vehicle = Vehicle(
                    id=row.get('id', i),
                    capacity=float(row.get('capacity', 100)),
                    speed=float(row.get('speed', 50)),
                    fixed_cost=float(row.get('fixed_cost', 100)),
                    variable_cost=float(row.get('variable_cost', 0.5)),
                    fuel_consumption=float(row.get('fuel_consumption', 0.1))
                )
                vehicles.append(vehicle)
        
        return customers, vehicles, depot
    
    def _load_from_json(self):
        """Load data from JSON file."""
        import json
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Extract depot data
        depot_data = data.get('depot', {'id': 0, 'x': 0, 'y': 0, 'open_time': 8, 'close_time': 18})
        depot = Depot(
            id=depot_data.get('id', 0),
            x=depot_data.get('x', 0),
            y=depot_data.get('y', 0),
            open_time=float(depot_data.get('open_time', 8)),
            close_time=float(depot_data.get('close_time', 18))
        )
        
        # Extract customer data
        customers = []
        for i, customer_data in enumerate(data.get('customers', [])):
            customer = Customer(
                id=customer_data.get('id', i + 1),
                x=customer_data.get('x', 0),
                y=customer_data.get('y', 0),
                demand=float(customer_data.get('demand', 1)),
                ready_time=float(customer_data.get('ready_time', depot.open_time)),
                due_time=float(customer_data.get('due_time', depot.close_time)),
                service_time=float(customer_data.get('service_time', 0.5))
            )
            customers.append(customer)
        
        # Extract vehicle data
        vehicles = []
        for i, vehicle_data in enumerate(data.get('vehicles', [])):
            vehicle = Vehicle(
                id=vehicle_data.get('id', i),
                capacity=float(vehicle_data.get('capacity', 100)),
                speed=float(vehicle_data.get('speed', 50)),
                fixed_cost=float(vehicle_data.get('fixed_cost', 100)),
                variable_cost=float(vehicle_data.get('variable_cost', 0.5)),
                fuel_consumption=float(vehicle_data.get('fuel_consumption', 0.1))
            )
            vehicles.append(vehicle)
        
        # If no vehicles provided, create default ones
        if not vehicles:
            for i in range(15):  # Default to 15 vehicles
                vehicle = Vehicle(
                    id=i,
                    capacity=100,
                    speed=50,
                    fixed_cost=100,
                    variable_cost=0.5,
                    fuel_consumption=0.1
                )
                vehicles.append(vehicle)
        
        return customers, vehicles, depot
    
    def _generate_sample_data(self):
        """
        Generate sample data for testing.
        
        Returns:
            tuple: (customers, vehicles, depot)
        """
        logger.info("Generating sample data")
        
        # Create depot
        depot = Depot(id=0, x=-87.6298, y=41.8781, open_time=8, close_time=18)  # Chicago coordinates
        
        # Create customers (random locations around Chicago)
        customers = []
        np.random.seed(42)  # For reproducibility
        for i in range(50):  # 50 customers
            # Random coordinates within ~10km of depot
            x = depot.x + np.random.uniform(-0.1, 0.1)
            y = depot.y + np.random.uniform(-0.1, 0.1)
            
            # Random demand between 1 and 20
            demand = np.random.randint(1, 21)
            
            # Random time windows (8 AM to 6 PM)
            ready_time = depot.open_time + np.random.uniform(0, 4)
            due_time = ready_time + np.random.uniform(4, 10)
            if due_time > depot.close_time:
                due_time = depot.close_time
            
            # Random service time (15-30 minutes)
            service_time = np.random.uniform(0.25, 0.5)
            
            customer = Customer(
                id=i+1,
                x=x,
                y=y,
                demand=demand,
                ready_time=ready_time,
                due_time=due_time,
                service_time=service_time
            )
            customers.append(customer)
        
        # Create vehicles
        vehicles = []
        for i in range(15):  # 15 vehicles
            vehicle = Vehicle(
                id=i,
                capacity=100,
                speed=50,
                fixed_cost=100,
                variable_cost=0.5,
                fuel_consumption=0.1
            )
            vehicles.append(vehicle)
        
        # Save sample data for future use
        self._save_sample_data(depot, customers, vehicles)
        
        return customers, vehicles, depot
    
    def _save_sample_data(self, depot, customers, vehicles):
        """Save sample data to CSV file."""
        import os
        
        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(self.data_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Create data for CSV
        data = []
        
        # Add depot
        data.append({
            'id': depot.id,
            'type': 'depot',
            'latitude': depot.y,
            'longitude': depot.x,
            'demand': 0,
            'ready_time': depot.open_time,
            'due_time': depot.close_time,
            'service_time': 0,
            'open_time': depot.open_time,
            'close_time': depot.close_time
        })
        
        # Add customers
        for customer in customers:
            data.append({
                'id': customer.id,
                'type': 'customer',
                'latitude': customer.y,
                'longitude': customer.x,
                'demand': customer.demand,
                'ready_time': customer.ready_time,
                'due_time': customer.due_time,
                'service_time': customer.service_time
            })
        
        # Add vehicles
        for vehicle in vehicles:
            data.append({
                'id': vehicle.id,
                'type': 'vehicle',
                'capacity': vehicle.capacity,
                'speed': vehicle.speed,
                'fixed_cost': vehicle.fixed_cost,
                'variable_cost': vehicle.variable_cost,
                'fuel_consumption': vehicle.fuel_consumption
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        logger.info(f"Sample data saved to {self.data_path}") 