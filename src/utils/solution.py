"""
Solution class for VRP Optimizer.
Represents a solution to the Vehicle Routing Problem.
"""

import logging
import copy
import random
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class Route:
    """Represents a single vehicle route."""
    
    def __init__(self, vehicle, depot):
        """
        Initialize a route.
        
        Args:
            vehicle (Vehicle): The vehicle assigned to this route
            depot (Depot): The depot node
        """
        self.vehicle = vehicle
        self.depot = depot
        self.customers = []  # Ordered list of customers
        self.load = 0  # Current load
        self.distance = 0  # Total distance
        self.duration = 0  # Total duration
        self.departure_time = depot.open_time  # Departure time from depot
        self.arrival_times = {}  # Arrival times at each customer
        self.wait_times = {}  # Wait times at each customer (if arrived before ready_time)
        
    def add_customer(self, customer):
        """
        Add a customer to the route.
        
        Args:
            customer: Customer to add
            
        Returns:
            bool: True if customer was added successfully, False otherwise
        """
        # Check if adding this customer would exceed vehicle capacity
        if self.load + customer.demand > self.vehicle.capacity:
            return False
        
        # If this is the first customer, calculate from depot
        if not self.customers:
            travel_time = self.vehicle.travel_time(self.depot, customer)
            arrival_time = self.departure_time + travel_time
            
            # Check if we can reach customer before due time
            if arrival_time > customer.due_time:
                return False
            
            # Calculate wait time if we arrive before ready time
            wait_time = max(0, customer.ready_time - arrival_time)
            service_start_time = arrival_time + wait_time
            
            # Update route information
            self.distance += self.depot.distance_to(customer)
            self.duration += travel_time + wait_time + customer.service_time
            self.load += customer.demand
            self.arrival_times[customer.id] = arrival_time
            self.wait_times[customer.id] = wait_time
            self.customers.append(customer)
        else:
            # Calculate from last customer
            prev_customer = self.customers[-1]
            travel_time = self.vehicle.travel_time(prev_customer, customer)
            
            # Calculate arrival time based on when we finished at previous customer
            prev_departure = (
                self.arrival_times[prev_customer.id] + 
                self.wait_times[prev_customer.id] + 
                prev_customer.service_time
            )
            arrival_time = prev_departure + travel_time
            
            # Check if we can reach customer before due time
            if arrival_time > customer.due_time:
                return False
            
            # Calculate wait time if we arrive before ready time
            wait_time = max(0, customer.ready_time - arrival_time)
            service_start_time = arrival_time + wait_time
            
            # Check if we can get back to depot before closing time
            time_to_depot = self.vehicle.travel_time(customer, self.depot)
            return_time = service_start_time + customer.service_time + time_to_depot
            
            if return_time > self.depot.close_time:
                return False
            
            # Update route information
            self.distance += prev_customer.distance_to(customer)
            self.duration += travel_time + wait_time + customer.service_time
            self.load += customer.demand
            self.arrival_times[customer.id] = arrival_time
            self.wait_times[customer.id] = wait_time
            self.customers.append(customer)
        
        return True
    
    def remove_customer(self, customer_id):
        """
        Remove a customer from the route.
        
        Args:
            customer_id (int): ID of the customer to remove
            
        Returns:
            Customer: The removed customer, or None if not found
        """
        # Find the customer in the route
        customer_index = None
        customer = None
        for i, c in enumerate(self.customers):
            if c.id == customer_id:
                customer_index = i
                customer = c
                break
        
        if customer_index is None:
            # Customer not found in this route
            return None
        
        # Update the route information
        # Get the previous and next nodes
        prev_node = self.depot if customer_index == 0 else self.customers[customer_index - 1]
        next_node = self.depot if customer_index == len(self.customers) - 1 else self.customers[customer_index + 1]
        
        # Calculate the change in distance
        old_distance = (
            prev_node.distance_to(customer) + 
            customer.distance_to(next_node)
        )
        new_distance = prev_node.distance_to(next_node)
        self.distance = self.distance - old_distance + new_distance
        
        # Update load
        self.load -= customer.demand
        
        # Remove customer from list
        self.customers.pop(customer_index)
        
        # Recalculate arrival and wait times for all subsequent customers
        if customer_index < len(self.customers):
            self._recalculate_times(customer_index)
        
        # Clean up arrival_times and wait_times
        if customer.id in self.arrival_times:
            del self.arrival_times[customer.id]
        if customer.id in self.wait_times:
            del self.wait_times[customer.id]
        
        return customer
    
    def _recalculate_times(self, start_index):
        """
        Recalculate arrival and wait times starting from the given index.
        
        Args:
            start_index (int): Index to start recalculation from
        """
        if start_index == 0:
            # First customer after removal, calculate from depot
            customer = self.customers[0]
            travel_time = self.vehicle.travel_time(self.depot, customer)
            arrival_time = self.departure_time + travel_time
            wait_time = max(0, customer.ready_time - arrival_time)
            
            self.arrival_times[customer.id] = arrival_time
            self.wait_times[customer.id] = wait_time
            start_index += 1
        
        # Recalculate for all subsequent customers
        for i in range(start_index, len(self.customers)):
            prev_customer = self.customers[i-1]
            customer = self.customers[i]
            
            prev_departure = (
                self.arrival_times[prev_customer.id] + 
                self.wait_times[prev_customer.id] + 
                prev_customer.service_time
            )
            
            travel_time = self.vehicle.travel_time(prev_customer, customer)
            arrival_time = prev_departure + travel_time
            wait_time = max(0, customer.ready_time - arrival_time)
            
            self.arrival_times[customer.id] = arrival_time
            self.wait_times[customer.id] = wait_time
    
    def calculate_total_time(self):
        """
        Calculate the total route duration.
        
        Returns:
            float: Total route duration in hours
        """
        if not self.customers:
            return 0
        
        # Time to first customer
        total_time = self.vehicle.travel_time(self.depot, self.customers[0])
        
        # Time between customers
        for i in range(len(self.customers) - 1):
            current = self.customers[i]
            next_customer = self.customers[i + 1]
            
            # Add service time at current location
            total_time += current.service_time
            
            # Add wait time at current location
            total_time += self.wait_times.get(current.id, 0)
            
            # Add travel time to next location
            total_time += self.vehicle.travel_time(current, next_customer)
        
        # Add service time for last customer
        if self.customers:
            total_time += self.customers[-1].service_time
            total_time += self.wait_times.get(self.customers[-1].id, 0)
        
        # Add time back to depot
        total_time += self.vehicle.travel_time(self.customers[-1], self.depot)
        
        return total_time
    
    def calculate_total_distance(self):
        """
        Calculate the total route distance.
        
        Returns:
            float: Total route distance in kilometers
        """
        if not self.customers:
            return 0
        
        # Distance from depot to first customer
        total_distance = self.depot.distance_to(self.customers[0])
        
        # Distance between customers
        for i in range(len(self.customers) - 1):
            total_distance += self.customers[i].distance_to(self.customers[i + 1])
        
        # Distance from last customer back to depot
        total_distance += self.customers[-1].distance_to(self.depot)
        
        return total_distance
    
    def calculate_fuel_consumption(self):
        """
        Calculate total fuel consumption for the route.
        
        Returns:
            float: Fuel consumption in liters
        """
        return self.calculate_total_distance() * self.vehicle.fuel_consumption
    
    def calculate_cost(self):
        """
        Calculate the total cost of the route.
        
        Returns:
            float: Total cost (fixed + variable)
        """
        # Only count fixed cost if route is used
        fixed_cost = self.vehicle.fixed_cost if self.customers else 0
        
        # Variable cost based on distance
        variable_cost = self.calculate_total_distance() * self.vehicle.variable_cost
        
        return fixed_cost + variable_cost
    
    def is_feasible(self):
        """
        Check if the route is feasible.
        
        Returns:
            bool: True if route is feasible, False otherwise
        """
        # Check capacity constraint
        if self.load > self.vehicle.capacity:
            return False
        
        # Check time windows
        time = self.departure_time
        current_node = self.depot
        
        for customer in self.customers:
            # Travel time to next customer
            time += self.vehicle.travel_time(current_node, customer)
            
            # Check if we arrive before due time
            if time > customer.due_time:
                return False
            
            # Wait if we arrive before ready time
            if time < customer.ready_time:
                time = customer.ready_time
            
            # Service time
            time += customer.service_time
            
            # Update current node
            current_node = customer
        
        # Check if we can return to depot before closing
        time += self.vehicle.travel_time(current_node, self.depot)
        if time > self.depot.close_time:
            return False
        
        return True
    
    def __str__(self):
        """Return string representation of the route."""
        customer_ids = [str(c.id) for c in self.customers]
        return f"Route(vehicle={self.vehicle.id}, customers={' -> '.join(customer_ids)}, load={self.load}, distance={self.distance:.2f})"

class Solution:
    """Represents a complete solution to the VRP."""
    
    def __init__(self, customers, vehicles, depot):
        """
        Initialize a solution.
        
        Args:
            customers (list): List of Customer objects
            vehicles (list): List of Vehicle objects
            depot (Depot): Depot node
        """
        self.customers = customers
        self.vehicles = vehicles
        self.depot = depot
        self.routes = []  # List of Route objects
        self.unassigned = customers.copy()  # Customers not yet assigned to any route
        self.total_distance = 0
        self.total_cost = 0
        self.objective_value = float('inf')
    
    def generate_initial_solution(self, method="nearest_neighbor"):
        """
        Generate an initial solution using various methods.
        
        Args:
            method (str): Method to use for generating initial solution.
                Options: 'nearest_neighbor', 'savings', 'random'
        """
        if method == "nearest_neighbor":
            self._nearest_neighbor_initialization()
        elif method == "savings":
            self._savings_initialization()
        else:  # Random
            self._random_initialization()
            
        # Calculate objective value
        self.evaluate()
    
    def _nearest_neighbor_initialization(self):
        """Initialize solution using the nearest neighbor heuristic."""
        # Clear existing routes
        self.routes = []
        
        # Start with all customers unassigned
        self.unassigned = self.customers.copy()
        
        # Create routes one by one
        for vehicle in self.vehicles:
            if not self.unassigned:
                break
            
            # Create a new route for this vehicle
            route = Route(vehicle, self.depot)
            
            # Start from depot
            current_node = self.depot
            
            # Keep adding nearest unassigned customer
            while self.unassigned:
                # Find nearest unassigned customer
                nearest_customer = None
                nearest_distance = float('inf')
                
                for customer in self.unassigned:
                    distance = current_node.distance_to(customer)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = customer
                
                # Try to add this customer to the route
                if nearest_customer and route.add_customer(nearest_customer):
                    # Customer was added successfully
                    self.unassigned.remove(nearest_customer)
                    current_node = nearest_customer
                else:
                    # Can't add any more customers to this route
                    break
            
            # Add route if it has any customers
            if route.customers:
                self.routes.append(route)
        
        # If there are still unassigned customers, create additional routes
        while self.unassigned and len(self.routes) < len(self.vehicles):
            # Get the next available vehicle
            vehicle = self.vehicles[len(self.routes)]
            
            # Create a new route
            route = Route(vehicle, self.depot)
            
            # Find the customer farthest from the depot
            farthest_customer = max(self.unassigned, key=lambda c: self.depot.distance_to(c))
            
            # Try to add this customer
            if route.add_customer(farthest_customer):
                self.unassigned.remove(farthest_customer)
                current_node = farthest_customer
                
                # Try to add more customers
                while self.unassigned:
                    nearest_customer = None
                    nearest_distance = float('inf')
                    
                    for customer in self.unassigned:
                        distance = current_node.distance_to(customer)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_customer = customer
                    
                    if nearest_customer and route.add_customer(nearest_customer):
                        self.unassigned.remove(nearest_customer)
                        current_node = nearest_customer
                    else:
                        break
            
            # Add route if it has any customers
            if route.customers:
                self.routes.append(route)
    
    def _savings_initialization(self):
        """Initialize solution using the Clarke-Wright savings heuristic."""
        # Clear existing routes
        self.routes = []
        
        # Start with all customers unassigned
        self.unassigned = self.customers.copy()
        
        # 1. Create initial solution with one route per customer
        initial_routes = {}
        
        for customer in self.customers:
            # Create a route for each customer
            route = Route(self.vehicles[0], self.depot)  # Use first vehicle for all initial routes
            
            # Try to add the customer
            if route.add_customer(customer):
                initial_routes[customer.id] = route
                self.unassigned.remove(customer)
        
        # 2. Calculate savings for all pairs of customers
        savings = []
        
        for i, customer_i in enumerate(self.customers):
            for customer_j in self.customers[i+1:]:
                # Calculate savings
                saving = (
                    self.depot.distance_to(customer_i) + 
                    self.depot.distance_to(customer_j) - 
                    customer_i.distance_to(customer_j)
                )
                
                savings.append((customer_i.id, customer_j.id, saving))
        
        # Sort savings in descending order
        savings.sort(key=lambda x: x[2], reverse=True)
        
        # 3. Merge routes based on savings
        for customer_i_id, customer_j_id, saving in savings:
            # Skip if saving is negative
            if saving <= 0:
                continue
            
            # Get routes
            route_i = initial_routes.get(customer_i_id)
            route_j = initial_routes.get(customer_j_id)
            
            # Skip if either route has been merged already
            if route_i is None or route_j is None:
                continue
            
            # Skip if both customers are not at the ends of their routes
            if len(route_i.customers) > 1 and route_i.customers[0].id != customer_i_id and route_i.customers[-1].id != customer_i_id:
                continue
            if len(route_j.customers) > 1 and route_j.customers[0].id != customer_j_id and route_j.customers[-1].id != customer_j_id:
                continue
            
            # Get customers
            customer_i = next(c for c in self.customers if c.id == customer_i_id)
            customer_j = next(c for c in self.customers if c.id == customer_j_id)
            
            # Create a new merged route
            merged_route = Route(self.vehicles[0], self.depot)
            
            # Get all customers in order
            route_i_customers = route_i.customers.copy()
            route_j_customers = route_j.customers.copy()
            
            # Ensure customer_i is at the end of route_i
            if route_i_customers[0].id == customer_i_id:
                route_i_customers.reverse()
            
            # Ensure customer_j is at the start of route_j
            if route_j_customers[-1].id == customer_j_id:
                route_j_customers.reverse()
            
            # Combine routes
            merged_customers = route_i_customers + route_j_customers
            
            # Try to add all customers to the merged route
            all_added = True
            for customer in merged_customers:
                if not merged_route.add_customer(customer):
                    all_added = False
                    break
            
            # If merge successful, update routes
            if all_added:
                del initial_routes[customer_i_id]
                del initial_routes[customer_j_id]
                initial_routes[merged_customers[0].id] = merged_route
                initial_routes[merged_customers[-1].id] = merged_route
        
        # 4. Create final routes by assigning vehicles
        unique_routes = {id(route): route for route in initial_routes.values()}
        
        # Sort routes by distance (longest first)
        sorted_routes = sorted(
            unique_routes.values(),
            key=lambda r: r.calculate_total_distance(),
            reverse=True
        )
        
        # Assign vehicles to routes
        for i, route in enumerate(sorted_routes):
            if i < len(self.vehicles):
                route.vehicle = self.vehicles[i]
                self.routes.append(route)
            else:
                # If more routes than vehicles, add customers back to unassigned
                self.unassigned.extend(route.customers)
    
    def _random_initialization(self):
        """Initialize solution randomly."""
        # Clear existing routes
        self.routes = []
        
        # Start with all customers unassigned
        self.unassigned = self.customers.copy()
        
        # Shuffle customers for random assignment
        random.shuffle(self.unassigned)
        
        # Create one route per vehicle
        for vehicle in self.vehicles:
            route = Route(vehicle, self.depot)
            self.routes.append(route)
        
        # Assign customers randomly
        for customer in self.unassigned[:]:
            # Try to add customer to a random route
            random.shuffle(self.routes)
            
            for route in self.routes:
                if route.add_customer(customer):
                    self.unassigned.remove(customer)
                    break
    
    def evaluate(self):
        """
        Evaluate the solution and update metrics.
        
        Returns:
            float: Objective function value
        """
        self.total_distance = sum(route.calculate_total_distance() for route in self.routes)
        self.total_cost = sum(route.calculate_cost() for route in self.routes)
        
        # Penalize unassigned customers
        penalty = len(self.unassigned) * 10000  # Large penalty for unassigned customers
        
        self.objective_value = self.total_distance + penalty
        
        return self.objective_value
    
    def is_feasible(self):
        """
        Check if the solution is feasible.
        
        Returns:
            bool: True if solution is feasible, False otherwise
        """
        # Check if all customers are assigned
        if self.unassigned:
            return False
        
        # Check if all routes are feasible
        return all(route.is_feasible() for route in self.routes)
    
    def copy(self):
        """
        Create a deep copy of the solution.
        
        Returns:
            Solution: Copy of the solution
        """
        new_solution = Solution(self.customers, self.vehicles, self.depot)
        new_solution.routes = copy.deepcopy(self.routes)
        new_solution.unassigned = self.unassigned.copy()
        new_solution.total_distance = self.total_distance
        new_solution.total_cost = self.total_cost
        new_solution.objective_value = self.objective_value
        
        return new_solution
    
    def get_customer_route(self, customer_id):
        """
        Find which route contains a customer.
        
        Args:
            customer_id (int): ID of the customer
            
        Returns:
            int: Index of the route, or -1 if not found
        """
        for i, route in enumerate(self.routes):
            for customer in route.customers:
                if customer.id == customer_id:
                    return i
        return -1
    
    def calculate_total_distance(self):
        """
        Calculate the total distance of all routes.
        
        Returns:
            float: Total distance in kilometers
        """
        return sum(route.calculate_total_distance() for route in self.routes)
    
    def calculate_total_time(self):
        """
        Calculate the total time of all routes.
        
        Returns:
            float: Total time in hours
        """
        return sum(route.calculate_total_time() for route in self.routes)
    
    def calculate_fuel_consumption(self):
        """
        Calculate the total fuel consumption of all routes.
        
        Returns:
            float: Total fuel consumption in liters
        """
        return sum(route.calculate_fuel_consumption() for route in self.routes)
    
    def calculate_total_cost(self):
        """
        Calculate the total cost of all routes.
        
        Returns:
            float: Total cost
        """
        return sum(route.calculate_cost() for route in self.routes)
    
    def get_summary(self):
        """
        Get a summary of the solution.
        
        Returns:
            dict: Summary statistics
        """
        num_vehicles_used = len([r for r in self.routes if r.customers])
        total_distance = self.calculate_total_distance()
        total_time = self.calculate_total_time()
        total_fuel = self.calculate_fuel_consumption()
        total_cost = self.calculate_total_cost()
        
        # Calculate capacity utilization
        total_capacity = sum(route.vehicle.capacity for route in self.routes if route.customers)
        total_demand = sum(route.load for route in self.routes)
        capacity_utilization = total_demand / total_capacity if total_capacity > 0 else 0
        
        # Count total stops
        total_stops = sum(len(route.customers) for route in self.routes)
        
        # Calculate average stops per route
        avg_stops = total_stops / num_vehicles_used if num_vehicles_used > 0 else 0
        
        # Calculate average distance per route
        avg_distance = total_distance / num_vehicles_used if num_vehicles_used > 0 else 0
        
        return {
            "num_vehicles_used": num_vehicles_used,
            "total_vehicles": len(self.vehicles),
            "utilization_rate": num_vehicles_used / len(self.vehicles),
            "total_distance": total_distance,
            "total_time": total_time,
            "total_fuel": total_fuel,
            "total_cost": total_cost,
            "capacity_utilization": capacity_utilization,
            "total_customers": len(self.customers),
            "unassigned_customers": len(self.unassigned),
            "total_stops": total_stops,
            "avg_stops_per_route": avg_stops,
            "avg_distance_per_route": avg_distance
        }
    
    def __str__(self):
        """Return string representation of the solution."""
        route_strings = [str(route) for route in self.routes]
        unassigned_string = f"Unassigned: {[c.id for c in self.unassigned]}" if self.unassigned else ""
        
        return f"Solution(objective={self.objective_value:.2f}, routes={len(self.routes)}, " + \
            f"total_distance={self.total_distance:.2f}, {unassigned_string})\n" + \
            "\n".join(route_strings) 