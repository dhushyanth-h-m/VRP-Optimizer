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
    """Route class for VRP solution."""
    
    def __init__(self, id, vehicle, depot):
        """
        Initialize a route.
        
        Args:
            id (int): Route identifier
            vehicle (Vehicle): Vehicle assigned to this route
            depot (Depot): Depot node
        """
        self.id = id
        self.vehicle = vehicle
        self.depot = depot
        self.customers = []
        
        # Driver constraints
        self.max_route_duration = 10.0  # Maximum route duration in hours
        self.max_continuous_driving = 4.5  # Maximum continuous driving time before break in hours
        self.required_break_time = 0.75  # Required break time in hours
        self.driver_start_time = depot.open_time  # Driver starting time
        self.break_locations = []  # List of indices where breaks are inserted
        
        # Traffic considerations
        self.use_time_dependent_speeds = False
        
        # Environmental impact
        self.co2_emissions_per_km = 2.5  # kg CO2 per km (average diesel truck)
        
        # Cost variables
        self.fixed_cost = vehicle.fixed_cost
        self.variable_cost_per_km = vehicle.variable_cost
    
    @property
    def total_demand(self):
        """Calculate total demand of the route."""
        return sum(customer.demand for customer in self.customers)
    
    def calculate_cost(self):
        """
        Calculate the cost of the route.
        
        Returns:
            float: Total cost
        """
        # If no customers, return 0
        if not self.customers:
            return 0
        
        # Calculate total distance and time
        distance = self.calculate_distance()
        travel_time = self.calculate_travel_time()
        
        # Cost components
        distance_cost = distance * self.variable_cost_per_km
        time_cost = travel_time * 30  # Assuming $30/hour driver cost
        fixed_cost = self.fixed_cost if self.customers else 0
        
        # Penalty for duration exceeding max route duration
        duration_violation = max(0, travel_time - self.max_route_duration)
        duration_penalty = duration_violation * 100  # High penalty for exceeding max duration
        
        # Penalty for continuous driving violations
        driving_violation = self.calculate_continuous_driving_violation()
        driving_penalty = driving_violation * 100
        
        # Total cost
        total_cost = distance_cost + time_cost + fixed_cost + duration_penalty + driving_penalty
        
        return total_cost
    
    def calculate_distance(self):
        """
        Calculate the total distance of the route.
        
        Returns:
            float: Total distance in kilometers
        """
        if not self.customers:
            return 0
        
        total_distance = 0
        
        # Distance from depot to first customer
        total_distance += self.depot.distance_to(self.customers[0])
        
        # Distance between customers
        for i in range(len(self.customers) - 1):
            total_distance += self.customers[i].distance_to(self.customers[i + 1])
        
        # Distance from last customer to depot
        total_distance += self.customers[-1].distance_to(self.depot)
        
        return total_distance
    
    def calculate_travel_time(self, with_service_time=True, with_breaks=True):
        """
        Calculate the total travel time of the route.
        
        Args:
            with_service_time (bool): Include service times at customers
            with_breaks (bool): Include break times for driver
            
        Returns:
            float: Total travel time in hours
        """
        if not self.customers:
            return 0
        
        current_time = self.driver_start_time
        total_driving_time = 0
        continuous_driving = 0
        total_time = 0
        
        # Initialize timeline for easier break scheduling
        timeline = []
        
        # Time from depot to first customer
        if self.use_time_dependent_speeds:
            travel_duration = self.vehicle.travel_time(self.depot, self.customers[0], current_time)
        else:
            travel_duration = self.vehicle.travel_time(self.depot, self.customers[0])
        
        current_time += travel_duration
        continuous_driving += travel_duration
        total_driving_time += travel_duration
        total_time += travel_duration
        
        timeline.append(("travel", travel_duration))
        
        # Process each customer visit and travel between customers
        for i in range(len(self.customers)):
            customer = self.customers[i]
            
            # Wait if arrived before ready time
            if current_time < customer.ready_time:
                wait_time = customer.ready_time - current_time
                current_time = customer.ready_time
                total_time += wait_time
                continuous_driving = 0  # Reset continuous driving during wait
                
                timeline.append(("wait", wait_time))
            
            # Service time at customer
            if with_service_time:
                service_duration = customer.service_time
                current_time += service_duration
                total_time += service_duration
                continuous_driving = 0  # Reset continuous driving during service
                
                timeline.append(("service", service_duration))
            
            # Travel to next node (next customer or depot)
            if i < len(self.customers) - 1:
                next_node = self.customers[i + 1]
            else:
                next_node = self.depot
            
            if self.use_time_dependent_speeds:
                travel_duration = self.vehicle.travel_time(customer, next_node, current_time)
            else:
                travel_duration = self.vehicle.travel_time(customer, next_node)
            
            # Check if need to take a break due to continuous driving
            if with_breaks and continuous_driving + travel_duration > self.max_continuous_driving:
                # Take a break before continuing
                break_time = self.required_break_time
                current_time += break_time
                total_time += break_time
                continuous_driving = 0
                
                timeline.append(("break", break_time))
                self.break_locations.append(i)  # Mark that a break is taken after customer i
            
            # Continue with travel
            current_time += travel_duration
            continuous_driving += travel_duration
            total_driving_time += travel_duration
            total_time += travel_duration
            
            timeline.append(("travel", travel_duration))
        
        # Store the timeline for reference
        self.timeline = timeline
        
        return total_time
    
    def calculate_continuous_driving_violation(self):
        """
        Calculate continuous driving time violations.
        
        Returns:
            float: Total violation amount in hours
        """
        if not self.customers:
            return 0
        
        continuous_driving = 0
        violations = 0
        
        # Trip from depot to first customer
        travel_duration = self.vehicle.travel_time(self.depot, self.customers[0])
        continuous_driving += travel_duration
        
        # Process travels between customers
        for i in range(len(self.customers)):
            # Reset continuous driving at service points
            continuous_driving = 0
            
            # Travel to next node
            if i < len(self.customers) - 1:
                next_node = self.customers[i + 1]
            else:
                next_node = self.depot
            
            travel_duration = self.vehicle.travel_time(self.customers[i], next_node)
            continuous_driving += travel_duration
            
            # Check if a break is scheduled here
            if i in self.break_locations:
                continuous_driving = 0
            
            # Accumulate violations
            if continuous_driving > self.max_continuous_driving:
                violations += (continuous_driving - self.max_continuous_driving)
        
        return violations
    
    def is_time_window_feasible(self):
        """
        Check if the route satisfies all time window constraints.
        
        Returns:
            bool: True if feasible, False otherwise
        """
        if not self.customers:
            return True
        
        current_time = self.driver_start_time
        continuous_driving = 0
        
        # Time from depot to first customer
        if self.use_time_dependent_speeds:
            travel_duration = self.vehicle.travel_time(self.depot, self.customers[0], current_time)
        else:
            travel_duration = self.vehicle.travel_time(self.depot, self.customers[0])
        
        current_time += travel_duration
        continuous_driving += travel_duration
        
        # Process each customer
        for i in range(len(self.customers)):
            customer = self.customers[i]
            
            # Check if arrived before due time
            if current_time > customer.due_time:
                return False
            
            # Wait if arrived before ready time
            if current_time < customer.ready_time:
                wait_time = customer.ready_time - current_time
                current_time = customer.ready_time
                continuous_driving = 0  # Reset continuous driving during wait
            
            # Service time at customer
            service_duration = customer.service_time
            current_time += service_duration
            continuous_driving = 0  # Reset continuous driving during service
            
            # Travel to next node (next customer or depot)
            if i < len(self.customers) - 1:
                next_node = self.customers[i + 1]
            else:
                next_node = self.depot
            
            if self.use_time_dependent_speeds:
                travel_duration = self.vehicle.travel_time(customer, next_node, current_time)
            else:
                travel_duration = self.vehicle.travel_time(customer, next_node)
            
            # Check if need to take a break due to continuous driving
            if continuous_driving + travel_duration > self.max_continuous_driving:
                # Take a break before continuing
                break_time = self.required_break_time
                current_time += break_time
                continuous_driving = 0
            
            # Continue with travel
            current_time += travel_duration
            continuous_driving += travel_duration
        
        # Check if returned to depot before closing time
        return current_time <= self.depot.close_time
    
    def calculate_time_window_violation(self):
        """
        Calculate the total time window violation.
        
        Returns:
            float: Total violation amount in hours
        """
        if not self.customers:
            return 0
        
        current_time = self.driver_start_time
        violations = 0
        
        # Time from depot to first customer
        travel_duration = self.vehicle.travel_time(self.depot, self.customers[0])
        current_time += travel_duration
        
        # Process each customer
        for i in range(len(self.customers)):
            customer = self.customers[i]
            
            # Calculate due time violation
            if current_time > customer.due_time:
                violations += (current_time - customer.due_time)
            
            # Wait if arrived before ready time
            if current_time < customer.ready_time:
                current_time = customer.ready_time
            
            # Service time at customer
            service_duration = customer.service_time
            current_time += service_duration
            
            # Travel to next node (next customer or depot)
            if i < len(self.customers) - 1:
                next_node = self.customers[i + 1]
            else:
                next_node = self.depot
            
            travel_duration = self.vehicle.travel_time(customer, next_node)
            current_time += travel_duration
        
        # Calculate depot closing time violation
        if current_time > self.depot.close_time:
            violations += (current_time - self.depot.close_time)
        
        return violations
    
    def calculate_co2_emissions(self):
        """
        Calculate the total CO2 emissions for the route.
        
        Returns:
            float: CO2 emissions in kg
        """
        distance = self.calculate_distance()
        return distance * self.co2_emissions_per_km
    
    def add_customer(self, customer):
        """
        Add a customer to the route.
        
        Args:
            customer: Customer to add
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        # Check if adding this customer would exceed vehicle capacity
        if self.total_demand + customer.demand > self.vehicle.capacity:
            return False
        
        # If this is the first customer, calculate from depot
        if not self.customers:
            travel_time = self.vehicle.travel_time(self.depot, customer)
            arrival_time = self.driver_start_time + travel_time
            
            # Check if we can reach customer before due time
            if arrival_time > customer.due_time:
                return False
                
            # Calculate wait time if arrived before ready time
            wait_time = max(0, customer.ready_time - arrival_time)
            
            # Update arrival and departure times
            self.customers.append(customer)
            
            # No need to update distance, travel time etc. since we calculate these dynamically now
            return True
        
        # For adding subsequent customers, find best insertion position
        best_position = None
        min_cost_increase = float('inf')
        
        # Try inserting at each position
        for position in range(len(self.customers) + 1):
            # Insert customer temporarily
            self.customers.insert(position, customer)
            
            # Check if route is still feasible
            if self.is_time_window_feasible():
                # Calculate cost increase
                new_cost = self.calculate_cost()
                
                if new_cost < min_cost_increase:
                    min_cost_increase = new_cost
                    best_position = position
            
            # Remove customer
            self.customers.pop(position)
        
        # If feasible position found, insert customer
        if best_position is not None:
            self.customers.insert(best_position, customer)
            return True
            
        return False
    
    def remove_customer(self, customer_id):
        """
        Remove a customer from the route.
        
        Args:
            customer_id: ID or Customer object to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        # If a Customer object was passed, get its ID
        if hasattr(customer_id, 'id'):
            customer_id = customer_id.id
            
        # Find the customer in the route
        customer_index = None
        for i, customer in enumerate(self.customers):
            if customer.id == customer_id:
                customer_index = i
                break
                
        if customer_index is None:
            return False  # Customer not found
            
        # Remove the customer
        self.customers.pop(customer_index)
        
        return True
    
    def _recalculate_times(self, start_index):
        """
        Recalculate route times starting from the given index.
        This method doesn't use arrival_times and wait_times dictionaries 
        as they've been removed in favor of dynamic calculations.
        
        Args:
            start_index (int): Index to start recalculation from
        """
        # This method is now a no-op since we calculate times dynamically
        # The method is kept for compatibility with existing code
        pass
    
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
        current_time = self.driver_start_time + total_time
        
        # Add wait time if we arrive before ready time
        if current_time < self.customers[0].ready_time:
            wait_time = self.customers[0].ready_time - current_time
            total_time += wait_time
            current_time = self.customers[0].ready_time
        
        # Process each customer
        for i in range(len(self.customers)):
            # Add service time at current location
            current = self.customers[i]
            total_time += current.service_time
            current_time += current.service_time
            
            # Travel to next node (next customer or depot)
            if i < len(self.customers) - 1:
                next_customer = self.customers[i + 1]
                travel_time = self.vehicle.travel_time(current, next_customer)
                total_time += travel_time
                current_time += travel_time
                
                # Add wait time if we arrive before ready time
                if current_time < next_customer.ready_time:
                    wait_time = next_customer.ready_time - current_time
                    total_time += wait_time
                    current_time = next_customer.ready_time
            else:
                # Add time back to depot
                total_time += self.vehicle.travel_time(current, self.depot)
        
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
    
    def is_feasible(self):
        """
        Check if the route is feasible.
        
        Returns:
            bool: True if route is feasible, False otherwise
        """
        # Check capacity constraint
        if self.total_demand > self.vehicle.capacity:
            return False
        
        # Check time windows
        time = self.driver_start_time
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
    
    def is_empty(self):
        """
        Check if the solution has no routes or all routes are empty.
        
        Returns:
            bool: True if solution is empty, False otherwise
        """
        if not self.routes:
            return True
        
        # Check if all routes are empty
        return all(len(route.customers) == 0 for route in self.routes)
    
    @property
    def all_customers(self):
        """
        Get all customers assigned to routes.
        
        Returns:
            list: All customers in the solution
        """
        customers = []
        for route in self.routes:
            customers.extend(route.customers)
        return customers
    
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
        """
        Initialize solution using nearest neighbor heuristic.
        
        For each vehicle, start at the depot and iteratively add the nearest
        unvisited customer until no more customers can be added.
        """
        # Create a copy of customers to keep track of unassigned customers
        unassigned = self.customers.copy()
        
        # Process each vehicle
        for i, vehicle in enumerate(self.vehicles):
            # Skip if no more unassigned customers
            if not unassigned:
                break
                
            # Create a new route for this vehicle
            route = Route(id=i, vehicle=vehicle, depot=self.depot)
            
            # Start at the depot
            current_node = self.depot
            
            # Keep adding customers until no more can be added
            while unassigned:
                # Find the nearest unassigned customer
                nearest_customer = None
                min_distance = float('inf')
                
                for customer in unassigned:
                    # Check if adding this customer would exceed vehicle capacity
                    if route.total_demand + customer.demand > vehicle.capacity:
                        continue
                        
                    # Calculate distance from current node to this customer
                    distance = current_node.distance_to(customer)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_customer = customer
                
                # If no suitable customer found, move to next vehicle
                if nearest_customer is None:
                    break
                    
                # Try to add customer to route
                if route.add_customer(nearest_customer):
                    # Update current node and remove from unassigned
                    current_node = nearest_customer
                    unassigned.remove(nearest_customer)
                else:
                    # If customer cannot be added (e.g., time window constraints),
                    # consider it unsuitable for this route
                    unassigned.remove(nearest_customer)
                    self.unassigned.append(nearest_customer)
            
            # Add route to solution if it has customers
            if route.customers:
                self.routes.append(route)
            
        # Any remaining unassigned customers could not be serviced
        self.unassigned.extend(unassigned)
    
    def _savings_initialization(self):
        """
        Initialize solution using the savings algorithm (Clarke & Wright).
        
        First creates one route per customer, then merges routes based on savings.
        """
        # Clear existing routes
        self.routes = []
        self.unassigned = []
        
        # 1. Create individual routes for each customer
        individual_routes = []
        for i, customer in enumerate(self.customers):
            # Only consider customers that have feasible time windows
            if customer.ready_time <= customer.due_time and customer.due_time >= self.depot.open_time:
                # Create a route with just this customer
                vehicle = self.vehicles[min(i, len(self.vehicles) - 1)]
                route = Route(id=i, vehicle=vehicle, depot=self.depot)
                
                # Try to add the customer
                if route.add_customer(customer):
                    individual_routes.append(route)
                else:
                    self.unassigned.append(customer)
        
        # 2. Calculate savings for all pairs of routes
        savings = []
        for i, route1 in enumerate(individual_routes):
            for j, route2 in enumerate(individual_routes):
                if i >= j:  # Skip duplicate pairs and self-pairs
                    continue
                
                # Only consider pairs where the customers can be served by the same vehicle
                # and total demand doesn't exceed capacity
                vehicle = self.vehicles[0]  # Use the first vehicle for capacity check
                if route1.total_demand + route2.total_demand > vehicle.capacity:
                    continue
                
                # Get the customers at the ends of the routes
                c1 = route1.customers[0]  # Only one customer per route at this stage
                c2 = route2.customers[0]
                
                # Calculate savings: d(depot, c1) + d(depot, c2) - d(c1, c2)
                saving = (self.depot.distance_to(c1) + 
                         self.depot.distance_to(c2) - 
                         c1.distance_to(c2))
                
                savings.append((saving, i, j))
        
        # 3. Sort savings in descending order
        savings.sort(reverse=True)
        
        # 4. Merge routes based on savings
        merged = [False] * len(individual_routes)
        
        max_routes = min(len(self.vehicles), len(individual_routes))
        route_id_counter = 0
        
        while savings and route_id_counter < max_routes:
            # Get highest saving
            saving, i, j = savings.pop(0)
            
            # Skip if either route has already been merged
            if merged[i] or merged[j]:
                continue
            
            # Get the routes
            route1 = individual_routes[i]
            route2 = individual_routes[j]
            
            # Get the customers
            customers = route1.customers + route2.customers
            
            # Create a new merged route
            vehicle = self.vehicles[route_id_counter]
            merged_route = Route(id=route_id_counter, vehicle=vehicle, depot=self.depot)
            route_id_counter += 1
            
            # Try to add all customers
            all_added = True
            for customer in customers:
                if not merged_route.add_customer(customer):
                    all_added = False
                    break
            
            if all_added:
                # Merge successful
                self.routes.append(merged_route)
                merged[i] = True
                merged[j] = True
        
        # 5. Add any routes that couldn't be merged
        for i, route in enumerate(individual_routes):
            if not merged[i] and route_id_counter < len(self.vehicles):
                # Create a new route with the next available vehicle
                vehicle = self.vehicles[route_id_counter]
                route_id_counter += 1
                
                new_route = Route(id=route_id_counter, vehicle=vehicle, depot=self.depot)
                
                # Add the customer
                if new_route.add_customer(route.customers[0]):
                    self.routes.append(new_route)
                else:
                    self.unassigned.append(route.customers[0])
    
    def _random_initialization(self):
        """
        Initialize solution randomly.
        
        Assigns customers to vehicles in random order.
        """
        # Clear existing routes
        self.routes = []
        
        # Make a copy of customers and shuffle
        import random
        unassigned = self.customers.copy()
        random.shuffle(unassigned)
        
        # Create a route for each vehicle
        for i, vehicle in enumerate(self.vehicles):
            route = Route(id=i, vehicle=vehicle, depot=self.depot)
            
            # Try to add customers until none can be added
            customers_to_try = unassigned.copy()
            for customer in customers_to_try:
                if route.add_customer(customer):
                    unassigned.remove(customer)
            
            # Add route to solution if it has customers
            if route.customers:
                self.routes.append(route)
                
            # Stop if no more customers to assign
            if not unassigned:
                break
        
        # Store any remaining unassigned customers
        self.unassigned = unassigned
    
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
    
    def remove_customer(self, customer):
        """
        Remove a customer from the solution.
        
        Args:
            customer: Customer object or customer ID to remove
            
        Returns:
            bool: True if customer was found and removed, False otherwise
        """
        # Get customer ID if a Customer object was passed
        customer_id = customer.id if hasattr(customer, 'id') else customer
        
        # Find which route the customer is in
        route_idx = self.get_customer_route(customer_id)
        
        # If not found, return False
        if route_idx == -1:
            return False
            
        # Get the route and find the customer
        route = self.routes[route_idx]
        customer_idx = None
        
        for i, c in enumerate(route.customers):
            if c.id == customer_id:
                customer_idx = i
                break
                
        # If found, remove it and add to unassigned
        if customer_idx is not None:
            # Get the actual customer object
            customer_obj = route.customers[customer_idx]
            
            # Remove from route
            route.customers.pop(customer_idx)
            
            # Add to unassigned
            self.unassigned.append(customer_obj)
            
            return True
            
        return False
    
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
        total_demand = sum(route.total_demand for route in self.routes)
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
    
    def calculate_environmental_impact(self):
        """
        Calculate the environmental impact of the solution.
        
        Returns:
            dict: Environmental impact metrics
        """
        total_co2 = sum(route.calculate_co2_emissions() for route in self.routes)
        total_distance = self.calculate_total_distance()
        
        return {
            "total_co2_emissions": total_co2,
            "co2_per_km": total_co2 / total_distance if total_distance > 0 else 0,
            "total_distance": total_distance
        }
    
    def calculate_driver_metrics(self):
        """
        Calculate driver-related metrics.
        
        Returns:
            dict: Driver metrics
        """
        total_driving_time = 0
        total_break_time = 0
        total_service_time = 0
        total_wait_time = 0
        
        for route in self.routes:
            # Calculate travel time components
            route.calculate_travel_time()  # This updates the timeline
            
            # Sum up different components from timeline
            for activity, duration in route.timeline:
                if activity == "travel":
                    total_driving_time += duration
                elif activity == "break":
                    total_break_time += duration
                elif activity == "service":
                    total_service_time += duration
                elif activity == "wait":
                    total_wait_time += duration
        
        return {
            "total_driving_time": total_driving_time,
            "total_break_time": total_break_time,
            "total_service_time": total_service_time,
            "total_wait_time": total_wait_time,
            "total_working_time": total_driving_time + total_break_time + total_service_time + total_wait_time,
            "driving_time_percentage": total_driving_time / (total_driving_time + total_break_time + total_service_time + total_wait_time) * 100 if (total_driving_time + total_break_time + total_service_time + total_wait_time) > 0 else 0
        }
    
    def can_add_route(self):
        """
        Check if more routes can be added to the solution.
        
        Returns:
            bool: True if more routes can be added, False otherwise
        """
        # Check if we have unused vehicles
        return len(self.routes) < len(self.vehicles)
    
    def add_route(self):
        """
        Add a new route to the solution.
        
        Returns:
            Route: The newly created route
        """
        # Only add if we have available vehicles
        if not self.can_add_route():
            return None
            
        # Get the next vehicle
        vehicle = self.vehicles[len(self.routes)]
            
        # Create a new route
        route = Route(id=len(self.routes), vehicle=vehicle, depot=self.depot)
            
        # Add to routes
        self.routes.append(route)
            
        return route
    
    def enable_time_dependent_routing(self, enable=True):
        """
        Enable or disable time-dependent routing.
        
        Args:
            enable (bool): Whether to enable time-dependent routing
        """
        for route in self.routes:
            route.use_time_dependent_speeds = enable
    
    def set_driver_constraints(self, max_route_duration=10.0, max_continuous_driving=4.5, required_break_time=0.75):
        """
        Set driver constraint parameters for all routes.
        
        Args:
            max_route_duration (float): Maximum route duration in hours
            max_continuous_driving (float): Maximum continuous driving time before break in hours
            required_break_time (float): Required break time in hours
        """
        for route in self.routes:
            route.max_route_duration = max_route_duration
            route.max_continuous_driving = max_continuous_driving
            route.required_break_time = required_break_time 