"""
Adaptive Large Neighborhood Search (ALNS) algorithm for VRP Optimizer.
"""

import logging
import random
import math
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class ALNS:
    """Adaptive Large Neighborhood Search algorithm for VRP."""
    
    def __init__(self, initial_solution, config):
        """
        Initialize the ALNS algorithm.
        
        Args:
            initial_solution: Initial solution
            config: Configuration object
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution.copy()
        self.config = config
        
        # ALNS parameters
        self.iterations = config.get('alns.iterations', 1000)
        self.segment_size = config.get('alns.segment_size', 100)
        self.cooling_rate = config.get('alns.cooling_rate', 0.99)
        self.initial_temperature = config.get('alns.initial_temperature', 100)
        self.reaction_factor = config.get('alns.reaction_factor', 0.1)
        self.weights_decay = config.get('alns.weights_decay', 0.8)
        self.min_destroy_percentage = config.get('alns.min_destroy_percentage', 0.1)
        self.max_destroy_percentage = config.get('alns.max_destroy_percentage', 0.5)
        self.noise_parameter = config.get('alns.noise_parameter', 0.1)
        
        # Destroy operators and their weights
        self.destroy_operators = [
            (self._random_removal, "Random Removal"),
            (self._worst_removal, "Worst Removal"),
            (self._route_removal, "Route Removal"),
            (self._time_oriented_removal, "Time-Oriented Removal"),
        ]
        self.destroy_weights = np.ones(len(self.destroy_operators))
        self.destroy_scores = np.zeros(len(self.destroy_operators))
        
        # Repair operators and their weights
        self.repair_operators = [
            (self._greedy_insertion, "Greedy Insertion"),
            (self._regret_insertion, "Regret Insertion"),
            (self._nearest_neighbor_insertion, "Nearest Neighbor"),
        ]
        self.repair_weights = np.ones(len(self.repair_operators))
        self.repair_scores = np.zeros(len(self.repair_operators))
        
        # Acceptance criteria
        self.temperature = self.initial_temperature
        
        # Statistics
        self.iter_objective_values = []
        self.iter_best_values = []
        self.iter_temperatures = []
    
    def solve(self, iterations=None):
        """
        Solve the VRP using ALNS.
        
        Args:
            iterations (int, optional): Number of iterations to run
            
        Returns:
            Solution: Best solution found
        """
        if iterations is not None:
            self.iterations = iterations
        
        # Initialize statistics
        self.iter_objective_values = []
        self.iter_best_values = []
        self.iter_temperatures = []
        
        # Store initial solution values
        current_obj = self.current_solution.evaluate()
        best_obj = self.best_solution.evaluate()
        
        self.iter_objective_values.append(current_obj)
        self.iter_best_values.append(best_obj)
        self.iter_temperatures.append(self.temperature)
        
        # Create progress bar
        progress_bar = tqdm(range(self.iterations), desc="ALNS")
        
        # Main loop
        for i in progress_bar:
            # 1. Select destroy and repair operators
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)
            
            destroy_operator, destroy_name = self.destroy_operators[destroy_idx]
            repair_operator, repair_name = self.repair_operators[repair_idx]
            
            # 2. Create a copy of the current solution
            temp_solution = self.current_solution.copy()
            
            # 3. Apply destroy operator
            removed_customers = destroy_operator(temp_solution)
            
            # 4. Apply repair operator
            repair_operator(temp_solution, removed_customers)
            
            # 5. Evaluate the new solution
            new_obj = temp_solution.evaluate()
            current_obj = self.current_solution.evaluate()
            best_obj = self.best_solution.evaluate()
            
            # 6. Update weights
            score = self._get_solution_score(new_obj, current_obj, best_obj)
            self.destroy_scores[destroy_idx] += score
            self.repair_scores[repair_idx] += score
            
            # 7. Accept or reject the new solution using simulated annealing
            if self._accept_solution(new_obj, current_obj):
                self.current_solution = temp_solution
                current_obj = new_obj
                
                # Update best solution if improved
                if new_obj < best_obj:
                    self.best_solution = temp_solution.copy()
                    best_obj = new_obj
                    logger.info(f"New best solution found at iteration {i+1}: {best_obj:.2f}")
            
            # 8. Update temperature
            self.temperature = self.cooling_rate * self.temperature
            
            # 9. Update weights every segment_size iterations
            if (i + 1) % self.segment_size == 0:
                self._update_weights()
            
            # 10. Store statistics
            self.iter_objective_values.append(current_obj)
            self.iter_best_values.append(best_obj)
            self.iter_temperatures.append(self.temperature)
            
            # 11. Update progress bar
            progress_bar.set_postfix({
                'current': f"{current_obj:.2f}", 
                'best': f"{best_obj:.2f}",
                'temp': f"{self.temperature:.2f}"
            })
        
        # Return the best solution found
        return self.best_solution
    
    def _select_operator(self, weights):
        """
        Select an operator based on weights.
        
        Args:
            weights (np.ndarray): Weights of operators
            
        Returns:
            int: Index of selected operator
        """
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select operator based on probabilities
        return np.random.choice(len(weights), p=probabilities)
    
    def _get_solution_score(self, new_obj, current_obj, best_obj):
        """
        Get score for the new solution.
        
        Args:
            new_obj (float): Objective value of new solution
            current_obj (float): Objective value of current solution
            best_obj (float): Objective value of best solution
            
        Returns:
            float: Score (higher is better)
        """
        if new_obj < best_obj:
            # New global best
            return 10
        elif new_obj < current_obj:
            # Better than current
            return 5
        elif self._accept_solution(new_obj, current_obj):
            # Accepted but not better
            return 2
        else:
            # Not accepted
            return 0
    
    def _accept_solution(self, new_obj, current_obj):
        """
        Decide whether to accept a new solution using simulated annealing.
        
        Args:
            new_obj (float): Objective value of new solution
            current_obj (float): Objective value of current solution
            
        Returns:
            bool: True if solution is accepted, False otherwise
        """
        if new_obj <= current_obj:
            # Always accept better solutions
            return True
        else:
            # Accept worse solutions with a probability
            delta = (new_obj - current_obj) / current_obj  # Relative difference
            p = math.exp(-delta / self.temperature)
            return random.random() < p
    
    def _update_weights(self):
        """Update the weights of destroy and repair operators."""
        # Apply weight decay
        self.destroy_weights = self.weights_decay * self.destroy_weights + (1 - self.weights_decay) * self.destroy_scores
        self.repair_weights = self.weights_decay * self.repair_weights + (1 - self.weights_decay) * self.repair_scores
        
        # Reset scores
        self.destroy_scores = np.zeros(len(self.destroy_operators))
        self.repair_scores = np.zeros(len(self.repair_operators))
    
    # ----- Destroy Operators -----
    
    def _random_removal(self, solution):
        """
        Remove random customers from the solution.
        
        Args:
            solution (Solution): Solution to modify
            
        Returns:
            list: Removed customers
        """
        # Calculate number of customers to remove
        total_customers = sum(len(route.customers) for route in solution.routes)
        if total_customers == 0:
            return []
            
        num_to_remove = int(random.uniform(
            self.min_destroy_percentage * total_customers,
            self.max_destroy_percentage * total_customers
        ))
        num_to_remove = min(num_to_remove, total_customers)
        
        # Get all customers in current routes
        all_customers = []
        for route in solution.routes:
            all_customers.extend([(route, customer) for customer in route.customers])
        
        # Randomly select customers to remove
        to_remove = random.sample(all_customers, num_to_remove)
        
        # Remove selected customers
        removed_customers = []
        for route, customer in to_remove:
            removed = route.remove_customer(customer.id)
            if removed:
                removed_customers.append(removed)
                solution.unassigned.append(removed)
        
        return removed_customers
    
    def _worst_removal(self, solution):
        """
        Remove customers with highest contribution to objective function.
        
        Args:
            solution (Solution): Solution to modify
            
        Returns:
            list: Removed customers
        """
        # Calculate number of customers to remove
        total_customers = sum(len(route.customers) for route in solution.routes)
        if total_customers == 0:
            return []
            
        num_to_remove = int(random.uniform(
            self.min_destroy_percentage * total_customers,
            self.max_destroy_percentage * total_customers
        ))
        num_to_remove = min(num_to_remove, total_customers)
        
        # Calculate contribution of each customer
        contributions = []
        
        for route in solution.routes:
            for customer in route.customers:
                # Calculate distance contribution
                prev_idx = route.customers.index(customer)
                prev_node = route.depot if prev_idx == 0 else route.customers[prev_idx - 1]
                next_node = route.depot if prev_idx == len(route.customers) - 1 else route.customers[prev_idx + 1]
                
                # Calculate distance with and without this customer
                dist_with = prev_node.distance_to(customer) + customer.distance_to(next_node)
                dist_without = prev_node.distance_to(next_node)
                contribution = dist_with - dist_without
                
                # Add some noise to randomize selection
                noise = random.uniform(0.8, 1.2)
                contributions.append((route, customer, contribution * noise))
        
        # Sort by contribution (highest first)
        contributions.sort(key=lambda x: x[2], reverse=True)
        
        # Remove top contributors
        removed_customers = []
        for route, customer, _ in contributions[:num_to_remove]:
            removed = route.remove_customer(customer.id)
            if removed:
                removed_customers.append(removed)
                solution.unassigned.append(removed)
        
        return removed_customers
    
    def _route_removal(self, solution):
        """
        Remove all customers from a random route.
        
        Args:
            solution (Solution): Solution to modify
            
        Returns:
            list: Removed customers
        """
        # Check if there are any non-empty routes
        non_empty_routes = [r for r in solution.routes if r.customers]
        if not non_empty_routes:
            return []
        
        # Select a random non-empty route
        route = random.choice(non_empty_routes)
        
        # Remove all customers from the route
        removed_customers = []
        for customer in route.customers.copy():
            removed = route.remove_customer(customer.id)
            if removed:
                removed_customers.append(removed)
                solution.unassigned.append(removed)
        
        return removed_customers
    
    def _time_oriented_removal(self, solution):
        """
        Remove customers with similar time windows.
        
        Args:
            solution (Solution): Solution to modify
            
        Returns:
            list: Removed customers
        """
        # Calculate number of customers to remove
        total_customers = sum(len(route.customers) for route in solution.routes)
        if total_customers == 0:
            return []
            
        num_to_remove = int(random.uniform(
            self.min_destroy_percentage * total_customers,
            self.max_destroy_percentage * total_customers
        ))
        num_to_remove = min(num_to_remove, total_customers)
        
        # Get all customers in current routes
        all_customers = []
        for route in solution.routes:
            all_customers.extend(route.customers)
        
        if not all_customers:
            return []
        
        # Randomly select a time period
        time_period_start = random.uniform(
            solution.depot.open_time,
            solution.depot.close_time - 2
        )
        time_period_end = time_period_start + 2  # 2-hour time window
        
        # Calculate time overlap for all customers
        time_overlaps = []
        for customer in all_customers:
            # Calculate overlap with time period
            overlap_start = max(customer.ready_time, time_period_start)
            overlap_end = min(customer.due_time, time_period_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Calculate overlap score (higher is better)
            if customer.due_time < time_period_start or customer.ready_time > time_period_end:
                overlap_score = 0  # No overlap
            else:
                window_length = customer.due_time - customer.ready_time
                overlap_score = overlap / window_length if window_length > 0 else 0
            
            # Add some noise
            noise = random.uniform(0.8, 1.2)
            time_overlaps.append((customer, overlap_score * noise))
        
        # Sort by overlap score (highest first)
        time_overlaps.sort(key=lambda x: x[1], reverse=True)
        
        # Remove customers with highest overlap
        removed_customers = []
        for customer, _ in time_overlaps[:num_to_remove]:
            # Find which route contains this customer
            route_idx = solution.get_customer_route(customer.id)
            if route_idx >= 0:
                route = solution.routes[route_idx]
                removed = route.remove_customer(customer.id)
                if removed:
                    removed_customers.append(removed)
                    solution.unassigned.append(removed)
        
        return removed_customers
    
    # ----- Repair Operators -----
    
    def _greedy_insertion(self, solution, customers_to_insert):
        """
        Insert customers greedily by choosing min cost insertion.
        
        Args:
            solution (Solution): Solution to modify
            customers_to_insert (list): Customers to insert
            
        Returns:
            bool: True if all customers were inserted, False otherwise
        """
        # Make a copy of customers to insert
        remaining = customers_to_insert.copy()
        
        # Keep inserting until all customers are inserted or no more can be inserted
        while remaining:
            # Find best insertion (customer, route, position) with minimum cost
            best_customer = None
            best_route = None
            best_position = None
            best_cost = float('inf')
            
            for customer in remaining:
                for route in solution.routes:
                    # Try to insert at each position
                    for i in range(len(route.customers) + 1):
                        # Insert the customer
                        route.customers.insert(i, customer)
                        
                        # Check if feasible
                        if route.is_feasible():
                            # Calculate cost
                            cost = route.calculate_total_distance()
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_customer = customer
                                best_route = route
                                best_position = i
                        
                        # Remove the customer (restore solution)
                        route.customers.pop(i)
            
            # If a feasible insertion was found, make it permanent
            if best_customer is not None:
                # Insert at best position
                best_route.customers.insert(best_position, best_customer)
                
                # Update route
                best_route._recalculate_times(0)
                
                # Update solution
                remaining.remove(best_customer)
                if best_customer in solution.unassigned:
                    solution.unassigned.remove(best_customer)
            else:
                # No feasible insertion found for any customer
                break
        
        return len(remaining) == 0
    
    def _regret_insertion(self, solution, customers_to_insert):
        """
        Insert customers using regret heuristic.
        
        Args:
            solution (Solution): Solution to modify
            customers_to_insert (list): Customers to insert
            
        Returns:
            bool: True if all customers were inserted, False otherwise
        """
        # Make a copy of customers to insert
        remaining = customers_to_insert.copy()
        
        # Keep inserting until all customers are inserted or no more can be inserted
        while remaining:
            # Calculate k-regret values for each customer
            k = 3  # Regret-k parameter
            regrets = []
            
            for customer in remaining:
                # Calculate the insertion cost for each route
                insertion_costs = []
                
                for route in solution.routes:
                    # Try to insert at each position
                    route_costs = []
                    
                    for i in range(len(route.customers) + 1):
                        # Insert the customer
                        route.customers.insert(i, customer)
                        
                        # Check if feasible
                        if route.is_feasible():
                            # Calculate cost
                            cost = route.calculate_total_distance()
                            route_costs.append((i, cost))
                        
                        # Remove the customer (restore solution)
                        route.customers.pop(i)
                    
                    if route_costs:
                        # Get best insertion position for this route
                        best_pos, best_cost = min(route_costs, key=lambda x: x[1])
                        insertion_costs.append((route, best_pos, best_cost))
                
                # Sort insertion costs (best first)
                insertion_costs.sort(key=lambda x: x[2])
                
                # Calculate regret value
                regret = 0
                best_route = None
                best_pos = None
                
                if insertion_costs:
                    best_route, best_pos, best_cost = insertion_costs[0]
                    
                    if len(insertion_costs) >= k:
                        for i in range(1, k):
                            regret += insertion_costs[i][2] - insertion_costs[0][2]
                
                regrets.append((customer, regret, best_route, best_pos))
            
            # Select customer with highest regret value
            if not regrets:
                break
            
            # Sort by regret value (highest first)
            regrets.sort(key=lambda x: x[1], reverse=True)
            
            # Get best customer
            best_customer, _, best_route, best_pos = regrets[0]
            
            # If no feasible insertion, break
            if best_route is None:
                break
            
            # Insert the customer
            best_route.customers.insert(best_pos, best_customer)
            
            # Update route
            best_route._recalculate_times(0)
            
            # Update solution
            remaining.remove(best_customer)
            if best_customer in solution.unassigned:
                solution.unassigned.remove(best_customer)
        
        return len(remaining) == 0
    
    def _nearest_neighbor_insertion(self, solution, customers_to_insert):
        """
        Insert customers using nearest neighbor heuristic.
        
        Args:
            solution (Solution): Solution to modify
            customers_to_insert (list): Customers to insert
            
        Returns:
            bool: True if all customers were inserted, False otherwise
        """
        # Make a copy of customers to insert
        remaining = customers_to_insert.copy()
        
        # Keep inserting until all customers are inserted or no more can be inserted
        while remaining:
            # Find the nearest customer to any existing route
            best_customer = None
            best_route = None
            best_position = None
            best_distance = float('inf')
            
            for customer in remaining:
                for route in solution.routes:
                    if not route.customers:
                        # Empty route - calculate distance from depot
                        distance = solution.depot.distance_to(customer)
                        
                        # Try inserting at position 0
                        route.customers.insert(0, customer)
                        
                        if route.is_feasible() and distance < best_distance:
                            best_distance = distance
                            best_customer = customer
                            best_route = route
                            best_position = 0
                        
                        # Remove the customer
                        route.customers.pop(0)
                    else:
                        # For each existing customer
                        for i, route_customer in enumerate(route.customers):
                            distance = route_customer.distance_to(customer)
                            
                            # Try inserting after this customer
                            position = i + 1
                            route.customers.insert(position, customer)
                            
                            if route.is_feasible() and distance < best_distance:
                                best_distance = distance
                                best_customer = customer
                                best_route = route
                                best_position = position
                            
                            # Remove the customer
                            route.customers.pop(position)
                        
                        # Try inserting at the beginning
                        distance = solution.depot.distance_to(customer)
                        position = 0
                        route.customers.insert(position, customer)
                        
                        if route.is_feasible() and distance < best_distance:
                            best_distance = distance
                            best_customer = customer
                            best_route = route
                            best_position = position
                        
                        # Remove the customer
                        route.customers.pop(position)
            
            # If a feasible insertion was found, make it permanent
            if best_customer is not None:
                best_route.customers.insert(best_position, best_customer)
                
                # Update route
                best_route._recalculate_times(0)
                
                # Update solution
                remaining.remove(best_customer)
                if best_customer in solution.unassigned:
                    solution.unassigned.remove(best_customer)
            else:
                # No feasible insertion found for any customer
                break
        
        return len(remaining) == 0 