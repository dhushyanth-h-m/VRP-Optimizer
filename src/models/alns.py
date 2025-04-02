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
            (self._shaw_removal, "Shaw Removal"),
            (self._proximity_cluster_removal, "Proximity Cluster Removal"),
            (self._historical_knowledge_removal, "Historical Knowledge Removal"),
            (self._time_window_removal, "Time Window Removal"),
        ]
        self.destroy_weights = np.ones(len(self.destroy_operators))
        self.destroy_scores = np.zeros(len(self.destroy_operators))
        
        # Repair operators and their weights
        self.repair_operators = [
            (self._greedy_insertion, "Greedy Insertion"),
            (self._regret_insertion, "Regret Insertion"),
            (self._nearest_neighbor_insertion, "Nearest Neighbor"),
            (self._sequential_insertion, "Sequential Insertion"),
            (self._best_position_insertion, "Best Position Insertion"),
            (self._two_phase_insertion, "Two-Phase Insertion"),
        ]
        self.repair_weights = np.ones(len(self.repair_operators))
        self.repair_scores = np.zeros(len(self.repair_operators))
        
        # Acceptance criteria
        self.temperature = self.initial_temperature
        
        # Statistics
        self.iter_objective_values = []
        self.iter_best_values = []
        self.iter_temperatures = []
        self.iter_details = []  # New list to store iteration details
        
        # Historical knowledge
        self.customer_route_history = {}  # Maps customer to list of route assignments
        self.customer_position_history = {}  # Maps customer to list of position assignments
        self.customer_pair_history = {}  # Maps customer pairs to frequency
    
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
        self.iter_details = []  # New list to store iteration details
        
        # Store initial solution values
        current_obj = self.current_solution.evaluate()
        best_obj = self.best_solution.evaluate()
        
        self.iter_objective_values.append(current_obj)
        self.iter_best_values.append(best_obj)
        self.iter_temperatures.append(self.temperature)
        
        # Create progress bar with compatible options
        progress_bar = tqdm(range(self.iterations), desc="ALNS", 
                          mininterval=0.1, maxinterval=1.0, 
                          dynamic_ncols=True, ncols=100)
                           
        # Add direct console output that works regardless of tqdm
        print(f"\n{'='*80}")
        print(f"Starting ALNS optimization with {self.iterations} iterations")
        print(f"Initial solution: {len([r for r in self.current_solution.routes if r.customers])} vehicles, {current_obj:.2f} objective")
        print(f"{'='*80}")
        print(f"{'Iteration':^10}|{'Objective':^12}|{'Best':^12}|{'Vehicles':^10}|{'Status':^10}")
        print(f"{'-'*10:^10}|{'-'*12:^12}|{'-'*12:^12}|{'-'*10:^10}|{'-'*10:^10}")
        
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
            num_removed = len(removed_customers)
            
            # 4. Apply repair operator
            repair_operator(temp_solution, removed_customers)
            num_unassigned = len(temp_solution.unassigned)
            num_inserted = num_removed - num_unassigned
            
            # 5. Evaluate the new solution
            new_obj = temp_solution.evaluate()
            current_obj = self.current_solution.evaluate()
            best_obj = self.best_solution.evaluate()
            
            # 6. Update weights
            score = self._get_solution_score(new_obj, current_obj, best_obj)
            self.destroy_scores[destroy_idx] += score
            self.repair_scores[repair_idx] += score
            
            # 7. Accept or reject the new solution using simulated annealing
            accepted = self._accept_solution(new_obj, current_obj)
            if accepted:
                self.current_solution = temp_solution
                current_obj = new_obj
                
                # Update best solution if improved
                improved = False
                if new_obj < best_obj:
                    self.best_solution = temp_solution.copy()
                    best_obj = new_obj
                    improved = True
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
            
            # Store detailed iteration info
            iter_detail = {
                "iteration": i+1,
                "destroy_operator": destroy_name,
                "repair_operator": repair_name,
                "customers_removed": num_removed,
                "customers_inserted": num_inserted,
                "customers_unassigned": num_unassigned,
                "objective_value": new_obj,
                "accepted": accepted,
                "improved": new_obj < best_obj,
                "temperature": self.temperature,
                "delta": (new_obj - current_obj) / current_obj if current_obj > 0 else 0
            }
            self.iter_details.append(iter_detail)
            
            # Log iteration information for every iteration to provide more visibility
            vehicles_used = len([r for r in temp_solution.routes if r.customers])
            log_message = f"Iteration {i+1}: objective={new_obj:.2f}, removed={num_removed}, inserted={num_inserted}, " + \
                          f"unassigned={num_unassigned}, vehicles={vehicles_used}, " + \
                          f"accepted={'✓' if accepted else '✗'}, improved={'✓' if improved else '✗'}"
            
            # Always log basic info, but use INFO level for milestones
            if (i+1) % 10 == 0 or i == 0 or improved:
                logger.info(log_message)
            else:
                logger.debug(log_message)
                
            # Always print status as a compact table row for every iteration
            status = "•" if accepted else "✗"
            if new_obj < best_obj:
                status = "★"  # Star for improvements
                
            print(f"{i+1:^10}|{new_obj:^12.2f}|{best_obj:^12.2f}|{vehicles_used:^10}|{status:^10}")
            
            # 11. Update progress bar with more detailed information
            progress_bar.set_postfix({
                'destroy': destroy_name.split()[0],
                'repair': repair_name.split()[0],
                'removed': num_removed,
                'current': f"{current_obj:.2f}", 
                'best': f"{best_obj:.2f}",
                'vehicles': vehicles_used,
                'accepted': '✓' if accepted else '✗'
            }, refresh=True)
            
            # Extra details for significant iterations
            if (i+1) % 10 == 0 or i == 0 or improved or i == self.iterations - 1:
                print(f"Iter {i+1}: {destroy_name} → {repair_name}, removed={num_removed}, inserted={num_inserted}")
            
            # Flush output to ensure it appears immediately
            import sys
            sys.stdout.flush()
        
        # Log final statistics
        vehicles_used = len([r for r in self.best_solution.routes if r.customers])
        total_customers = len(self.best_solution.customers)
        unassigned = len(self.best_solution.unassigned)
        logger.info(f"ALNS completed: objective={best_obj:.2f}, vehicles={vehicles_used}, " +
                  f"customers={total_customers-unassigned}/{total_customers}")
        
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
    
    def plot_progress(self, output_file):
        """
        Plot the progress of the ALNS algorithm.
        
        Args:
            output_file (str): Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            fig.suptitle('ALNS Algorithm Progress', fontsize=16)
            
            iterations = range(len(self.iter_objective_values))
            
            # Objective function values
            axes[0].plot(iterations, self.iter_objective_values, 'b-', label='Current Solution')
            axes[0].plot(iterations, self.iter_best_values, 'r-', label='Best Solution')
            axes[0].set_ylabel('Objective Value')
            axes[0].set_title('Objective Function Value')
            axes[0].legend()
            axes[0].grid(True)
            
            # Temperature
            axes[1].plot(iterations, self.iter_temperatures, 'g-')
            axes[1].set_ylabel('Temperature')
            axes[1].set_title('Simulated Annealing Temperature')
            axes[1].grid(True)
            
            # Acceptance/Improvement Markers
            if self.iter_details:
                # Extract data from iteration details
                accepted_iter = [d["iteration"]-1 for d in self.iter_details if d["accepted"]]
                improved_iter = [d["iteration"]-1 for d in self.iter_details if d["improved"]]
                
                # Calculate % of iterations accepted
                acceptance_rate = len(accepted_iter) / len(self.iter_details) * 100
                improvement_rate = len(improved_iter) / len(self.iter_details) * 100
                
                # Plot markers for accepted and improved solutions
                for i in accepted_iter:
                    axes[2].axvline(x=i, color='g', alpha=0.2)
                for i in improved_iter:
                    axes[2].axvline(x=i, color='r', alpha=0.5)
                
                # Create legend
                accepted_patch = mpatches.Patch(color='g', alpha=0.2, label=f'Accepted ({acceptance_rate:.1f}%)')
                improved_patch = mpatches.Patch(color='r', alpha=0.5, label=f'Improved ({improvement_rate:.1f}%)')
                axes[2].legend(handles=[accepted_patch, improved_patch])
                
                # Set plot labels
                axes[2].set_xlabel('Iteration')
                axes[2].set_ylabel('Events')
                axes[2].set_title('Solution Acceptance and Improvement')
                axes[2].set_ylim(0, 1)  # Dummy Y-axis
                axes[2].set_yticks([])  # Hide Y-ticks
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(output_file)
            plt.close(fig)
            logger.info(f"Progress plot saved to {output_file}")
            
        except ImportError:
            logger.warning("Could not plot progress - matplotlib not available")
    
    def generate_iteration_report(self, output_file):
        """
        Generate a detailed HTML report of all iterations.
        
        Args:
            output_file (str): Path to save the HTML report
        """
        import os
        
        # Calculate summary statistics
        if not self.iter_details:
            logger.warning("No iteration details available for report")
            return
            
        total_iterations = len(self.iter_details)
        accepted_count = sum(1 for d in self.iter_details if d["accepted"])
        improved_count = sum(1 for d in self.iter_details if d["improved"])
        
        destroy_ops_usage = {}
        repair_ops_usage = {}
        
        for detail in self.iter_details:
            # Count destroy operators
            destroy_name = detail["destroy_operator"]
            if destroy_name not in destroy_ops_usage:
                destroy_ops_usage[destroy_name] = {"count": 0, "accepted": 0, "improved": 0}
            destroy_ops_usage[destroy_name]["count"] += 1
            if detail["accepted"]:
                destroy_ops_usage[destroy_name]["accepted"] += 1
            if detail["improved"]:
                destroy_ops_usage[destroy_name]["improved"] += 1
                
            # Count repair operators
            repair_name = detail["repair_operator"]
            if repair_name not in repair_ops_usage:
                repair_ops_usage[repair_name] = {"count": 0, "accepted": 0, "improved": 0}
            repair_ops_usage[repair_name]["count"] += 1
            if detail["accepted"]:
                repair_ops_usage[repair_name]["accepted"] += 1
            if detail["improved"]:
                repair_ops_usage[repair_name]["improved"] += 1
        
        # Calculate percentages
        for op in destroy_ops_usage.values():
            op["acceptance_rate"] = op["accepted"] / op["count"] * 100 if op["count"] > 0 else 0
            op["improvement_rate"] = op["improved"] / op["count"] * 100 if op["count"] > 0 else 0
        
        for op in repair_ops_usage.values():
            op["acceptance_rate"] = op["accepted"] / op["count"] * 100 if op["count"] > 0 else 0
            op["improvement_rate"] = op["improved"] / op["count"] * 100 if op["count"] > 0 else 0
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ALNS Progress Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .summary-section {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 15px;
                    width: 300px;
                }}
                .summary-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .accepted-row {{ 
                    background-color: rgba(0, 255, 0, 0.1) !important; 
                }}
                .improved-row {{ 
                    background-color: rgba(255, 215, 0, 0.2) !important; 
                }}
                .best-row {{ 
                    background-color: rgba(255, 99, 71, 0.2) !important; 
                }}
                .progress-container {{
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    height: 10px;
                    width: 100%;
                    margin-top: 5px;
                }}
                .progress-bar {{
                    height: 10px;
                    border-radius: 5px;
                    background-color: #3498db;
                }}
            </style>
        </head>
        <body>
            <h1>ALNS Algorithm Progress Report</h1>
            
            <div class="summary-section">
                <div class="summary-card">
                    <h3>Iterations</h3>
                    <div class="summary-value">{total_iterations}</div>
                    <p>Total iterations performed</p>
                </div>
                
                <div class="summary-card">
                    <h3>Acceptance Rate</h3>
                    <div class="summary-value">{accepted_count / total_iterations * 100:.1f}%</div>
                    <p>{accepted_count} solutions accepted</p>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {accepted_count / total_iterations * 100}%;"></div>
                    </div>
                </div>
                
                <div class="summary-card">
                    <h3>Improvement Rate</h3>
                    <div class="summary-value">{improved_count / total_iterations * 100:.1f}%</div>
                    <p>{improved_count} solutions improved the best</p>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {improved_count / total_iterations * 100}%;"></div>
                    </div>
                </div>
            </div>
            
            <h2>Operator Performance</h2>
            
            <h3>Destroy Operators</h3>
            <table>
                <tr>
                    <th>Operator</th>
                    <th>Usage</th>
                    <th>Acceptance Rate</th>
                    <th>Improvement Rate</th>
                </tr>
        """
        
        # Add destroy operators rows
        for op_name, stats in destroy_ops_usage.items():
            html_content += f"""
                <tr>
                    <td>{op_name}</td>
                    <td>{stats["count"]} ({stats["count"] / total_iterations * 100:.1f}%)</td>
                    <td>
                        {stats["accepted"]} ({stats["acceptance_rate"]:.1f}%)
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {stats["acceptance_rate"]}%;"></div>
                        </div>
                    </td>
                    <td>
                        {stats["improved"]} ({stats["improvement_rate"]:.1f}%)
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {stats["improvement_rate"]}%;"></div>
                        </div>
                    </td>
                </tr>
            """
            
        html_content += """
            </table>
            
            <h3>Repair Operators</h3>
            <table>
                <tr>
                    <th>Operator</th>
                    <th>Usage</th>
                    <th>Acceptance Rate</th>
                    <th>Improvement Rate</th>
                </tr>
        """
        
        # Add repair operators rows
        for op_name, stats in repair_ops_usage.items():
            html_content += f"""
                <tr>
                    <td>{op_name}</td>
                    <td>{stats["count"]} ({stats["count"] / total_iterations * 100:.1f}%)</td>
                    <td>
                        {stats["accepted"]} ({stats["acceptance_rate"]:.1f}%)
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {stats["acceptance_rate"]}%;"></div>
                        </div>
                    </td>
                    <td>
                        {stats["improved"]} ({stats["improvement_rate"]:.1f}%)
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {stats["improvement_rate"]}%;"></div>
                        </div>
                    </td>
                </tr>
            """
            
        html_content += """
            </table>
            
            <h2>Iteration Details</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>Destroy</th>
                    <th>Repair</th>
                    <th>Removed</th>
                    <th>Inserted</th>
                    <th>Unassigned</th>
                    <th>Objective Value</th>
                    <th>Delta</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add iteration detail rows
        best_obj = float('inf')
        for detail in self.iter_details:
            obj_value = detail["objective_value"]
            is_improved = detail["improved"]
            is_accepted = detail["accepted"]
            
            # Update best objective value
            if is_improved:
                best_obj = obj_value
            
            # Determine row class
            row_class = ""
            if is_improved:
                row_class = "best-row"
            elif is_accepted:
                row_class = "accepted-row"
            
            # Format delta
            delta = detail["delta"]
            delta_display = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
            
            # Format status
            if is_improved:
                status = "✓ New Best"
            elif is_accepted:
                status = "✓ Accepted"
            else:
                status = "✗ Rejected"
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{detail["iteration"]}</td>
                    <td>{detail["destroy_operator"]}</td>
                    <td>{detail["repair_operator"]}</td>
                    <td>{detail["customers_removed"]}</td>
                    <td>{detail["customers_inserted"]}</td>
                    <td>{detail["customers_unassigned"]}</td>
                    <td>{obj_value:.2f}</td>
                    <td>{delta_display}</td>
                    <td>{status}</td>
                </tr>
            """
            
        html_content += """
            </table>
            
            <p style="text-align: center; margin-top: 50px; color: #777;">
                Generated by VRP Optimizer
            </p>
        </body>
        </html>
        """
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Iteration report saved to {output_file}")
    
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
    
    def _shaw_removal(self, solution):
        """
        Shaw removal operator - removes customers that are similar to each other.
        
        The similarity between customers is based on:
        - Geographic proximity
        - Time window similarity
        - Demand similarity
        
        Args:
            solution: Solution to modify
            
        Returns:
            list: Removed customers
        """
        if solution.is_empty():
            return []
        
        # Determine number of customers to remove
        destroy_percentage = random.uniform(self.min_destroy_percentage, self.max_destroy_percentage)
        num_to_remove = max(1, int(destroy_percentage * len(solution.all_customers)))
        
        # Select initial customer to remove
        initial_customer = random.choice(solution.all_customers)
        removed_customers = [initial_customer]
        solution.remove_customer(initial_customer)
        
        # Calculate similarity scores for remaining customers
        while len(removed_customers) < num_to_remove and not solution.is_empty():
            reference_customer = removed_customers[-1]
            remaining_customers = solution.all_customers.copy()
            
            # Calculate similarity scores
            similarity_scores = []
            for customer in remaining_customers:
                # Geographic proximity (normalized distance)
                geo_similarity = reference_customer.distance_to(customer) / 100
                
                # Time window similarity
                tw_similarity = abs(reference_customer.ready_time - customer.ready_time) / 24
                tw_similarity += abs(reference_customer.due_time - customer.due_time) / 24
                tw_similarity /= 2
                
                # Demand similarity (normalized difference)
                demand_similarity = abs(reference_customer.demand - customer.demand) / max(1, max(solution.all_customers, key=lambda c: c.demand).demand)
                
                # Combined similarity score (lower is more similar)
                similarity = 0.5 * geo_similarity + 0.3 * tw_similarity + 0.2 * demand_similarity
                
                # Add some noise to avoid getting stuck in local optima
                noise = random.uniform(0, self.noise_parameter)
                similarity_scores.append((customer, similarity + noise))
            
            # Sort by similarity (ascending)
            similarity_scores.sort(key=lambda x: x[1])
            
            # Select the most similar customer
            next_customer = similarity_scores[0][0]
            removed_customers.append(next_customer)
            solution.remove_customer(next_customer)
        
        return removed_customers
    
    def _proximity_cluster_removal(self, solution):
        """
        Proximity cluster removal - removes customers that form a geographic cluster.
        
        Args:
            solution: Solution to modify
            
        Returns:
            list: Removed customers
        """
        if solution.is_empty():
            return []
        
        # Determine number of customers to remove
        destroy_percentage = random.uniform(self.min_destroy_percentage, self.max_destroy_percentage)
        num_to_remove = max(1, int(destroy_percentage * len(solution.all_customers)))
        
        # Get all customer coordinates
        customer_coords = np.array([[c.x, c.y] for c in solution.all_customers])
        
        # If too few customers, do random removal
        if len(customer_coords) < 3:
            return self._random_removal(solution)
            
        # Create clusters using K-means
        # Number of clusters = min(sqrt(n), 10)
        n_clusters = min(int(np.sqrt(len(customer_coords))), 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(customer_coords)
        
        # Count customers in each cluster
        cluster_counts = np.bincount(clusters)
        
        # Select a random cluster with more than 1 customer
        valid_clusters = [i for i, count in enumerate(cluster_counts) if count > 1]
        if not valid_clusters:
            # Fall back to random removal if no valid clusters
            return self._random_removal(solution)
            
        target_cluster = random.choice(valid_clusters)
        
        # Get customers in the target cluster
        cluster_customers = [c for i, c in enumerate(solution.all_customers) if clusters[i] == target_cluster]
        
        # Limit to the number to remove
        customers_to_remove = cluster_customers[:num_to_remove]
        
        # Remove customers
        removed_customers = []
        for customer in customers_to_remove:
            if customer in solution.all_customers:
                removed_customers.append(customer)
                solution.remove_customer(customer)
        
        return removed_customers
    
    def _historical_knowledge_removal(self, solution):
        """
        Historical knowledge removal - removes customers based on historical assignments.
        
        Args:
            solution: Solution to modify
            
        Returns:
            list: Removed customers
        """
        if solution.is_empty() or not self.customer_route_history:
            # Fall back to random removal if no history
            return self._random_removal(solution)
        
        # Determine number of customers to remove
        destroy_percentage = random.uniform(self.min_destroy_percentage, self.max_destroy_percentage)
        num_to_remove = max(1, int(destroy_percentage * len(solution.all_customers)))
        
        # Get customer and current route assignments
        current_routes = {}
        for route_idx, route in enumerate(solution.routes):
            for customer in route.customers:
                current_routes[customer] = route_idx
        
        # Calculate instability scores (higher = more unstable)
        instability_scores = []
        for customer in solution.all_customers:
            if customer not in self.customer_route_history:
                # No history for this customer
                instability_scores.append((customer, 0))
                continue
                
            current_route = current_routes.get(customer, -1)
            route_history = self.customer_route_history[customer]
            
            # Calculate how often this customer has been in different routes
            route_counts = {}
            for route_idx in route_history:
                route_counts[route_idx] = route_counts.get(route_idx, 0) + 1
            
            # More different routes = more unstable
            instability = len(route_counts)
            
            # If current route rarely used in history, higher instability
            if current_route in route_counts:
                route_frequency = route_counts[current_route] / len(route_history)
                instability += (1 - route_frequency) * 2
                
            instability_scores.append((customer, instability))
        
        # Sort by instability (descending)
        instability_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select the most unstable customers
        customers_to_remove = [item[0] for item in instability_scores[:num_to_remove]]
        
        # Remove customers
        removed_customers = []
        for customer in customers_to_remove:
            if customer in solution.all_customers:
                removed_customers.append(customer)
                solution.remove_customer(customer)
        
        return removed_customers
    
    def _time_window_removal(self, solution):
        """
        Time window removal - removes customers with similar time windows.
        
        Args:
            solution: Solution to modify
            
        Returns:
            list: Removed customers
        """
        if solution.is_empty():
            return []
        
        # Determine number of customers to remove
        destroy_percentage = random.uniform(self.min_destroy_percentage, self.max_destroy_percentage)
        num_to_remove = max(1, int(destroy_percentage * len(solution.all_customers)))
        
        # Select a random time window as reference
        reference_customer = random.choice(solution.all_customers)
        reference_time = random.uniform(reference_customer.ready_time, reference_customer.due_time)
        
        # Calculate time window proximity for all customers
        time_window_scores = []
        for customer in solution.all_customers:
            # Check if the reference time falls within the customer's time window
            if customer.ready_time <= reference_time <= customer.due_time:
                # Inside time window: close proximity
                proximity = 0
            else:
                # Outside time window: distance to closest boundary
                proximity = min(
                    abs(reference_time - customer.ready_time),
                    abs(reference_time - customer.due_time)
                )
            
            # Add noise
            noise = random.uniform(0, self.noise_parameter)
            time_window_scores.append((customer, proximity + noise))
        
        # Sort by proximity (ascending)
        time_window_scores.sort(key=lambda x: x[1])
        
        # Select the customers with closest time windows
        customers_to_remove = [item[0] for item in time_window_scores[:num_to_remove]]
        
        # Remove customers
        removed_customers = []
        for customer in customers_to_remove:
            if customer in solution.all_customers:
                removed_customers.append(customer)
                solution.remove_customer(customer)
        
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
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        # Make a copy of customers to insert
        remaining = valid_customers.copy()
        
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
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        # Make a copy of customers to insert
        remaining = valid_customers.copy()
        
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
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        # Make a copy of customers to insert
        remaining = valid_customers.copy()
        
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
    
    def _sequential_insertion(self, solution, customers_to_insert):
        """
        Insert customers sequentially into the best route.
        
        Args:
            solution (Solution): Solution to modify
            customers_to_insert (list): Customers to insert
            
        Returns:
            bool: True if all customers were inserted, False otherwise
        """
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        # Sort customers by ready time
        valid_customers.sort(key=lambda x: x.ready_time)
        
        # Insert each customer
        for customer in valid_customers:
            best_route = None
            best_position = None
            best_cost_increase = float('inf')
            
            # Try all routes
            for route in solution.routes:
                # Skip if capacity would be exceeded
                if route.total_demand + customer.demand > route.vehicle.capacity:
                    continue
                
                # Calculate current cost
                current_cost = route.calculate_cost()
                
                # Try all positions
                for position in range(len(route.customers) + 1):
                    # Insert customer temporarily
                    route.customers.insert(position, customer)
                    
                    # Check if feasible
                    if route.is_feasible():
                        # Calculate new cost
                        new_cost = route.calculate_cost()
                        cost_increase = new_cost - current_cost
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_route = route
                            best_position = position
                    
                    # Remove customer
                    route.customers.pop(position)
            
            # If no feasible insertion found, create a new route
            if best_route is None:
                if solution.can_add_route():
                    new_route = solution.add_route()
                    new_route.customers.append(customer)
                else:
                    # Fall back to greedy insertion
                    self._insert_one_customer_greedy(solution, customer)
            else:
                # Insert at best position
                best_route.customers.insert(best_position, customer)
                
                # Update historical knowledge
                self._update_historical_knowledge(best_route, best_position, customer)
        
        return len(solution.unassigned) == 0
    
    def _best_position_insertion(self, solution, customers_to_insert):
        """
        Insert customers into their best positions.
        
        Args:
            solution (Solution): Solution to modify
            customers_to_insert (list): Customers to insert
            
        Returns:
            bool: True if all customers were inserted, False otherwise
        """
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        # Insert each customer
        for customer in valid_customers:
            best_cost_increase = float('inf')
            best_route = None
            best_position = None
            
            # Calculate cost increase for each possible insertion
            for route in solution.routes:
                # Skip infeasible routes (e.g., capacity)
                if route.total_demand + customer.demand > route.vehicle.capacity:
                    continue
                
                # Calculate current route cost
                current_cost = route.calculate_cost()
                
                # Try each position in the route
                for position in range(len(route.customers) + 1):
                    # Insert customer
                    route.customers.insert(position, customer)
                    
                    # Check time window feasibility
                    is_feasible = route.is_time_window_feasible()
                    
                    # Calculate cost if feasible
                    if is_feasible:
                        new_cost = route.calculate_cost()
                        cost_increase = new_cost - current_cost
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_route = route
                            best_position = position
                    
                    # Revert insertion
                    route.customers.pop(position)
            
            # If no feasible insertion found, create a new route
            if best_route is None:
                if solution.can_add_route():
                    new_route = solution.add_route()
                    new_route.customers.append(customer)
                else:
                    # Fall back to greedy insertion for this customer
                    self._insert_one_customer_greedy(solution, customer)
            else:
                # Insert at best position
                best_route.customers.insert(best_position, customer)
                
                # Update historical knowledge
                self._update_historical_knowledge(best_route, best_position, customer)
    
    def _two_phase_insertion(self, solution, customers_to_insert):
        """
        Two-phase insertion - first assign customers to best routes, then optimize positions.
        
        Args:
            solution: Solution to modify
            customers_to_insert: List of customers to insert
        """
        # Filter out any non-Customer objects
        valid_customers = []
        for customer in customers_to_insert:
            if hasattr(customer, 'demand') and hasattr(customer, 'id'):
                valid_customers.append(customer)
            else:
                logger.warning(f"Skipping invalid customer object: {customer}")
        
        if not valid_customers:
            return
        
        # Phase 1: Assign customers to routes
        route_assignments = {}  # {route_idx: [customers]}
        
        for customer in valid_customers:
            best_route_idx = -1
            best_cost = float('inf')
            
            # Find best route for this customer
            for route_idx, route in enumerate(solution.routes):
                # Skip infeasible routes (e.g., capacity)
                remaining_capacity = route.vehicle.capacity - route.total_demand
                customers_assigned = route_assignments.get(route_idx, [])
                assigned_demand = sum(c.demand for c in customers_assigned)
                
                if customer.demand > remaining_capacity - assigned_demand:
                    continue
                
                # Evaluate cost based on distance to route centroid
                if not route.customers and not customers_assigned:
                    # Empty route - evaluate distance from depot
                    depot = solution.depot
                    cost = depot.distance_to(customer)
                else:
                    # Calculate centroid of the route (including already assigned customers)
                    all_customers = route.customers + customers_assigned
                    x_sum = sum(c.x for c in all_customers)
                    y_sum = sum(c.y for c in all_customers)
                    centroid_x = x_sum / len(all_customers)
                    centroid_y = y_sum / len(all_customers)
                    
                    # Create temporary node for centroid
                    centroid = type('Node', (), {'x': centroid_x, 'y': centroid_y, 'distance_to': lambda other: ((centroid_x - other.x) ** 2 + (centroid_y - other.y) ** 2) ** 0.5})
                    
                    cost = centroid.distance_to(customer)
                
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = route_idx
            
            # If no feasible route found, create a new one
            if best_route_idx == -1:
                if solution.can_add_route():
                    best_route_idx = len(solution.routes)
                    solution.add_route()
                else:
                    # Assign to route with most remaining capacity
                    max_capacity = -1
                    for route_idx, route in enumerate(solution.routes):
                        remaining_capacity = route.vehicle.capacity - route.total_demand
                        customers_assigned = route_assignments.get(route_idx, [])
                        assigned_demand = sum(c.demand for c in customers_assigned)
                        curr_capacity = remaining_capacity - assigned_demand
                        
                        if curr_capacity > max_capacity:
                            max_capacity = curr_capacity
                            best_route_idx = route_idx
            
            # Add customer to assigned route
            if best_route_idx not in route_assignments:
                route_assignments[best_route_idx] = []
            route_assignments[best_route_idx].append(customer)
        
        # Phase 2: Optimize positions within each route
        for route_idx, customers in route_assignments.items():
            route = solution.routes[route_idx]
            
            # Try all permutations for small sets
            if len(customers) <= 5:
                self._optimize_small_insertions(route, customers)
            else:
                # Use nearest neighbor heuristic for larger sets
                self._optimize_large_insertions(route, customers)
    
    def _optimize_small_insertions(self, route, customers):
        """
        Optimize insertions for small sets by trying all permutations.
        
        Args:
            route: Route to modify
            customers: Customers to insert
        """
        import itertools
        
        original_customers = route.customers.copy()
        best_cost = float('inf')
        best_sequence = None
        
        # Try all permutations
        for perm in itertools.permutations(customers):
            for i in range(len(original_customers) + 1):
                # Insert the permutation at position i
                route.customers = original_customers.copy()
                route.customers[i:i] = perm
                
                # Check feasibility
                is_feasible = route.is_time_window_feasible() and route.total_demand <= route.vehicle.capacity
                
                if is_feasible:
                    cost = route.calculate_cost()
                    if cost < best_cost:
                        best_cost = cost
                        best_sequence = route.customers.copy()
        
        # Use best sequence found or original if none feasible
        if best_sequence:
            route.customers = best_sequence
        else:
            route.customers = original_customers
            
            # Force insert one by one
            for customer in customers:
                self._insert_one_customer_greedy(route, customer)
    
    def _optimize_large_insertions(self, route, customers):
        """
        Optimize insertions for larger sets using nearest neighbor heuristic.
        
        Args:
            route: Route to modify
            customers: Customers to insert
        """
        # Start with original customers
        original_customers = route.customers.copy()
        
        # Sort customers by ready time
        customers.sort(key=lambda c: c.ready_time)
        
        # Insert customers one by one using best insertion
        route.customers = original_customers.copy()
        for customer in customers:
            best_cost_increase = float('inf')
            best_position = -1
            
            # Calculate current route cost
            current_cost = route.calculate_cost()
            
            # Try each position
            for position in range(len(route.customers) + 1):
                # Insert customer
                route.customers.insert(position, customer)
                
                # Check feasibility
                is_feasible = route.is_time_window_feasible() and route.total_demand <= route.vehicle.capacity
                
                if is_feasible:
                    new_cost = route.calculate_cost()
                    cost_increase = new_cost - current_cost
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_position = position
                
                # Revert insertion
                route.customers.pop(position)
            
            # Insert at best position if found
            if best_position != -1:
                route.customers.insert(best_position, customer)
            else:
                # Fall back: just append at the end
                route.customers.append(customer)
    
    def _update_historical_knowledge(self, route, position, customer):
        """
        Update historical knowledge after inserting a customer.
        
        Args:
            route: Route where customer was inserted
            position: Position in the route
            customer: The inserted customer
        """
        route_idx = route.id
        
        # Update route history
        if customer not in self.customer_route_history:
            self.customer_route_history[customer] = []
        self.customer_route_history[customer].append(route_idx)
        
        # Update position history
        if customer not in self.customer_position_history:
            self.customer_position_history[customer] = []
        self.customer_position_history[customer].append(position)
        
        # Update customer pair history
        if position > 0 and position <= len(route.customers) - 1:
            # Get customers before and after
            before = route.customers[position - 1]
            
            pair = (min(before.id, customer.id), max(before.id, customer.id))
            if pair not in self.customer_pair_history:
                self.customer_pair_history[pair] = 0
            self.customer_pair_history[pair] += 1
        
        if position < len(route.customers) - 1:
            # Get customer after
            after = route.customers[position + 1]
            
            pair = (min(customer.id, after.id), max(customer.id, after.id))
            if pair not in self.customer_pair_history:
                self.customer_pair_history[pair] = 0
            self.customer_pair_history[pair] += 1
    
    def _insert_one_customer_greedy(self, solution_or_route, customer):
        """
        Insert a single customer using greedy approach.
        
        Args:
            solution_or_route: Solution or Route object
            customer: Customer to insert
            
        Returns:
            bool: True if inserted successfully
        """
        # Determine if working with solution or route
        if hasattr(solution_or_route, 'routes'):
            # It's a solution
            solution = solution_or_route
            routes = solution.routes
        else:
            # It's a route
            route = solution_or_route
            routes = [route]
        
        best_cost_increase = float('inf')
        best_route = None
        best_position = None
        
        # Try each route and position
        for route in routes:
            # Skip infeasible routes (capacity)
            if route.total_demand + customer.demand > route.vehicle.capacity:
                continue
            
            # Calculate current route cost
            current_cost = route.calculate_cost()
            
            # Try each position
            for position in range(len(route.customers) + 1):
                # Insert customer
                route.customers.insert(position, customer)
                
                # Check time window feasibility
                is_feasible = route.is_time_window_feasible()
                
                if is_feasible:
                    new_cost = route.calculate_cost()
                    cost_increase = new_cost - current_cost
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route = route
                        best_position = position
                
                # Revert insertion
                route.customers.pop(position)
        
        # Insert at best position if found
        if best_route is not None:
            best_route.customers.insert(best_position, customer)
            return True
        else:
            # No feasible insertion found
            if hasattr(solution_or_route, 'routes') and solution_or_route.can_add_route():
                # Create new route if working with solution
                new_route = solution_or_route.add_route()
                new_route.customers.append(customer)
                return True
            else:
                # Forced insertion at end of the first route
                if routes:
                    routes[0].customers.append(customer)
                    return True
        
        return False 