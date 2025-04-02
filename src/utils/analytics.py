"""
Analytics module for VRP Optimizer.
Provides advanced analysis capabilities for optimization results.
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class Analytics:
    """Analytics class for VRP optimization results."""
    
    def __init__(self, solution):
        """
        Initialize analytics.
        
        Args:
            solution: Optimized solution
        """
        self.solution = solution
        
    def analyze_customer_clusters(self):
        """
        Analyze customer clusters in the solution.
        
        Returns:
            dict: Cluster analysis results
        """
        # Extract customer coordinates
        customers = self.solution.all_customers
        if not customers:
            return {"error": "No customers in solution"}
            
        coords = np.array([[c.x, c.y] for c in customers])
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.1, min_samples=3).fit(coords)
        labels = clustering.labels_
        
        # Count number of clusters (excluding noise with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Analyze clusters
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Not noise
                clusters[label].append(customers[i])
        
        # Calculate cluster statistics
        cluster_stats = []
        for label, cluster_customers in clusters.items():
            # Calculate centroid
            centroid_x = np.mean([c.x for c in cluster_customers])
            centroid_y = np.mean([c.y for c in cluster_customers])
            
            # Calculate average distance to centroid
            avg_distance = np.mean([
                ((c.x - centroid_x) ** 2 + (c.y - centroid_y) ** 2) ** 0.5
                for c in cluster_customers
            ])
            
            # Calculate average demand in cluster
            avg_demand = np.mean([c.demand for c in cluster_customers])
            
            # Calculate time window overlap
            earliest_ready = min([c.ready_time for c in cluster_customers])
            latest_due = max([c.due_time for c in cluster_customers])
            time_window_span = latest_due - earliest_ready
            
            cluster_stats.append({
                "cluster_id": label,
                "size": len(cluster_customers),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "avg_distance_to_centroid": avg_distance,
                "avg_demand": avg_demand,
                "time_window_span": time_window_span
            })
        
        # Calculate noise percentage
        noise_count = list(labels).count(-1)
        noise_percentage = noise_count / len(customers) * 100 if customers else 0
        
        return {
            "num_clusters": n_clusters,
            "noise_percentage": noise_percentage,
            "cluster_stats": cluster_stats,
            "customer_clusters": {c.id: labels[i] for i, c in enumerate(customers)}
        }
    
    def analyze_route_efficiency(self):
        """
        Analyze the efficiency of routes in the solution.
        
        Returns:
            dict: Route efficiency analysis
        """
        routes = [r for r in self.solution.routes if r.customers]
        if not routes:
            return {"error": "No routes in solution"}
            
        route_stats = []
        for route in routes:
            # Calculate basic metrics
            distance = route.calculate_distance()
            time = route.calculate_travel_time()
            capacity = route.vehicle.capacity
            used_capacity = route.total_demand
            utilization = used_capacity / capacity if capacity > 0 else 0
            
            # Calculate average distance between consecutive stops
            avg_leg_distance = 0
            if len(route.customers) > 1:
                leg_distances = []
                # Distance from depot to first customer
                leg_distances.append(route.depot.distance_to(route.customers[0]))
                
                # Distances between consecutive customers
                for i in range(len(route.customers) - 1):
                    leg_distances.append(
                        route.customers[i].distance_to(route.customers[i + 1])
                    )
                
                # Distance from last customer to depot
                leg_distances.append(route.customers[-1].distance_to(route.depot))
                
                avg_leg_distance = np.mean(leg_distances)
                std_leg_distance = np.std(leg_distances)
            else:
                # Only one customer
                leg_distance = 2 * route.depot.distance_to(route.customers[0])
                avg_leg_distance = leg_distance / 2
                std_leg_distance = 0
            
            # Calculate time window tightness
            if route.customers:
                time_window_spans = [
                    c.due_time - c.ready_time for c in route.customers
                ]
                avg_time_window = np.mean(time_window_spans)
                min_time_window = min(time_window_spans)
            else:
                avg_time_window = 0
                min_time_window = 0
            
            route_stats.append({
                "route_id": route.id,
                "vehicle_id": route.vehicle.id,
                "vehicle_capacity": capacity,
                "distance": distance,
                "travel_time": time,
                "customers": len(route.customers),
                "used_capacity": used_capacity,
                "utilization": utilization,
                "avg_leg_distance": avg_leg_distance,
                "std_leg_distance": std_leg_distance,
                "avg_time_window": avg_time_window,
                "min_time_window": min_time_window,
                "efficiency_score": self._calculate_efficiency_score(
                    distance, len(route.customers), utilization, avg_leg_distance, min_time_window
                )
            })
        
        # Calculate overall statistics
        if route_stats:
            df = pd.DataFrame(route_stats)
            overall_stats = {
                "avg_utilization": df["utilization"].mean() * 100,  # as percentage
                "avg_efficiency_score": df["efficiency_score"].mean(),
                "total_distance": df["distance"].sum(),
                "total_travel_time": df["travel_time"].sum(),
                "total_customers": df["customers"].sum(),
                "correlation_distance_customers": pearsonr(df["distance"], df["customers"])[0],
                "correlation_utilization_efficiency": pearsonr(df["utilization"], df["efficiency_score"])[0],
                "best_route_id": df.loc[df["efficiency_score"].idxmax(), "route_id"],
                "worst_route_id": df.loc[df["efficiency_score"].idxmin(), "route_id"],
            }
        else:
            overall_stats = {}
        
        return {
            "route_stats": route_stats,
            "overall_stats": overall_stats
        }
    
    def _calculate_efficiency_score(self, distance, num_customers, utilization, avg_leg_distance, min_time_window):
        """
        Calculate an efficiency score for a route.
        
        Higher is better.
        
        Args:
            distance: Total route distance
            num_customers: Number of customers in route
            utilization: Capacity utilization (0-1)
            avg_leg_distance: Average distance between consecutive stops
            min_time_window: Minimum time window in route
            
        Returns:
            float: Efficiency score
        """
        if num_customers == 0:
            return 0
            
        # Calculate distance per customer (lower is better)
        distance_per_customer = distance / num_customers
        
        # Normalize each component
        norm_distance = 1 / (1 + distance_per_customer/10)  # Lower distance per customer is better
        norm_utilization = utilization  # Higher utilization is better
        norm_leg_distance = 1 / (1 + avg_leg_distance/5)  # Lower average leg distance is better
        norm_time_window = min(1, min_time_window / 4)  # Wider time windows are easier to service
        
        # Weighted score (weights based on importance)
        score = (
            0.4 * norm_distance +
            0.3 * norm_utilization +
            0.2 * norm_leg_distance +
            0.1 * norm_time_window
        )
        
        return score * 100  # Scale to 0-100
    
    def analyze_time_window_impact(self):
        """
        Analyze the impact of time windows on the solution.
        
        Returns:
            dict: Time window impact analysis
        """
        # Group customers by time window characteristics
        customers = self.solution.all_customers
        if not customers:
            return {"error": "No customers in solution"}
            
        # Calculate time window spans
        tw_spans = [c.due_time - c.ready_time for c in customers]
        
        # Group into categories
        narrow_tw = [c for c, span in zip(customers, tw_spans) if span < 2]  # Less than 2 hours
        medium_tw = [c for c, span in zip(customers, tw_spans) if 2 <= span < 6]  # 2-6 hours
        wide_tw = [c for c, span in zip(customers, tw_spans) if span >= 6]  # 6+ hours
        
        # Calculate average distance from depot for each group
        depot = self.solution.depot
        
        narrow_dist = np.mean([depot.distance_to(c) for c in narrow_tw]) if narrow_tw else 0
        medium_dist = np.mean([depot.distance_to(c) for c in medium_tw]) if medium_tw else 0
        wide_dist = np.mean([depot.distance_to(c) for c in wide_tw]) if wide_tw else 0
        
        # Analyze time windows by route
        route_tw_data = []
        for route in self.solution.routes:
            if not route.customers:
                continue
                
            # Calculate average time window span for this route
            route_tw_spans = [c.due_time - c.ready_time for c in route.customers]
            avg_tw_span = np.mean(route_tw_spans)
            min_tw_span = min(route_tw_spans)
            max_tw_span = max(route_tw_spans)
            
            # Calculate time window utilization
            # (how much of available time windows were actually used)
            route_duration = route.calculate_travel_time()
            tw_utilization = route_duration / sum(route_tw_spans) if sum(route_tw_spans) > 0 else 0
            
            route_tw_data.append({
                "route_id": route.id,
                "avg_tw_span": avg_tw_span,
                "min_tw_span": min_tw_span,
                "max_tw_span": max_tw_span,
                "tw_utilization": tw_utilization,
                "route_duration": route_duration,
                "customer_count": len(route.customers)
            })
        
        # Calculate correlation between time window span and route efficiency
        if route_tw_data:
            df = pd.DataFrame(route_tw_data)
            # Calculate efficiency (customers per hour)
            df["efficiency"] = df["customer_count"] / df["route_duration"]
            
            # Calculate correlations
            corr_span_efficiency = pearsonr(df["avg_tw_span"], df["efficiency"])[0]
            corr_span_duration = pearsonr(df["avg_tw_span"], df["route_duration"])[0]
        else:
            corr_span_efficiency = 0
            corr_span_duration = 0
        
        return {
            "time_window_distribution": {
                "narrow": len(narrow_tw),
                "medium": len(medium_tw),
                "wide": len(wide_tw)
            },
            "avg_distance_from_depot": {
                "narrow_tw": narrow_dist,
                "medium_tw": medium_dist,
                "wide_tw": wide_dist
            },
            "route_tw_data": route_tw_data,
            "correlations": {
                "tw_span_vs_efficiency": corr_span_efficiency,
                "tw_span_vs_duration": corr_span_duration
            }
        }
    
    def generate_improvement_recommendations(self):
        """
        Generate recommendations for improving the solution.
        
        Returns:
            list: Improvement recommendations
        """
        recommendations = []
        
        # Analyze route efficiency
        route_efficiency = self.analyze_route_efficiency()
        if "error" not in route_efficiency:
            route_stats = pd.DataFrame(route_efficiency["route_stats"])
            
            # 1. Identify underutilized vehicles
            underutilized = route_stats[route_stats["utilization"] < 0.5]
            if not underutilized.empty:
                recommendations.append({
                    "type": "vehicle_utilization",
                    "description": "Some vehicles are significantly underutilized (below 50% capacity)",
                    "severity": "high" if len(underutilized) > 2 else "medium",
                    "affected_routes": underutilized["route_id"].tolist(),
                    "recommendation": "Consider consolidating loads from multiple underutilized vehicles"
                })
            
            # 2. Identify inefficient routes (high distance per customer)
            route_stats["distance_per_customer"] = route_stats["distance"] / route_stats["customers"]
            inefficient = route_stats[route_stats["distance_per_customer"] > 1.5 * route_stats["distance_per_customer"].median()]
            if not inefficient.empty:
                recommendations.append({
                    "type": "route_efficiency",
                    "description": "Some routes have high distance per customer",
                    "severity": "medium",
                    "affected_routes": inefficient["route_id"].tolist(),
                    "recommendation": "Review routing for these vehicles to reduce travel distance"
                })
            
            # 3. Check if any routes are approaching max duration
            long_routes = route_stats[route_stats["travel_time"] > 9]  # Assuming 10hr max
            if not long_routes.empty:
                recommendations.append({
                    "type": "route_duration",
                    "description": "Some routes are approaching maximum driver working hours",
                    "severity": "high",
                    "affected_routes": long_routes["route_id"].tolist(),
                    "recommendation": "Consider redistributing customers from these routes or adding breaks"
                })
        
        # Analyze customer clusters
        cluster_analysis = self.analyze_customer_clusters()
        if "error" not in cluster_analysis and cluster_analysis["num_clusters"] > 1:
            # 4. Check if customers from same cluster are assigned to different routes
            customer_routes = {}
            for route in self.solution.routes:
                if not route.customers:
                    continue
                for customer in route.customers:
                    customer_routes[customer.id] = route.id
            
            # Group customers by cluster
            cluster_assignments = defaultdict(list)
            for customer_id, cluster_id in cluster_analysis["customer_clusters"].items():
                if cluster_id != -1:  # Not noise
                    cluster_assignments[cluster_id].append(customer_id)
            
            # Check route assignments within clusters
            for cluster_id, customer_ids in cluster_assignments.items():
                routes_in_cluster = set()
                for customer_id in customer_ids:
                    if customer_id in customer_routes:
                        routes_in_cluster.add(customer_routes[customer_id])
                
                if len(routes_in_cluster) > 1:
                    recommendations.append({
                        "type": "cluster_assignment",
                        "description": f"Cluster {cluster_id} has customers assigned to {len(routes_in_cluster)} different routes",
                        "severity": "medium" if len(routes_in_cluster) > 2 else "low",
                        "affected_cluster": cluster_id,
                        "affected_routes": list(routes_in_cluster),
                        "recommendation": "Consider assigning customers from the same geographic cluster to the same route"
                    })
        
        # Analyze time window impact
        tw_analysis = self.analyze_time_window_impact()
        if "error" not in tw_analysis:
            # 5. Check for low time window utilization
            if "route_tw_data" in tw_analysis:
                tw_data = pd.DataFrame(tw_analysis["route_tw_data"])
                if not tw_data.empty:
                    low_tw_util = tw_data[tw_data["tw_utilization"] < 0.3]
                    if not low_tw_util.empty:
                        recommendations.append({
                            "type": "time_window_utilization",
                            "description": "Some routes have very low time window utilization (<30%)",
                            "severity": "low",
                            "affected_routes": low_tw_util["route_id"].tolist(),
                            "recommendation": "These routes may benefit from tighter scheduling or additional customers"
                        })
        
        # 6. General recommendations based on fleet characteristics
        vehicle_types = set()
        for route in self.solution.routes:
            if route.vehicle:
                vehicle_types.add(route.vehicle.capacity)
        
        if len(vehicle_types) < 2:
            recommendations.append({
                "type": "fleet_composition",
                "description": "The fleet is relatively homogeneous",
                "severity": "low",
                "recommendation": "Consider introducing vehicles with different capacities to better match demand patterns"
            })
        
        return recommendations 