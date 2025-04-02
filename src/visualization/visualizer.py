"""
Visualizer module for the VRP Optimizer.
"""

import logging
import os
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
import random

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualizer class for VRP solutions."""
    
    def __init__(self, solution):
        """
        Initialize visualizer with a solution.
        
        Args:
            solution: VRP solution
        """
        self.solution = solution
        self.route_colors = [
            '#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#A633FF', '#33FFF9',
            '#FFBD33', '#7DFF33', '#33FFBD', '#FF33BD', '#BD33FF', '#33BDFF'
        ]
        
        # If we have more routes than colors, generate random colors
        if len(self.solution.routes) > len(self.route_colors):
            # Generate more random colors
            for _ in range(len(self.solution.routes) - len(self.route_colors)):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self.route_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    def plot_routes(self, output_file):
        """
        Plot routes on a map using folium.
        
        Args:
            output_file (str): Output file path
        """
        # Calculate center of the map
        all_lats = [self.solution.depot.y]
        all_lons = [self.solution.depot.x]
        
        for route in self.solution.routes:
            for customer in route.customers:
                all_lats.append(customer.y)
                all_lons.append(customer.x)
        
        center_lat = sum(all_lats) / len(all_lats) if all_lats else 0
        center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
        
        # Create a folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add depot marker
        folium.Marker(
            location=[self.solution.depot.y, self.solution.depot.x],
            popup=f"Depot (ID: {self.solution.depot.id})",
            icon=folium.Icon(color='red', icon='home'),
        ).add_to(m)
        
        # Add routes
        for i, route in enumerate(self.solution.routes):
            if not route.customers:
                continue
                
            # Get color for this route
            color = self.route_colors[i % len(self.route_colors)]
            
            # Create a feature group for this route
            route_group = folium.FeatureGroup(name=f"Route {i+1}")
            
            # Add customer markers
            for j, customer in enumerate(route.customers):
                # Create popup content
                popup_content = f"""
                <b>Customer ID:</b> {customer.id}<br>
                <b>Demand:</b> {customer.demand}<br>
                <b>Time Window:</b> {customer.ready_time:.2f} - {customer.due_time:.2f}<br>
                <b>Service Time:</b> {customer.service_time:.2f}<br>
                <b>Arrival Time:</b> {route.arrival_times.get(customer.id, 'N/A'):.2f}<br>
                <b>Wait Time:</b> {route.wait_times.get(customer.id, 0):.2f}<br>
                """
                
                # Add marker
                folium.Marker(
                    location=[customer.y, customer.x],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue', icon='info-sign'),
                ).add_to(route_group)
                
                # Add customer number
                folium.CircleMarker(
                    location=[customer.y, customer.x],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Stop {j+1}",
                ).add_to(route_group)
            
            # Add lines connecting nodes in the route
            route_points = [(self.solution.depot.y, self.solution.depot.x)]
            for customer in route.customers:
                route_points.append((customer.y, customer.x))
            route_points.append((self.solution.depot.y, self.solution.depot.x))
            
            folium.PolyLine(
                route_points,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Route {i+1}: {len(route.customers)} customers, {route.calculate_total_distance():.2f} km",
            ).add_to(route_group)
            
            # Add the route group to the map
            route_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map to file
        m.save(output_file)
        logger.info(f"Routes map saved to {output_file}")
    
    def create_kpi_dashboard(self, output_file):
        """
        Create a KPI dashboard as an HTML file.
        
        Args:
            output_file (str): Output file path
        """
        # Get solution summary
        summary = self.solution.get_summary()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VRP Optimizer - KPI Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    margin-top: 20px;
                }}
                .kpi-card {{
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    width: 220px;
                    text-align: center;
                }}
                .kpi-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                    color: #3498db;
                }}
                .kpi-title {{
                    font-size: 14px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                }}
                .kpi-description {{
                    font-size: 12px;
                    color: #95a5a6;
                    margin-top: 10px;
                }}
                .chart-container {{
                    width: 100%;
                    margin-top: 30px;
                    text-align: center;
                }}
                .route-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 30px;
                }}
                .route-table th, .route-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .route-table th {{
                    background-color: #f2f2f2;
                }}
                .route-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .route-table tr:hover {{
                    background-color: #f5f5f5;
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
            <h1>VRP Optimizer - KPI Dashboard</h1>
            
            <div class="container">
                <div class="kpi-card">
                    <div class="kpi-title">Total Distance</div>
                    <div class="kpi-value">{summary["total_distance"]:.2f} km</div>
                    <div class="kpi-description">Total distance traveled by all vehicles</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Total Time</div>
                    <div class="kpi-value">{summary["total_time"]:.2f} h</div>
                    <div class="kpi-description">Total time spent on routes</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Vehicles Used</div>
                    <div class="kpi-value">{summary["num_vehicles_used"]} / {summary["total_vehicles"]}</div>
                    <div class="kpi-description">Number of vehicles used vs. available</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {100 * summary["num_vehicles_used"] / summary["total_vehicles"] if summary["total_vehicles"] > 0 else 0}%;"></div>
                    </div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Capacity Utilization</div>
                    <div class="kpi-value">{100 * summary["capacity_utilization"]:.1f}%</div>
                    <div class="kpi-description">Average vehicle capacity utilization</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {100 * summary["capacity_utilization"]}%;"></div>
                    </div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Fuel Consumption</div>
                    <div class="kpi-value">{summary["total_fuel"]:.2f} L</div>
                    <div class="kpi-description">Total estimated fuel consumption</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Total Cost</div>
                    <div class="kpi-value">${summary["total_cost"]:.2f}</div>
                    <div class="kpi-description">Total operational cost</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Avg. Stops Per Route</div>
                    <div class="kpi-value">{summary["avg_stops_per_route"]:.1f}</div>
                    <div class="kpi-description">Average number of stops per route</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-title">Customers Served</div>
                    <div class="kpi-value">{summary["total_customers"] - summary["unassigned_customers"]} / {summary["total_customers"]}</div>
                    <div class="kpi-description">Number of customers served vs. total</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {100 * (summary["total_customers"] - summary["unassigned_customers"]) / summary["total_customers"] if summary["total_customers"] > 0 else 0}%;"></div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Route Details</h2>
                <table class="route-table">
                    <tr>
                        <th>Route #</th>
                        <th>Vehicle ID</th>
                        <th>Customers</th>
                        <th>Distance (km)</th>
                        <th>Duration (h)</th>
                        <th>Load</th>
                        <th>Capacity</th>
                        <th>Utilization</th>
                    </tr>
                    {self._generate_route_table_rows()}
                </table>
            </div>
            
            <div class="chart-container">
                <p><i>Generated by VRP Optimizer</i></p>
            </div>
        </body>
        </html>
        """
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"KPI dashboard saved to {output_file}")
    
    def _generate_route_table_rows(self):
        """Generate HTML table rows for route details."""
        rows = []
        
        for i, route in enumerate(self.solution.routes):
            if not route.customers:
                continue
                
            distance = route.calculate_total_distance()
            duration = route.calculate_total_time()
            utilization = route.load / route.vehicle.capacity if route.vehicle.capacity > 0 else 0
            
            row = f"""
            <tr>
                <td>{i+1}</td>
                <td>{route.vehicle.id}</td>
                <td>{len(route.customers)}</td>
                <td>{distance:.2f}</td>
                <td>{duration:.2f}</td>
                <td>{route.load}</td>
                <td>{route.vehicle.capacity}</td>
                <td>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {100 * utilization:.1f}%;"></div>
                    </div>
                    {100 * utilization:.1f}%
                </td>
            </tr>
            """
            rows.append(row)
        
        return ''.join(rows)
    
    def plot_metrics(self, output_file):
        """
        Plot solution metrics using matplotlib.
        
        Args:
            output_file (str): Output file path
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VRP Solution Metrics', fontsize=16)
        
        # 1. Route distances
        distances = [route.calculate_total_distance() for route in self.solution.routes if route.customers]
        labels = [f"Route {i+1}" for i in range(len(distances))]
        
        axes[0, 0].bar(labels, distances, color=self.route_colors[:len(distances)])
        axes[0, 0].set_title('Distance per Route')
        axes[0, 0].set_ylabel('Distance (km)')
        axes[0, 0].set_xlabel('Route')
        for tick in axes[0, 0].get_xticklabels():
            tick.set_rotation(45)
        
        # 2. Vehicle utilization
        utilizations = [100 * route.load / route.vehicle.capacity if route.vehicle.capacity > 0 else 0 
                        for route in self.solution.routes if route.customers]
        
        axes[0, 1].bar(labels, utilizations, color=self.route_colors[:len(utilizations)])
        axes[0, 1].set_title('Capacity Utilization per Route')
        axes[0, 1].set_ylabel('Utilization (%)')
        axes[0, 1].set_xlabel('Route')
        axes[0, 1].set_ylim(0, 100)
        for tick in axes[0, 1].get_xticklabels():
            tick.set_rotation(45)
        
        # 3. Customers per route
        customers_per_route = [len(route.customers) for route in self.solution.routes if route.customers]
        
        axes[1, 0].bar(labels, customers_per_route, color=self.route_colors[:len(customers_per_route)])
        axes[1, 0].set_title('Customers per Route')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_xlabel('Route')
        for tick in axes[1, 0].get_xticklabels():
            tick.set_rotation(45)
        
        # 4. Pie chart of unassigned vs. assigned customers
        unassigned = len(self.solution.unassigned)
        assigned = len(self.solution.customers) - unassigned
        
        axes[1, 1].pie([assigned, unassigned], labels=['Assigned', 'Unassigned'], autopct='%1.1f%%', 
                      colors=['#3498db', '#e74c3c'], startangle=90)
        axes[1, 1].set_title('Customer Assignment')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file)
        logger.info(f"Metrics plot saved to {output_file}")
        plt.close(fig) 