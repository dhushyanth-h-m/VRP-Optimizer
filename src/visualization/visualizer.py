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
            
            # Calculate arrival times for this route
            arrival_times = {}
            wait_times = {}
            current_time = route.driver_start_time
            current_node = self.solution.depot
            
            # Calculate time to first customer
            if route.customers:
                first_customer = route.customers[0]
                travel_time = route.vehicle.travel_time(current_node, first_customer)
                current_time += travel_time
                
                arrival_times[first_customer.id] = current_time
                
                # Calculate wait time
                wait_time = max(0, first_customer.ready_time - current_time)
                wait_times[first_customer.id] = wait_time
                
                # Update current time and node
                current_time += wait_time + first_customer.service_time
                current_node = first_customer
            
            # Calculate for subsequent customers
            for j in range(1, len(route.customers)):
                customer = route.customers[j]
                travel_time = route.vehicle.travel_time(current_node, customer)
                current_time += travel_time
                
                arrival_times[customer.id] = current_time
                
                # Calculate wait time
                wait_time = max(0, customer.ready_time - current_time)
                wait_times[customer.id] = wait_time
                
                # Update current time and node
                current_time += wait_time + customer.service_time
                current_node = customer
            
            # Add customer markers
            for j, customer in enumerate(route.customers):
                # Create popup content
                popup_content = f"""
                <b>Customer ID:</b> {customer.id}<br>
                <b>Demand:</b> {customer.demand}<br>
                <b>Time Window:</b> {customer.ready_time:.2f} - {customer.due_time:.2f}<br>
                <b>Service Time:</b> {customer.service_time:.2f}<br>
                <b>Arrival Time:</b> {arrival_times.get(customer.id, 0):.2f}<br>
                <b>Wait Time:</b> {wait_times.get(customer.id, 0):.2f}<br>
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
                    <div class="kpi-value">{max(0, summary["total_stops"])} / {summary["total_customers"]}</div>
                    <div class="kpi-description">Number of customers served vs. total</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {100 * max(0, summary["total_stops"]) / summary["total_customers"] if summary["total_customers"] > 0 else 0}%;"></div>
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
            utilization = route.total_demand / route.vehicle.capacity if route.vehicle.capacity > 0 else 0
            
            row = f"""
            <tr>
                <td>{i+1}</td>
                <td>{route.vehicle.id}</td>
                <td>{len(route.customers)}</td>
                <td>{distance:.2f}</td>
                <td>{duration:.2f}</td>
                <td>{route.total_demand}</td>
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
        utilizations = [100 * route.total_demand / route.vehicle.capacity if route.vehicle.capacity > 0 else 0 
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
    
    def plot_environmental_impact(self, output_file):
        """
        Create an environmental impact visualization.
        
        Args:
            output_file (str): Path to save the visualization
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Calculate metrics for each route
        route_data = []
        for route in self.solution.routes:
            if not route.customers:
                continue
                
            route_data.append({
                "route_id": route.id,
                "vehicle_id": route.vehicle.id,
                "distance": route.calculate_distance(),
                "co2_emissions": route.calculate_co2_emissions(),
                "customer_count": len(route.customers),
                "fuel_consumption": route.calculate_distance() * route.vehicle.fuel_consumption
            })
        
        if not route_data:
            return
            
        df = pd.DataFrame(route_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "CO2 Emissions by Route", 
                "Fuel Consumption by Route",
                "Emissions vs. Distance", 
                "Average Emissions per Customer"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Convert route IDs to strings for better display
        df["route_id_str"] = df["route_id"].apply(lambda x: f"Route {x}")
        
        # CO2 Emissions by Route
        fig.add_trace(
            go.Bar(
                x=df["route_id_str"], 
                y=df["co2_emissions"],
                name="CO2 Emissions (kg)",
                marker_color="green"
            ),
            row=1, col=1
        )
        
        # Fuel Consumption by Route
        fig.add_trace(
            go.Bar(
                x=df["route_id_str"], 
                y=df["fuel_consumption"],
                name="Fuel Consumption (L)",
                marker_color="blue"
            ),
            row=1, col=2
        )
        
        # Emissions vs Distance scatter plot
        fig.add_trace(
            go.Scatter(
                x=df["distance"], 
                y=df["co2_emissions"],
                mode="markers",
                name="CO2 vs Distance",
                marker=dict(
                    size=10,
                    color=df["vehicle_id"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Vehicle ID")
                ),
                text=df["route_id_str"],  # Add route ID as hover text
            ),
            row=2, col=1
        )
        
        # Average emissions per customer
        df["emissions_per_customer"] = df["co2_emissions"] / df["customer_count"]
        
        fig.add_trace(
            go.Bar(
                x=df["route_id_str"], 
                y=df["emissions_per_customer"],
                name="CO2 per Customer (kg)",
                marker_color="red"
            ),
            row=2, col=2
        )
        
        # Add overall eco-efficiency score
        total_co2 = df["co2_emissions"].sum()
        total_distance = df["distance"].sum()
        total_customers = df["customer_count"].sum()
        
        eco_efficiency = {
            "CO2 per km": f"{total_co2 / total_distance:.2f} kg/km",
            "CO2 per customer": f"{total_co2 / total_customers:.2f} kg/customer",
            "Total CO2": f"{total_co2:.2f} kg"
        }
        
        # Add annotations
        for i, (key, value) in enumerate(eco_efficiency.items()):
            fig.add_annotation(
                x=0.02, y=0.1 - (i * 0.05),
                xref="paper", yref="paper",
                text=f"<b>{key}:</b> {value}",
                showarrow=False,
                font=dict(size=14)
            )
        
        # Update layout
        fig.update_layout(
            title_text="Environmental Impact Analysis",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Route", row=1, col=1)
        fig.update_yaxes(title_text="CO2 Emissions (kg)", row=1, col=1)
        
        fig.update_xaxes(title_text="Route", row=1, col=2)
        fig.update_yaxes(title_text="Fuel Consumption (L)", row=1, col=2)
        
        fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
        fig.update_yaxes(title_text="CO2 Emissions (kg)", row=2, col=1)
        
        fig.update_xaxes(title_text="Route", row=2, col=2)
        fig.update_yaxes(title_text="CO2 per Customer (kg)", row=2, col=2)
        
        # If we have only one route, fix axis ranges
        if len(df) == 1:
            fig.update_xaxes(tickmode='array', tickvals=[df["route_id_str"].iloc[0]], row=1, col=1)
            fig.update_xaxes(tickmode='array', tickvals=[df["route_id_str"].iloc[0]], row=1, col=2)
            fig.update_xaxes(tickmode='array', tickvals=[df["route_id_str"].iloc[0]], row=2, col=2)
        
        # Save to file
        fig.write_html(output_file)
    
    def plot_traffic_patterns(self, output_file):
        """
        Create a visualization of traffic patterns.
        
        Args:
            output_file (str): Path to save the visualization
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        
        # Get the first vehicle to extract traffic patterns
        # (assuming all vehicles have the same pattern)
        if not self.solution.routes or not self.solution.routes[0].vehicle:
            return
            
        vehicle = self.solution.routes[0].vehicle
        
        if not hasattr(vehicle, 'time_dependent_speed_factors'):
            # No traffic patterns defined
            return
            
        # Extract traffic pattern data
        hours = list(range(24))
        speed_factors = [vehicle.time_dependent_speed_factors.get(h, 1.0) for h in hours]
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Traffic Speed Factors Throughout the Day",
                "Optimal Departure Times Analysis"
            ),
            specs=[
                [{"type": "scatter"}],
                [{"type": "heatmap"}]
            ],
            vertical_spacing=0.15
        )
        
        # Traffic pattern line chart
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=speed_factors,
                mode='lines+markers',
                name='Speed Factor',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Add reference line at y=1.0 (normal speed)
        fig.add_trace(
            go.Scatter(
                x=[0, 23],
                y=[1, 1],
                mode='lines',
                name='Normal Speed',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Convert to travel time factors for better visualization
        # (inverse of speed factor: lower speed = higher travel time)
        travel_time_factors = [1/factor for factor in speed_factors]
        
        # Create travel time matrix
        # For each starting hour, calculate the travel time for a journey taking 1-5 hours
        time_matrix = np.zeros((24, 5))
        
        for start_hour in range(24):
            for duration in range(1, 6):  # 1-5 hour journeys
                total_travel_time = 0
                current_hour = start_hour
                
                # Simulate travel
                for _ in range(duration):
                    travel_time_factor = 1/vehicle.time_dependent_speed_factors.get(current_hour, 1.0)
                    total_travel_time += travel_time_factor
                    current_hour = (current_hour + 1) % 24
                
                # Calculate efficiency (ratio of actual travel time to ideal travel time)
                efficiency = total_travel_time / duration
                time_matrix[start_hour, duration-1] = efficiency
        
        # Create heatmap of travel time efficiency
        fig.add_trace(
            go.Heatmap(
                z=time_matrix,
                x=[f"{i+1} hr" for i in range(5)],
                y=hours,
                colorscale='RdYlGn_r',  # Red=bad (high travel time), Green=good (low travel time)
                zmin=0.8,
                zmax=1.2,
                colorbar=dict(title="Travel Time<br>Efficiency"),
            ),
            row=2, col=1
        )
        
        # Add annotations explaining the heatmap
        fig.add_annotation(
            x=0.5, y=0.45,
            xref="paper", yref="paper",
            text="Lower values (green) indicate more efficient departure times",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Update layout
        fig.update_layout(
            title_text="Traffic Pattern Analysis",
            height=1000,
            width=1000
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Hour of Day (0-23)", row=1, col=1)
        fig.update_yaxes(title_text="Speed Factor (% of normal speed)", row=1, col=1, 
                        tickformat=".0%", range=[0, max(speed_factors) * 1.1])
        
        fig.update_xaxes(title_text="Journey Duration", row=2, col=1)
        fig.update_yaxes(title_text="Departure Hour", row=2, col=1)
        
        # Save to file
        fig.write_html(output_file)
    
    def plot_fleet_composition(self, output_file):
        """
        Create a visualization of the heterogeneous fleet composition.
        
        Args:
            output_file (str): Path to save the visualization
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Extract vehicle data
        vehicle_data = []
        for route in self.solution.routes:
            if route.vehicle:
                vehicle_data.append({
                    "vehicle_id": route.vehicle.id,
                    "capacity": route.vehicle.capacity,
                    "speed": route.vehicle.speed,
                    "fixed_cost": route.vehicle.fixed_cost,
                    "variable_cost": route.vehicle.variable_cost,
                    "fuel_consumption": route.vehicle.fuel_consumption,
                    "is_used": len(route.customers) > 0,
                    "customers_served": len(route.customers),
                    "utilization": route.total_demand / route.vehicle.capacity if route.vehicle.capacity > 0 else 0,
                    "distance": route.calculate_distance() if len(route.customers) > 0 else 0
                })
        
        if not vehicle_data:
            return
            
        df = pd.DataFrame(vehicle_data)
        
        # Group vehicles by capacity to identify types
        # Determine vehicle types based on capacity
        capacity_groups = df['capacity'].unique()
        capacity_groups.sort()
        
        # Assign type names based on capacity
        vehicle_types = []
        for capacity in capacity_groups:
            if capacity <= 50:
                vehicle_types.append("Small Van")
            elif capacity <= 100:
                vehicle_types.append("Medium Truck")
            elif capacity <= 200:
                vehicle_types.append("Large Truck")
            else:
                vehicle_types.append("Extra Large Truck")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fleet Composition by Vehicle Type", 
                "Vehicle Utilization",
                "Vehicle Capacity vs. Distance", 
                "Cost Structure by Vehicle Type"
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Add vehicle type information to dataframe
        df['vehicle_type'] = None
        for i, capacity in enumerate(capacity_groups):
            df.loc[df['capacity'] == capacity, 'vehicle_type'] = vehicle_types[i]
        
        # Fleet composition pie chart
        vehicle_type_counts = df['vehicle_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=vehicle_type_counts.index,
                values=vehicle_type_counts.values,
                textinfo='percent+label',
                marker=dict(colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
            ),
            row=1, col=1
        )
        
        # Vehicle utilization bar chart
        # Only include used vehicles
        used_vehicles = df[df['is_used']]
        if not used_vehicles.empty:
            fig.add_trace(
                go.Bar(
                    x=used_vehicles['vehicle_id'],
                    y=used_vehicles['utilization'] * 100,  # Convert to percentage
                    marker_color=used_vehicles['vehicle_type'].map({
                        'Small Van': '#66c2a5',
                        'Medium Truck': '#fc8d62',
                        'Large Truck': '#8da0cb',
                        'Extra Large Truck': '#e78ac3'
                    }),
                    text=used_vehicles['vehicle_type'],
                    hovertemplate='<b>Vehicle %{x}</b><br>Utilization: %{y:.1f}%<br>Type: %{text}'
                ),
                row=1, col=2
            )
        
        # Capacity vs. Distance scatter plot
        fig.add_trace(
            go.Scatter(
                x=df['capacity'],
                y=df['distance'],
                mode='markers',
                marker=dict(
                    size=df['customers_served'] * 5,
                    color=df['vehicle_type'].map({
                        'Small Van': '#66c2a5',
                        'Medium Truck': '#fc8d62',
                        'Large Truck': '#8da0cb',
                        'Extra Large Truck': '#e78ac3'
                    }),
                    line=dict(width=1, color='black')
                ),
                text=df['vehicle_type'],
                hovertemplate='<b>Capacity: %{x}</b><br>Distance: %{y:.1f} km<br>Type: %{text}<br>Customers: %{marker.size:.0f}'
            ),
            row=2, col=1
        )
        
        # Cost structure by vehicle type
        cost_by_type = df.groupby('vehicle_type').agg({
            'fixed_cost': 'mean',
            'variable_cost': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=cost_by_type['vehicle_type'],
                y=cost_by_type['fixed_cost'],
                name='Fixed Cost',
                marker_color='#1f77b4'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=cost_by_type['vehicle_type'],
                y=cost_by_type['variable_cost'] * 100,  # Scale up to be visible on same chart
                name='Variable Cost (per km) x100',
                marker_color='#ff7f0e'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Fleet Composition Analysis",
            height=1000,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Utilization (%)", row=1, col=2)
        fig.update_xaxes(title_text="Vehicle ID", row=1, col=2)
        
        fig.update_xaxes(title_text="Vehicle Capacity", row=2, col=1)
        fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
        
        fig.update_xaxes(title_text="Vehicle Type", row=2, col=2)
        fig.update_yaxes(title_text="Cost", row=2, col=2)
        
        # Save to file
        fig.write_html(output_file) 