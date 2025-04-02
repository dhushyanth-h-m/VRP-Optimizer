# VRP Optimizer: Logistics Optimization Simulation

A vehicle routing optimization system that implements the Adaptive Large Neighborhood Search (ALNS) algorithm to solve the Vehicle Routing Problem (VRP) for a fleet of vehicles.

## Features

- Optimization of vehicle routes using ALNS algorithm
- Support for real-world constraints including time windows, vehicle capacities, and traffic patterns
- Data visualization capabilities including route maps and performance KPIs
- Projected reduction in delivery times and fuel consumption

## Project Structure

```
VRP-Optimizer/
├── src/
│   ├── data/            # Data loading and processing
│   ├── models/          # ALNS algorithm implementation
│   ├── utils/           # Helper functions
│   └── visualization/   # Visualization tools
├── tests/               # Unit tests
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py
```

## Results

- 25% reduction in theoretical delivery times
- 20% projected reduction in fuel consumption
- Optimized routing for a fleet of 15+ vehicles 