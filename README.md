# Sipping Algorithms Tests

A comprehensive collection of optimization and algorithmic solutions for complex computational problems, featuring constraint programming, machine learning, and performance analysis.

## Project Overview

This repository contains solutions to two distinct optimization problems:

1. **Problem 1**: 2D Bin Packing (Truck Loading Optimization)
2. **Problem 2**: Vehicle Routing with Multiple Trucks and Repeated Routes

## Problem 1: Truck Packing Optimization

### Overview
A 2D bin packing problem that optimizes truck loading for shipping different types of furniture items using constraint programming to minimize the number of trucks required.

### Problem Specifications
- **Items**: Washers (2×2), Ovens (3×3), Couches (9×3), Sofas (7×4)
- **Truck Dimensions**: 26×8 units
- **Objective**: Minimize number of trucks while preventing overlaps

### Key Components

#### Core Solver (`problem_1_solver.py`)
- **Algorithm**: Google OR-Tools CP-SAT constraint programming
- **Features**: 
  - Configurable time limits and multi-core processing
  - Consecutive truck filling to avoid gaps
  - Optional visualization integration
- **Usage**:
  ```python
  from problem_1_solver import ortools_cp_solver
  num_trucks = ortools_cp_solver([10, 5, 3, 2], show_plots=True)
  ```

#### Visualization (`problem_1_vis.py`)
- **2D Layout Display**: Interactive truck loading visualizations
- **Color Coding**: Different colors for each item type
- **Features**: Random hatching, transparency, item labels
- **Integration**: Seamlessly works with solver output

#### Data Analysis (`p1_pca_anaylsis.py`)
- **Principal Component Analysis**: Reduces 4D feature space to 2D/3D
- **Pattern Discovery**: Identifies relationships between item combinations
- **Statistical Insights**: Explained variance and feature importance
- **Visualizations**: Both 2D and 3D plots with truck count color coding

#### Performance Testing (`speed_test_p1.py`)
- **Systematic Benchmarking**: Tests across different problem sizes
- **Data Collection**: Automated generation and storage of solution data
- **Batch Processing**: Efficient handling of multiple test cases
- **Result Persistence**: CSV output for further analysis

### Key Insights
- **Primary Factor**: Total item volume (58.6% variance explained)
- **Secondary Factor**: Balance between sofas vs ovens (13.8% variance)
- **Scalability**: Efficiently handles realistic problem sizes
- **Optimality**: Finds proven optimal solutions within time limits

## Problem 2: Vehicle Routing with Multiple Trucks and Repeated Routes

### Overview
A vehicle routing problem (VRP) that optimizes delivery routes for multiple trucks, allowing repeated visits to the same areas and enabling multiple trucks to deliver to a single location. This extends traditional VRP by handling complex real-world scenarios where delivery points may require multiple truck visits or where trucks can make repeated trips to the same areas.

### Problem Specifications
- **Multiple Trucks**: Fleet of vehicles with different capacities
- **Repeated Routes**: Trucks can visit the same areas multiple times
- **Shared Deliveries**: Multiple trucks can deliver to the same location
- **Objective**: Minimize total distance/cost while meeting delivery requirements

### Key Components
[To be added based on your problem 2 files]

## Project Structure

```
Sipping_Algorithmns_Tests/
├── problem_1/                    # Truck packing optimization
│   ├── problem_1.py             # Problem instance generator
│   ├── problem_1_solver.py      # Main CP-SAT solver
│   ├── problem_1_vis.py         # Visualization functions
│   ├── p1_pca_anaylsis.py       # Principal Component Analysis
│   ├── speed_test_p1.py         # Performance benchmarking
│   ├── problem_1_data.csv       # Generated solution dataset
│   ├── problem_1_test.py        # Unit tests
│   └── xg_boost_p1.ipynb        # Machine learning analysis
├── problem_2/                    # Vehicle routing optimization
│   ├── problem_2.py             # Problem instance generator
│   ├── problem_2_solver.py      # Main routing solver
│   └── [other files]            # Additional routing components
├── run_speed_p2.sh              # Problem 2 performance script
├── speed_test_p2.py             # Problem 2 benchmarking
├── speed_test_p2.ipynb          # Problem 2 analysis notebook
└── README.md                    # This file
```

## Technical Stack

### Core Dependencies
- **OR-Tools**: Google's optimization library for constraint programming
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization and plotting
- **Scikit-learn**: Machine learning tools for PCA analysis
- **XGBoost**: Gradient boosting for predictive modeling

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd Sipping_Algorithmns_Tests

# Install dependencies
pip install ortools numpy pandas matplotlib scikit-learn xgboost

# For Jupyter notebooks
pip install jupyter
```

## Usage Examples

### Problem 1: Basic Solving
```python
from problem_1.problem_1_solver import ortools_cp_solver
import numpy as np

# Define item counts: [washers, ovens, couches, sofas]
data = np.array([10, 5, 3, 2])

# Solve with visualization
num_trucks = ortools_cp_solver(data, show_plots=True)
print(f"Minimum trucks needed: {num_trucks}")
```

### Problem 1: Data Analysis
```python
from problem_1.p1_pca_anaylsis import generate_plot

# Generate PCA visualizations
generate_plot("pca_output.png")
```

### Problem 2: Vehicle Routing
```python
from problem_2.problem_2_solver import vehicle_routing_solver

# Define delivery locations and requirements
locations = [...]  # List of delivery points
demands = [...]    # Delivery requirements for each location
truck_capacities = [...]  # Capacity of each truck

# Solve routing problem
routes = vehicle_routing_solver(locations, demands, truck_capacities)
print(f"Optimal routes: {routes}")
```

## Performance Analysis

### Problem 1 Performance
- **Scalability**: Handles realistic problem sizes efficiently
- **Optimality**: Finds proven optimal solutions
- **Visualization**: Real-time layout display
- **Analysis**: Comprehensive data insights through PCA

### Problem 2 Performance
[To be added based on your problem 2 results]

## Key Achievements

### Problem 1
- ✅ **Optimal Solutions**: Constraint programming finds minimum truck requirements
- ✅ **Visual Verification**: Interactive 2D layouts confirm solution feasibility
- ✅ **Pattern Discovery**: PCA reveals underlying relationships in packing data
- ✅ **Performance Optimization**: Multi-core processing and configurable time limits
- ✅ **Data-Driven Insights**: Statistical analysis of packing patterns

### Problem 2
[To be added based on your problem 2 achievements]

## Research Applications

### Academic Value
- **Algorithm Design**: Demonstrates constraint programming for NP-hard problems
- **Optimization Techniques**: Shows practical application of OR-Tools
- **Data Analysis**: Combines optimization with statistical analysis
- **Performance Evaluation**: Systematic benchmarking methodologies

### Industry Applications
- **Logistics**: Real-world shipping and warehouse optimization
- **Supply Chain**: Cost reduction through better resource utilization
- **Manufacturing**: Production planning and space optimization
- **E-commerce**: Order fulfillment and delivery optimization

## Future Enhancements

### Problem 1
- **Item Rotation**: Allow items to be rotated for better space utilization
- **Weight Constraints**: Add weight limits in addition to dimensional constraints
- **Multi-objective**: Optimize for both truck count and loading efficiency
- **Real-time Interface**: Web-based interactive problem solver

### Problem 2
[To be added based on your problem 2 roadmap]

### General Improvements
- **API Development**: RESTful interface for external integration
- **Cloud Deployment**: Scalable cloud-based solving capabilities
- **Advanced Analytics**: Machine learning for solution prediction
- **Benchmarking Suite**: Comprehensive performance testing framework

## Contributing

This project demonstrates advanced algorithmic solutions and optimization techniques. Contributions are welcome for:
- Algorithm improvements
- Performance optimizations
- Additional problem domains
- Documentation enhancements
- Testing and validation

---

**Note**: This repository showcases the practical application of advanced algorithms and optimization techniques to solve real-world computational problems. Each problem demonstrates different aspects of algorithmic design, from constraint programming to machine learning analysis.
