from ortools.sat.python import cp_model
from problem_1 import Problem_1
import numpy as np
import os

# Mapping for sized [(2,2),(3,3),(3,9),(4,7)]
# w - washers
# o - oven
# c - couches
# s - sofas

# TODO Add a plotting function, and a function that returns all of the data points

def ortools_cp_solver(data: np.ndarray, solve_time_limit: int = 60, return_spots: bool = False):
    """Original constraint programming solver"""
    m = cp_model.CpModel()
    
    if return_spots:
        raise NotImplementedError("Returning the spots not implemented")

    min_num_of_trucks = int(np.dot(np.array([4, 9, 27, 28]), data) / 208) # total area divided by truck area 8 * 26
    max_num_of_trucks = int(np.dot(np.array([4, 9, 27, 28]), data) * 1.5 / 208 + 1) # total area * 1.5 divided by truck area 8 * 26 + 1
    
    items = [f"w_{i}" for i in range(data[0])] + [f"o_{i}" for i in range(data[1])] + [f"c_{i}" for i in range(data[2])] + [f"s_{i}" for i in range(data[3])]

    # Create a boolean variable for each item to be in each truck
    item_is_in_bools = [[m.NewBoolVar(f"{item}_in_{j}") for j in range(max_num_of_trucks)] for item in items]
    for item_bools in item_is_in_bools:
        m.Add(sum(item_bools) == 1)

    # Create intervals holders
    all_intervals = [[] for _ in range(max_num_of_trucks)]

    # Create all intervals for Washers
    for i in range(data[0]):
        # Create the specific data points
        x = m.NewIntVar(0, 26 - 2, f"x_{items[i]}")
        y = m.NewIntVar(0, 8 - 2, f"y_{items[i]}")
        x_end = m.NewIntVar(2, 26, f"x_end_{items[i]}")
        y_end = m.NewIntVar(2, 8, f"y_end_{items[i]}")
        
        # Constrain end positions
        m.Add(x_end == x + 2)
        m.Add(y_end == y + 2)
        
        for j, item_bool in enumerate(item_is_in_bools[i]):
            # Create conditional intervals for each truck
            x_interval = m.NewOptionalIntervalVar(x, 2, x_end, item_bool, f"x_interval_for_{items[i]}_on_{j}")
            y_interval = m.NewOptionalIntervalVar(y, 2, y_end, item_bool, f"y_interval_for_{items[i]}_on_{j}")
            all_intervals[j].append((x_interval, y_interval))
    
    # Create all invervals for ovens
    c = data[0] # How many data points need to be skipped
    for i in range(data[1]):
        # Create the specific data points
        x = m.NewIntVar(0, 26 - 3, f"x_{items[i + c]}")
        y = m.NewIntVar(0, 8 - 3, f"y_{items[i + c]}")
        x_end = m.NewIntVar(3, 26, f"x_end_{items[i + c]}")
        y_end = m.NewIntVar(3, 8, f"y_end_{items[i + c]}")
        
        # Constrain end positions
        m.Add(x_end == x + 3)
        m.Add(y_end == y + 3)
        
        for j, item_bool in enumerate(item_is_in_bools[i + c]):
            # Create conditional intervals for each truck
            x_interval = m.NewOptionalIntervalVar(x, 3, x_end, item_bool, f"x_interval_for_{items[i + c]}_on_{j}")
            y_interval = m.NewOptionalIntervalVar(y, 3, y_end, item_bool, f"y_interval_for_{items[i + c]}_on_{j}")
            all_intervals[j].append((x_interval, y_interval))
    
    # Create all invervals for couches
    c = data[0] + data[1] # How many data points need to be skipped
    for i in range(data[2]):
        # Create the specific data points
        x = m.NewIntVar(0, 26 - 9, f"x_{items[i + c]}")
        y = m.NewIntVar(0, 8 - 3, f"y_{items[i + c]}")
        x_end = m.NewIntVar(9, 26, f"x_end_{items[i + c]}")
        y_end = m.NewIntVar(3, 8, f"y_end_{items[i + c]}")
        
        # Constrain end positions
        m.Add(x_end == x + 9)
        m.Add(y_end == y + 3)
        
        for j, item_bool in enumerate(item_is_in_bools[i + c]):
            # Create conditional intervals for each truck
            x_interval = m.NewOptionalIntervalVar(x, 9, x_end, item_bool, f"x_interval_for_{items[i + c]}_on_{j}")
            y_interval = m.NewOptionalIntervalVar(y, 3, y_end, item_bool, f"y_interval_for_{items[i + c]}_on_{j}")
            all_intervals[j].append((x_interval, y_interval))
    
    # Create all invervals for sofas
    # TODO Add rotation
    c = data[0] + data[1] + data[2] # How many data points need to be skipped
    for i in range(data[3]):
        # Create the specific data points for non-rotated (4x7)
        x = m.NewIntVar(0, 26 - 7, f"x_{items[i + c]}")
        y = m.NewIntVar(0, 8 - 4, f"y_{items[i + c]}")
        x_end = m.NewIntVar(7, 26, f"x_end_{items[i + c]}")
        y_end = m.NewIntVar(4, 8, f"y_end_{items[i + c]}")

        # Constrain end positions
        m.Add(x_end == x + 7)
        m.Add(y_end == y + 4)

        for j, item_bool in enumerate(item_is_in_bools[i + c]):
            # Create conditional intervals for each truck
            x_interval = m.NewOptionalIntervalVar(x, 7, x_end, item_bool, f"x_interval_for_{items[i + c]}_on_{j}")
            y_interval = m.NewOptionalIntervalVar(y, 4, y_end, item_bool, f"y_interval_for_{items[i + c]}_on_{j}")
            all_intervals[j].append((x_interval, y_interval))
    
    for intervals in all_intervals:
        x_intervals = [interval[0] for interval in intervals]
        y_intervals = [interval[1] for interval in intervals]
        m.AddNoOverlap2D(x_intervals, y_intervals)
    
    is_item_in_truck = [m.NewBoolVar(f"items_in_{i}") for i in range(max_num_of_trucks)]

    for i, truck_bool in enumerate(is_item_in_truck):
        is_on_bools = [item[i] for item in item_is_in_bools]
        m.AddBoolOr(is_on_bools).OnlyEnforceIf(truck_bool) # Atleast one of the conditions must be true
        m.AddBoolAnd([bool.Not() for bool in is_on_bools]).OnlyEnforceIf(truck_bool.Not()) # Everybool must be false
    
    # Define the objective
    num_of_trucks = m.NewIntVar(min_num_of_trucks, max_num_of_trucks, "num_of_trucks")
    m.Add(num_of_trucks == sum(is_item_in_truck))
    m.Minimize(num_of_trucks)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1 # Use all CPU cores (unless debugging)
    solver.parameters.cp_model_presolve = True  # Already on by default
    solver.parameters.linearization_level = 2  # Enables tighter internal constraints
    solver.parameters.symmetry_level = 2  # Breaks more symmetries (can help in packing)
    solver.parameters.max_time_in_seconds = solve_time_limit
    
    status = solver.Solve(m)

    if status == cp_model.OPTIMAL:
        pass
    elif status == cp_model.FEASIBLE:
        print("Feasible solution found (but not proven optimal).")
    elif status == cp_model.INFEASIBLE:
        raise RuntimeError("No solution exists.")
    else:
        raise RuntimeError(f"Solver ended with unknown status: {status}")

    best_num_of_trucks = solver.Value(num_of_trucks)

    return best_num_of_trucks

if __name__ == "__main__":
    data = Problem_1(40).get_data()
    print(data)
    sol = ortools_cp_solver(data)
    print(sol)