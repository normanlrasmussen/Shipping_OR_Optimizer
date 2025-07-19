from ortools.sat.python import cp_model
from problem_1 import Problem_1
import numpy as np

# Mapping for sized [(2,2),(3,3),(3,9),(4,7)]
# w - washers
# o - oven
# c - couches
# s - sofas


def find_optimal_packing(data:np.ndarray, return_spots:bool = False):
    m = cp_model.CpModel()

    
    if return_spots:
        # TODO make a method to return the positions of all the spots so we can have visualization
        raise NotImplementedError("Returng the spots not implemented")

    max_num_of_trucks = int(np.dot(np.array([4, 9, 27, 21]), data) * 1.5 / 208 + 1) # total area * 1.5 divided by truck area 8 * 26 + 1
    
    items = [f"w_{i}" for i in range(data[0])] + [f"o_{i}" for i in range(data[1])] + [f"c_{i}" for i in range(data[2])] + [f"s_{i}" for i in range(data[3])]

    # Create a boolean variable for each item to be in each truck
    item_is_in_bools = [[m.NewBoolVar(f"{item}_in_{j}") for j in range(max_num_of_trucks)] for item in items]
    for item_bools in item_is_in_bools:
        m.Add(sum(item_bools) == 1)

    # Create cordinates
    all_intervals = [[] for _ in range(max_num_of_trucks)]
    for i in range(data[0]):
        # Create the specific data points
        x = m.NewIntVar(0, 26 - 2, f"x_{items[i]}")
        y = m.NewIntVar(0, 8 - 2, f"y_{items[i]}")
        x_end = m.NewIntVar(2, 26, f"x_end_{items[i]}")
        y_end = m.NewIntVar(2, 26, f"y_end_{items[i]}")
        
        for j, item_bool in enumerate(item_is_in_bools[i]):
            # Create conditional intervals for each truck
            x_interval = m.NewOptionalIntervalVar(x, 2, x_end, item_bool, f"x_interval_for_{items[i]}_on_{j}")
            y_interval = m.NewOptionalIntervalVar(y, 2, y_end, item_bool, f"y_interval_for_{items[i]}_on_{j}")
            all_intervals[j].append([x_interval, y_interval])

    print(all_intervals)        


            
        

    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(m)

if __name__ == "__main__":
    data = Problem_1().get_data()
    print(data)
    find_optimal_packing(data)