from ortools.sat.python import cp_model
from problem_2 import Problem_2
import numpy as np
import os

def shipping_solver(travel_matrix:np.ndarray, # n locations x n locations cost matrix
                    orders:np.ndarray, # n locations x 1 array of orders
                    num_of_trucks:int, # number of trucks
                    max_orders_per_run:int, # max orders per run
                    cost_per_truck:int, # cost per truck
                    time_limit_s:int = 30 # time limit in seconds
                    ):
    """
    # TODO : Finish the docstring
    """

    m = cp_model.CpModel()
    N = len(travel_matrix) 
    num_of_trucks_HORIZON = int(np.ceil(np.sum(orders))/max_orders_per_run) + 1 # This gives us a HORIZON for the maximun number of runs that we can have


    # Generates all the bools saying whether a path is taken in a run
    takes_path_bool = [
        [
            [
                m.NewBoolVar(f"route_{n}_from_{i}_to_{j}") if i != j else 0
                for j in range(len(travel_matrix))
            ] 
            for i in range(len(travel_matrix))
        ]
        for n in range(num_of_trucks_HORIZON)
    ]
    
    # Enforce that the amount coming to a place has to be equal to the amount going out
    for n in range(num_of_trucks_HORIZON):
        for i in range(len(travel_matrix)):
            # The of the amount coming to the depot has to be equal to the sum coming back
            m.Add(sum([takes_path_bool[n][i][j] for j in range(len(travel_matrix))]) 
            == sum([takes_path_bool[n][j][i] for j in range(len(travel_matrix))])
            )

    # Ensure continuous routes - no disconnected circuits
    # For each truck run, if it visits any location, it must form a single connected component
    visited_all = []  # Collect all visited arrays
    for n in range(num_of_trucks_HORIZON):
        U = max_orders_per_run  # safe upper bound

        # 1) visited flag: turns on iff any arc touches node i (customers only)
        visited = [
            m.NewBoolVar(f"run{n}_vis_{i}") for i in range(N)
        ]
        visited_all.append(visited)  # Store for later use
        m.Add(visited[0] == 0)  # depot doesn't "consume" flow

        def deg_out(i): return sum(takes_path_bool[n][i][j] for j in range(N) if j != i)
        def deg_in(i):  return sum(takes_path_bool[n][j][i] for j in range(N) if j != i)

        BIGM = N
        for i in range(1, N):
            touch = deg_out(i) + deg_in(i)
            m.Add(touch >= visited[i])
            m.Add(touch <= BIGM * visited[i])

        # 2) flow vars on arcs, only along chosen arcs
        flow = [
            [m.NewIntVar(0, U, f"f_run{n}_{i}_{j}") if i != j else None
             for j in range(N)]
            for i in range(N)
        ]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                m.Add(flow[i][j] <= U * takes_path_bool[n][i][j])

        # 3) conservation: depot supplies 1 unit to each visited customer; customers consume 1
        # depot balance
        m.Add(
            sum(flow[0][j] for j in range(1, N)) -
            sum(flow[j][0] for j in range(1, N))
            == sum(visited[i] for i in range(1, N))
        )
        # customer balances
        for i in range(1, N):
            m.Add(
                sum(flow[j][i] for j in range(N) if j != i) -
                sum(flow[i][j] for j in range(N) if j != i)
                == visited[i]
            )

    # Enforce that every truck has a maximum number of orders
    orders_in_truck = []
    truck_has_inventory = []
    for n in range(num_of_trucks_HORIZON):
        orders_in_truck.append(m.NewIntVar(0, max_orders_per_run, f"orders_in_truck_{n}"))
        truck_has_inventory.append(m.NewBoolVar(f"truck_{n}_has_inventory"))
        m.Add(orders_in_truck[n] >= 1).OnlyEnforceIf(truck_has_inventory[n])
        m.Add(orders_in_truck[n] == 0).OnlyEnforceIf(truck_has_inventory[n].Not())
        
        if n != 0: 
            # Enforce filling up earlier trucks first
            m.AddImplication(truck_has_inventory[n], truck_has_inventory[n-1])

    specific_inventories_trucks = [[] for _ in range(num_of_trucks_HORIZON)]
    deliveries_for_orders = [[] for _ in range(len(orders))]
    for i in range(len(orders)):
        for n in range(num_of_trucks_HORIZON):
            specific_order = m.NewIntVar(0, max_orders_per_run, f"delivery_for_order_{i}_in_truck_{n}")
            specific_inventories_trucks[n].append(specific_order)
            deliveries_for_orders[i].append(specific_order)
        # Enforces that the sum of the deliveries for an order is equal to the order
        m.Add(sum(deliveries_for_orders[i]) == orders[i])
    for n in range(num_of_trucks_HORIZON):
        # Enforces that the sum of the specific inventories for a truck is less than or equal to the orders in the truck
        m.Add(sum(specific_inventories_trucks[n]) <= orders_in_truck[n])

    
    # Add in that when a truck has an order it must go to that place
    # If a truck has inventory at a location, it must visit that location
    for n in range(num_of_trucks_HORIZON):
        for i in range(1, N):  # skip depot
            # Simple gate: if delivering orders, must visit
            U = min(max_orders_per_run, int(orders[i]))  # Tighter bound
            m.Add(specific_inventories_trucks[n][i] <= U * visited_all[n][i])

    # Fix name collision and improve cost calculation
    run_cost = [m.NewIntVar(0, 10**12, f"run_cost_{n}") for n in range(num_of_trucks_HORIZON)]

    # Add in all the costs
    for n in range(num_of_trucks_HORIZON):
        m.Add(
            run_cost[n] == sum([
                takes_path_bool[n][i][j] * travel_matrix[i][j]
                for i in range(N)
                for j in range(N)
                if i != j  # Guard against i == j
            ]) + cost_per_truck * truck_has_inventory[n]
        )

    total_cost = m.NewIntVar(0, 10**12, "total_cost")
    m.Add(total_cost == sum(run_cost))

    # Add in the minimize function
    m.Minimize(total_cost)

    # Solver with optimized parameters
    solver = cp_model.CpSolver()
    
    # Parallel search - use most cores but leave one free for system
    solver.parameters.num_search_workers = max(1, os.cpu_count() - 1)
    
    # Presolve and preprocessing
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 2  # Aggressive linearization for better bounds
    solver.parameters.symmetry_level = 2  # Maximum symmetry breaking
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.log_search_progress = True  # Useful for monitoring
    solver.parameters.interleave_search = True  # Better for parallel search
    
    status = solver.Solve(m)

    if status == cp_model.OPTIMAL:
        print("Optimal solution found.")
    elif status == cp_model.FEASIBLE:
        print("Feasible solution found (but not proven optimal).")
    elif status == cp_model.INFEASIBLE:
        raise RuntimeError("No solution exists.")
    print("")

    # Print out the solution

    

if __name__ == "__main__":
    travel_matrix, orders = Problem_2(4, 20).get_data()
    num_of_trucks, max_orders_per_run, intial_cost_per_truck = 5, 4, 10

    shipping_solver(travel_matrix, orders, num_of_trucks, max_orders_per_run, intial_cost_per_truck, time_limit_s=60)

