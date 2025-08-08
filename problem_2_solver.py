from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from problem_2 import Problem_2
import numpy as np
import multiprocessing as mp

# TODO build model
# https://developers.google.com/optimization/routing/vrp

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

    n_locations = travel_matrix.shape[0]
    n_vehicles = num_of_trucks
    depot = 0 # Depot index

    # Normalize orders: ensure ints and zero demand at depot
    orders = np.array(orders, dtype=int).copy()

    manager = pywrapcp.RoutingIndexManager(n_locations, n_vehicles, depot) # Creates an index mapping for the locations
    routing = pywrapcp.RoutingModel(manager) # Creates the actual routing model

    def distance_cb(from_index, to_index):
        # A distance callback is a function that returns the distance between two nodes
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(travel_matrix[i][j])
    transit_cb_idx = routing.RegisterTransitCallback(distance_cb) # Register the distance callback
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx) # Set the arc cost evaluator for all vehicles
    
    def demand_cb(from_index):
        # Returns the demand at the given index
        i = manager.IndexToNode(from_index)
        return int(orders[i])
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb) # Register the demand callback
    max_demand = int(orders.max(initial=0))
    effective_capacity = int(max(max_orders_per_run, max_demand))
    vehicle_capacities = [effective_capacity] * n_vehicles
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # slack allowed on orders, none for others
        vehicle_capacities, # per-vehicle capacities
        True,  # Each load starts at zero
        "Capacity" # name of the dimension
    )
    
    routing.SetFixedCostOfAllVehicles(cost_per_truck) # Set the fixed cost of all trucks

    search_params = pywrapcp.DefaultRoutingSearchParameters() # TODO learn what these do
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(time_limit_s)
    search_params.log_search = False

    # Solve
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        raise Exception("No solution found")
    
    # 8) Extract routes
    # TODO : Check if this is correct
    routes = []
    total_cost = solution.ObjectiveValue()
    for v in range(n_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            # Vehicle unused (route is just Start->End)
            continue
        route_nodes = []
        load = 0
        route_cost = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            load += int(orders[node])
            nxt = solution.Value(routing.NextVar(idx))
            route_cost += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
        route_nodes.append(manager.IndexToNode(idx))  # end depot
        routes.append({
            "vehicle": v,
            "stops": route_nodes,
            "load": load,
            "route_cost": route_cost + cost_per_truck
        })
    return {"total_cost": total_cost, "routes": routes}
    





if __name__ == "__main__":
    travel_matrix, orders = Problem_2(7, 40).get_data()
    num_of_trucks, max_orders_per_run, intial_cost_per_truck = 10, 4, 10

    print(shipping_solver(travel_matrix, orders, num_of_trucks, max_orders_per_run, intial_cost_per_truck))

