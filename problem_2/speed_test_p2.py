from problem_2 import Problem_2
from problem_2_solver import shipping_solver
import pandas as pd
import os

# I want to test various order sizes, numbers of locations, allowered solver time
# I want to get a a simple idea of how fast and how often the solver gets the right solution

def run_simulation(seeds, orders, num_of_locations, start_solver_time, output_file):
    all_combinations = [(order_size, num_of_location, seed) for order_size in order_sizes for num_of_location in number_of_locations for seed in seeds]
    
    # Get initial data
    data_dict = {}
    for num_of_location in number_of_locations:
        for order_size in order_sizes:
            for seed in seeds:
                travel_matrix, orders = Problem_2(num_of_location, order_size, seed).get_data()
                data_dict[(num_of_location, order_size, seed)] = (travel_matrix, orders)

    print("Successfully got initial data")
    
    while all_combinations:
        print(f"Running simulation for {start_solver_time} seconds with {len(all_combinations)} combinations left")

        # Run the simulation and record which results get back a optimal solution
        can_be_recorded = []
        for index, combination in enumerate(all_combinations):
            order_size, num_of_location, seed = combination
            travel_matrix, orders = data_dict[(num_of_location, order_size, seed)]
            num_of_trucks, max_orders_per_run, intial_cost_per_truck = 5, 4, 10
            try:
                is_optimal = shipping_solver(travel_matrix, orders, num_of_trucks, max_orders_per_run, intial_cost_per_truck, time_limit_s=start_solver_time, verbose=False)
                can_be_recorded.append([index, [start_solver_time, order_size, num_of_location, seed, is_optimal]])
            except Exception as e:
                print(f"Error running simulation for {order_size} {num_of_location} {seed}: {e}")
                can_be_recorded.append([index, [start_solver_time, order_size, num_of_location, seed, False]])

        # Save the results
        if not os.path.exists(output_file):
            data_to_record = [data[1] for data in can_be_recorded]
            df = pd.DataFrame(data_to_record, columns=["start_solver_time", "order_size", "num_of_location", "seed", "is_optimal"])
            df.to_csv(output_file, index=False)
        else:
            data_to_record = [data[1] for data in can_be_recorded]
            df_to_record = pd.DataFrame(data_to_record, columns=["start_solver_time", "order_size", "num_of_location", "seed", "is_optimal"])
            df = pd.read_csv(output_file)
            df = pd.concat([df, df_to_record], ignore_index=True)
            df.to_csv(output_file, index=False)
        
        indexs_to_remove = [(data[0], data[1][-1]) for data in can_be_recorded][::-1]
        for index, is_optimal in indexs_to_remove:
            if is_optimal:
                all_combinations.pop(index)
        
        start_solver_time += 60
    
    print("Successfully ran the simulation")
    return
        



if __name__ == "__main__":
    
    seeds = [42, 24601, 1879]
    order_sizes = [10, 20, 40, 50, 70, 100]
    number_of_locations = [3, 4, 5, 6, 7]
    
    run_simulation(seeds, order_sizes, number_of_locations, 60, "p2_data.csv")
    