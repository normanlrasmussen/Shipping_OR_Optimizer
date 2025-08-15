from problem_1 import Problem_1
from problem_1_solver import ortools_cp_solver
import pandas as pd
import os
import ast

# I want to see how fast the solver solves various problems

def run_simulation(start_solver_time, batch_size, runs, output_file, auto_batch_start=False):
    """
    # TODO : add a docstring
    """
    
    if os.path.exists(output_file):
        # The df will be of the format [orders, min_num_of_trucks] # 0 if not solved in time
        df = pd.read_csv(output_file)
        done_combinations = set(tuple(ast.literal_eval(s)) for s in df['orders'].tolist())

        done_combinations_list = list(done_combinations)
        sums = [sum(combo) for combo in done_combinations_list]
        batch_start = int(max(sums) / len(done_combinations_list[0])) - batch_size
        batch_start = max(batch_start, 0)
    else:
        done_combinations = set()
        batch_start = 0

    if auto_batch_start:
        batch_start = 0

    for _ in range(runs):
        print(f"Solving batch {batch_start} to {batch_start + batch_size}")

        # Solve in batch sizes
        combinations_to_solve = [[i, j, k, h] for i in range(batch_start, batch_start + batch_size) for j in range(batch_start, batch_start + batch_size) for k in range(batch_start, batch_start + batch_size) for h in range(batch_start, batch_start + batch_size)]
        temp_data = []

        # Get data
        while combinations_to_solve:
            if tuple(combinations_to_solve[0]) in done_combinations:
                combinations_to_solve.pop(0)
                continue
            data = combinations_to_solve.pop(0)
            try:
                sol = ortools_cp_solver(data, solve_time_limit=start_solver_time, show_plots=False)
            except Exception as e:
                sol = 0
            temp_data.append([data, sol])
        
        # Save data
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            new_df = pd.DataFrame(temp_data, columns=['orders', 'min_num_of_trucks'])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(output_file, index=False)
        else:
            df = pd.DataFrame(temp_data, columns=['orders', 'min_num_of_trucks'])
            df.to_csv(output_file, index=False)
        
        # Increment batch start
        batch_start += batch_size
    
    return 
            
                




if __name__ == "__main__":
    run_simulation(60, 6, 10, "test.csv", auto_batch_start=True)
    

    