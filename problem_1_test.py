from problem_1 import Problem_1
from problem_1_solver import find_optimal_packing

for i in range(5, 16):
    data = Problem_1(i * 10).get_data()
    print(f"Test for {10 * i} Items")
    print(data)
    print(find_optimal_packing(data))
    print("\n")