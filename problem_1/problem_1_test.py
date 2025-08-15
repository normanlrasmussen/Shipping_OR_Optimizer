from problem_1 import Problem_1
from problem_1_solver import ortools_cp_solver

for i in range(5, 16):
    data = Problem_1(i * 10).get_data()
    print(f"Test for {10 * i} Items")
    print(data)
    print(ortools_cp_solver(data))
    print("\n")