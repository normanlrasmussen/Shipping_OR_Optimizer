# Problem 2
# Optimize a shipping problem
# Will be given a nxn cost matrix for travel, a list of number of orders, and trucks with given constraints

import numpy as np


class Problem_2:
    def __init__(self, num_of_places:int=10, num_of_orders:int=40, seed:int = 42):
        np.random.seed(seed)
        
        # Let the starting place be at index 0
        travel_matrix = np.random.randint(10, 50, size=(num_of_places, num_of_places))
        travel_matrix = np.triu(travel_matrix)
        travel_matrix = travel_matrix + travel_matrix.T
        pertubations = np.random.randint(-5, 5, size=(num_of_places, num_of_places))
        travel_matrix += pertubations
        np.fill_diagonal(travel_matrix, 0)
        self.travel_matrix = travel_matrix
        self.orders = np.random.multinomial(num_of_orders, [1/num_of_places for _ in range(num_of_places)])
    
    def get_data(self):
        return self.travel_matrix, self.orders

if __name__ == "__main__":
    p2 = Problem_2(6, 40)