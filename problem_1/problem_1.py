# Problem 1
# Optimize for the least amount of trucks possible given a random assortment of different sized products
# Based of a 26ft x 8ft Uhaul truck
# Lets say Washers/Dryers are 2 x 2 feet
# Stoves are 3 x 3 feet
# Couches are 3 x 9 feet
# Sofas are 4 x 7 feet

import numpy as np


class Problem_1:
    def __init__(self, num_of_items:int = 40, seed:int = 42):
        np.random.seed(seed)
        
        p = [0.4, 0.3, 0.1, 0.2] # Probabilites for Washers/Dryers, Stoves, Couches, Sofas respectivly
        self.items = np.random.multinomial(num_of_items, p) # Creates the items
    
    def get_data(self):
        return self.items

    @staticmethod
    def generate_data(num_of_items:int, num_of_instances:int, p:list = [0.4, 0.3, 0.1, 0.2]):
        # Generates data of num_of_items for num_of_instances instances with probability p
        data = []
        for _ in range(num_of_instances):
            data.append(np.random.multinomial(num_of_items, p))
        return data
    
    @staticmethod
    def get_size():
        # Get sizes
        return [(2,2),(3,3),(3,9),(4,7)]

if __name__ == "__main__":
    print(Problem_1.generate_data(100, 20))