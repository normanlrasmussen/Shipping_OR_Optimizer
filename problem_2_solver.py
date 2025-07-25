from ortools.sat.python import cp_model
from problem_2 import Problem_2
import numpy as np
import os

# TODO build model

def shipping_cp_solver(travel_matrix:np.ndarray, orders:np.ndarray, num_of_trucks:int, max_orders_per_run:int, intial_cost_per_truck:int):
    """
    Takes in the parameters of a travel matrix with costs, orders per place, number of trucks you have, 
    how many orders they can take at a time, and how much it costs to send out each truck (to not promote sending out all the trucks at once.)
    """
    m = cp_model.CpModel()

    

if __name__ == "__main__":
    travel_matrix, orders = Problem_2(7, 40).get_data()
    num_of_trucks, max_orders_per_run, intial_cost_per_truck = 10, 4, 10

