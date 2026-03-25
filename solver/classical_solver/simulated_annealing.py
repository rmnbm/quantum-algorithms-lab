# TODO 2: Code a simulated annealing method to solve the QUBO problem. Assume you have the
# QUBO matrix as input (which can be evaluated as xTQx). The use of the method should provide at
# least the following parameters:
# • Cooling schedule law and decreasing_factor (we will assume at first an exponential decrease
# of the temperature, but in practice you could use any law, for example linear or geometric).
# The decreasing_factor relates to the constant used in the slides of the lesson.
# • Number of fixed steps per temperature step.
# • Stopping criteria (temperature threshold).


import numpy as np
import random
import math
import time
import scipy.sparse as sp
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

from problem.qubo_problem import QuboProblem

class SimulatedAnnealing:
    def __init__(self, initial_temp=1000.0, decreasing_factor=0.95, steps_per_temp=100, min_temp=0.01):
        self.initial_temp = initial_temp
        self.decreasing_factor = decreasing_factor 
        self.steps_per_temp = steps_per_temp       
        self.min_temp = min_temp                   

    def solve_matrix_product(self, qubo_problem):
        n = qubo_problem.matrix.shape[0]
        def to_dict(x_array):
            return {f'x{i}': x_array[i] for i in range(n)}
        
        #random vector
        current_x = np.random.randint(2, size=n)
        current_cost = qubo_problem.eval(to_dict(current_x))
        
        #best solution yet
        best_x = np.copy(current_x)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        #Annealing Loop
        while temp > self.min_temp:
            
            #fixed number of exploration steps at this temperature
            for _ in range(self.steps_per_temp):
                
                #flip a random bit to create a "neighbor" solution
                flip_index = random.randint(0, n - 1)
                neighbor_x = np.copy(current_x)
                neighbor_x[flip_index] = 1 - neighbor_x[flip_index] 
                
                #compute neighbor's cost
                neighbor_cost = qubo_problem.eval(to_dict(neighbor_x))
                delta_e = neighbor_cost - current_cost
                
                #Acceptance criteria
                if delta_e < 0:
                    
                    accept = True
                else:
                    probability = math.exp(-delta_e / temp)
                    accept = random.random() < probability
                    
                if accept:
                    current_x = np.copy(neighbor_x)
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = np.copy(current_x)
            
            #Cooling
            temp *= self.decreasing_factor
            
        return to_dict(best_x), best_cost

    def solve_efficient(self, qubo_problem):
        n = qubo_problem.matrix.shape[0]
        def to_dict(x_array):
            return {f'x{i}': x_array[i] for i in range(n)}
        
        #random vector
        current_x = np.random.randint(2, size=n)
        current_cost = qubo_problem.eval(to_dict(current_x))
        
        #best solution yet
        best_x = np.copy(current_x)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        #Annealing Loop
        while temp > self.min_temp:
            
            #fixed number of exploration steps at this temperature
            for _ in range(self.steps_per_temp):
                
                #flip a random bit to create a "neighbor" solution
                flip_index = random.randint(0, n - 1)
                delta_e = self._calculate_delta_e(qubo_problem.matrix, current_x, flip_index)
                
                
                #Acceptance criteria
                if delta_e < 0:
                    
                    accept = True
                else:
                    probability = math.exp(-delta_e / temp)
                    accept = random.random() < probability
                    
                if accept:
                    current_x[flip_index] = 1 - current_x[flip_index]
                    current_cost += delta_e
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = np.copy(current_x)
            
            #Cooling
            temp *= self.decreasing_factor
            
        return to_dict(best_x), best_cost



    def _calculate_delta_e(self, qubo_matrix, current_x, flip_index):
        
        x_k = current_x[flip_index]
        

        delta = qubo_matrix[flip_index, flip_index]
        
        row = qubo_matrix.getrow(flip_index)
        indices = row.indices
        data = row.data
        for idx, val in zip(indices, data):
            if idx != flip_index and current_x[idx] == 1:
                delta += val
                
        col = qubo_matrix.getcol(flip_index)
        indices = col.indices
        data = col.data
        for idx, val in zip(indices, data):
            if idx != flip_index and current_x[idx] == 1:
                delta += val
        return (1 - 2 * x_k) * delta


if __name__ == "__main__":
    N = 1000
    print(f"Generating a {N}*{N} matrix...")

    random_matrix = sp.random(N, N,density=0.1, format='csr')
    Q_data = sp.triu(random_matrix, format='csr')
    qubo = QuboProblem(Q_data)

    sa = SimulatedAnnealing()

    print("\nRunning the slow method (solve_matrix_product)")

    start_time_old = time.time()
    best_sol_old, best_cost_old = sa.solve_matrix_product(qubo)
    end_time_old = time.time()
    old_duration = end_time_old - start_time_old
    print(f"The old program finished in {old_duration} seconds. The cost needed is {best_cost_old}")


    print("\nRunning the fast method")
    start_time_fast = time.time()
    best_sol_fast, best_cost_fast = sa.solve_efficient(qubo)
    end_time_fast = time.time()
    fast_duration = end_time_fast - start_time_fast
    print(f"The fast program finished in {fast_duration} seconds. The cost needed is {best_cost_fast}")
