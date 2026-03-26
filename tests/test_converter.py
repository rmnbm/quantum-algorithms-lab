import os
import sys
import numpy as np
from itertools import product

# Fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver.classical_solver.branch_and_bound import BranchAndBound
from solver.classical_solver.converter import knapsack_to_qubo

def solve_qubo_exhaustively(Q):
    size = Q.shape[0]
    best_cost = float('inf')
    best_state = None
    
    # On teste les 2^size combinaisons possibles
    for x in product([0, 1], repeat=size):
        x_vec = np.array(x)
        current_cost = x_vec.T @ Q @ x_vec
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_state = x_vec
            
    return best_state, best_cost

def test_qubo_conversion_equivalence():
    # Instance simple de Knapsack
    items = ["A", "B", "C"]
    values = [10, 15, 20]
    weights = [1, 2, 3]
    capacity = 3
    
    # Solution via Branch & Bound 
    bb = BranchAndBound(items, values, weights)
    bb_selection, bb_value = bb.solve(capacity)
    print(f"B&B Optimal Value: {bb_value}")

    # 3. Conversion en QUBO 
    Q = knapsack_to_qubo(values, weights, capacity, penalty_factor=50)
    qubo_state, qubo_energy = solve_qubo_exhaustively(Q)
    
    # On ne garde que les n_items premiers bits 
    qubo_selection = list(qubo_state[:len(items)])
    
    # Calcul de la valeur réelle des objets sélectionnés par le QUBO
    qubo_real_value = sum(qubo_selection[i] * values[i] for i in range(len(items)))
    qubo_real_weight = sum(qubo_selection[i] * weights[i] for i in range(len(items)))

    print(f"QUBO Selection: {qubo_selection} (Value: {qubo_real_value}, Weight: {qubo_real_weight})")

    # 5. Vérifications
    # La sélection doit être identique et le poids respecté
    assert qubo_real_value == bb_value, "Le QUBO n'a pas trouvé la même valeur optimale que le B&B"
    assert qubo_real_weight <= capacity, "La solution QUBO dépasse la capacité autorisée"
    print("Test de conversion réussi : L'optimum est préservé.")

if __name__ == "__main__":
    test_qubo_conversion_equivalence()