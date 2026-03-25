from itertools import product
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solver.classical_solver.branch_and_bound import BranchAndBound



def exhaustive_knapsack(values, weights, capacity):
    n = len(values)
    best_value = -1
    best_selection = None

    for bits in product([0, 1], repeat=n):
        total_weight = sum(bits[i] * weights[i] for i in range(n))
        total_value = sum(bits[i] * values[i] for i in range(n))

        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_selection = list(bits)

    return best_selection, best_value


def test_branch_and_bound_matches_exhaustive():
    items = ["A", "B", "C", "D"]
    values = [20, 30, 35, 12]
    weights = [2, 5, 7, 3]
    capacity = 10

    solver = BranchAndBound(items, values, weights)
    selection, best_value = solver.solve(capacity)

    expected_selection, expected_value = exhaustive_knapsack(values, weights, capacity)

    assert best_value == expected_value
    assert selection == expected_selection


def test_branch_and_bound_zero_capacity():
    items = ["A", "B", "C"]
    values = [10, 20, 30]
    weights = [1, 2, 3]
    capacity = 0

    solver = BranchAndBound(items, values, weights)
    selection, best_value = solver.solve(capacity)

    assert selection == [0, 0, 0]
    assert best_value == 0


def test_branch_and_bound_is_deterministic():
    items = ["A", "B", "C", "D"]
    values = [24, 18, 18, 10]
    weights = [6, 3, 3, 2]
    capacity = 6

    solver1 = BranchAndBound(items, values, weights)
    solver2 = BranchAndBound(items, values, weights)

    selection1, value1 = solver1.solve(capacity)
    selection2, value2 = solver2.solve(capacity)

    assert selection1 == selection2
    assert value1 == value2


if __name__ == "__main__":
    test_branch_and_bound_matches_exhaustive()
    test_branch_and_bound_zero_capacity()
    test_branch_and_bound_is_deterministic()
    print("All tests passed.")