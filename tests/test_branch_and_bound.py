from itertools import product

from solver.classical_solver.branch_and_bound import BranchAndBound


def exhaustive_knapsack(values, weights, capacity):
    best_value = -1
    best_selection = None

    for bits in product([0, 1], repeat=len(values)):
        total_weight = sum(bits[i] * weights[i] for i in range(len(values)))
        total_value = sum(bits[i] * values[i] for i in range(len(values)))
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
    solver = BranchAndBound(["A", "B", "C"], [10, 20, 30], [1, 2, 3])
    selection, best_value = solver.solve(0)

    assert selection == [0, 0, 0]
    assert best_value == 0
