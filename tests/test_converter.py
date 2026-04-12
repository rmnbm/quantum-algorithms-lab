import numpy as np

from problem.ising_problem import IsingProblem
from problem.qubo_problem import QuboProblem
from solver.classical_solver.branch_and_bound import BranchAndBound
from solver.classical_solver.converter import ising_to_qubo, knapsack_to_qubo, qubo_to_ising


def test_knapsack_to_qubo_preserves_optimum():
    items = ["A", "B", "C"]
    values = [10, 15, 20]
    weights = [1, 2, 3]
    capacity = 3

    solver = BranchAndBound(items, values, weights)
    bb_selection, bb_value = solver.solve(capacity)

    qubo_matrix = knapsack_to_qubo(values, weights, capacity, penalty_factor=50)
    qubo = QuboProblem(qubo_matrix)

    best_solution = None
    best_energy = None
    for solution in qubo.generate_complete_search_space():
        energy = qubo.eval(solution)
        if best_energy is None or energy < best_energy:
            best_solution = solution
            best_energy = energy

    qubo_selection = [best_solution[f"x{i}"] for i in range(len(items))]
    qubo_value = sum(bit * value for bit, value in zip(qubo_selection, values))
    qubo_weight = sum(bit * weight for bit, weight in zip(qubo_selection, weights))

    assert qubo_value == bb_value
    assert qubo_weight <= capacity
    assert qubo_selection == bb_selection


def test_qubo_and_ising_roundtrip_preserves_energy():
    qubo = QuboProblem(np.array([[3.0, -2.0], [0.0, 1.5]]))
    ising = qubo_to_ising(qubo)
    qubo_roundtrip = ising_to_qubo(ising)

    converted_offsets = []
    roundtrip_offsets = []
    for qubo_solution in qubo.generate_complete_search_space():
        x0 = qubo_solution["x0"]
        x1 = qubo_solution["x1"]
        ising_solution = {"s0": 1 - 2 * x0, "s1": 1 - 2 * x1}

        original = qubo.eval(qubo_solution)
        converted = ising.eval(ising_solution)
        roundtrip = qubo_roundtrip.eval(qubo_solution)

        converted_offsets.append(original - converted)
        roundtrip_offsets.append(original - roundtrip)

    assert np.allclose(converted_offsets, converted_offsets[0])
    assert np.allclose(roundtrip_offsets, roundtrip_offsets[0])


def test_ising_to_qubo_handles_symbolic_expression():
    problem = IsingProblem(string="2*s0*s1 - s0 + 3*s1")
    qubo = ising_to_qubo(problem)

    assert qubo.matrix.shape == (2, 2)
    assert np.isclose(qubo.matrix[0, 1], 8.0)
