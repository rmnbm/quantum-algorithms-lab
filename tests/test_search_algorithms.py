import numpy as np

from problem.knapsack_problem import KnapsackProblem
from problem.qubo_problem import QuboProblem
from solver.classical_solver.exhaustive_search import exhaustive_search
from solver.classical_solver.local_search import local_search
from solver.classical_solver.random_search import random_search
from solver.classical_solver.simulated_annealing import SimulatedAnnealing


def test_exhaustive_search_solves_small_knapsack():
    problem = KnapsackProblem(values=[6, 10, 12], weights=[1, 2, 3], capacity=5)
    solution, value = exhaustive_search(problem)

    assert solution == {"x0": 0, "x1": 1, "x2": 1}
    assert value == 22.0


def test_random_search_returns_feasible_knapsack_solution():
    problem = KnapsackProblem(values=[6, 10, 12], weights=[1, 2, 3], capacity=3)
    solution, value = random_search(problem, iterations=200, seed=0)

    assert problem.is_feasible(solution)
    assert value is not None


def test_local_search_improves_simple_qubo():
    problem = QuboProblem(np.array([[-1.0, 0.0], [0.0, -2.0]]))
    initial_solution = {"x0": 0, "x1": 0}
    solution, value = local_search(problem, iterations=10, seed=0, initial_solution=initial_solution)

    assert solution == {"x0": 1, "x1": 1}
    assert np.isclose(value, -3.0)


def test_simulated_annealing_returns_binary_solution():
    problem = QuboProblem(np.array([[-1.0, 0.5], [0.0, -2.0]]))
    annealer = SimulatedAnnealing(initial_temp=5.0, decreasing_factor=0.8, steps_per_temp=25, min_temp=0.1)
    solution, value = annealer.solve(problem, seed=0)

    assert set(solution.keys()) == {"x0", "x1"}
    assert all(bit in {0, 1} for bit in solution.values())
    assert isinstance(value, float)
