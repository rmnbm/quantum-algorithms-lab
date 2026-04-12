from __future__ import annotations

import numpy as np

from solver.classical_solver.exhaustive_search import _is_better


def local_search(problem, iterations=1000, seed=None, initial_solution=None):
    """Hill-climbing style local search with random neighbor proposals."""

    rng = np.random.default_rng(seed)
    current_solution = initial_solution if initial_solution is not None else problem.random_solution(rng)
    current_value = problem.eval(current_solution)

    best_solution = current_solution
    best_value = current_value

    for _ in range(iterations):
        neighbor = problem.generate_neighbor_sol(current_solution, rng)
        try:
            neighbor_value = problem.eval(neighbor)
        except ValueError:
            continue

        if _is_better(problem, neighbor_value, current_value):
            current_solution = neighbor
            current_value = neighbor_value

        if _is_better(problem, current_value, best_value):
            best_solution = current_solution
            best_value = current_value

    return best_solution, best_value
