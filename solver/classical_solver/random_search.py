from __future__ import annotations

import numpy as np

from solver.classical_solver.exhaustive_search import _is_better


def random_search(problem, iterations=1000, seed=None):
    """Sample random solutions and keep the best feasible one."""

    rng = np.random.default_rng(seed)
    best_solution = None
    best_value = None

    for _ in range(iterations):
        solution = problem.random_solution(rng)
        try:
            value = problem.eval(solution)
        except ValueError:
            continue

        if _is_better(problem, value, best_value):
            best_solution = solution
            best_value = value

    return best_solution, best_value
