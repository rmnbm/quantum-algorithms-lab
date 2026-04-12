from __future__ import annotations


def _is_better(problem, candidate_value, best_value):
    if best_value is None:
        return True
    if getattr(problem, "objective_sense", "min") == "max":
        return candidate_value > best_value
    return candidate_value < best_value


def exhaustive_search(problem):
    """Evaluate every candidate in the complete search space."""

    best_solution = None
    best_value = None

    for solution in problem.generate_complete_search_space():
        try:
            value = problem.eval(solution)
        except ValueError:
            continue

        if _is_better(problem, value, best_value):
            best_solution = solution
            best_value = value

    return best_solution, best_value
