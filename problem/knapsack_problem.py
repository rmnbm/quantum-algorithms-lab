from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping, Sequence


class KnapsackProblem:
    """Represent a 0/1 knapsack maximization problem."""

    objective_sense = "max"

    def __init__(
        self,
        values: Sequence[float],
        weights: Sequence[float],
        capacity: float,
        items: Sequence[str] | None = None,
    ):
        if len(values) != len(weights):
            raise ValueError("values and weights must have the same length.")

        self.values = list(values)
        self.weights = list(weights)
        self.capacity = capacity
        self.items = list(items) if items is not None else [f"item_{i}" for i in range(len(values))]

    def is_feasible(self, solution: Mapping):
        return self.total_weight(solution) <= self.capacity

    def total_weight(self, solution: Mapping):
        return sum(int(solution.get(f"x{i}", 0)) * self.weights[i] for i in range(len(self.values)))

    def evaluate(self, solution: Mapping):
        if not self.is_feasible(solution):
            raise ValueError("The provided knapsack solution is infeasible.")
        return float(sum(int(solution.get(f"x{i}", 0)) * self.values[i] for i in range(len(self.values))))

    def eval(self, solution: Mapping):
        return self.evaluate(solution)

    def random_solution(self, rng):
        return {f"x{i}": int(rng.integers(0, 2)) for i in range(len(self.values))}

    def generate_neighbor_sol(self, solution: Mapping, rng):
        neighbor = {f"x{i}": int(solution.get(f"x{i}", 0)) for i in range(len(self.values))}
        if not neighbor:
            return neighbor
        index = int(rng.integers(0, len(self.values)))
        key = f"x{index}"
        neighbor[key] = 1 - neighbor[key]
        return neighbor

    def generate_complete_search_space(self) -> Iterable[dict[str, int]]:
        for assignment in product([0, 1], repeat=len(self.values)):
            yield {f"x{i}": bit for i, bit in enumerate(assignment)}
