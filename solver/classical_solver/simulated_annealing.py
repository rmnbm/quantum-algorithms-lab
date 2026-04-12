from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse

from problem.qubo_problem import QuboProblem


@dataclass
class SimulatedAnnealing:
    """Simulated annealing solver for QUBO problems."""

    initial_temp: float = 1000.0
    decreasing_factor: float = 0.95
    steps_per_temp: int = 100
    min_temp: float = 0.01

    def _cool(self, temperature):
        return temperature * self.decreasing_factor

    def _accept(self, delta_energy, temperature, rng):
        if delta_energy < 0:
            return True
        probability = math.exp(-delta_energy / temperature)
        return rng.random() < probability

    def _to_dict(self, vector):
        return {f"x{i}": int(value) for i, value in enumerate(vector)}

    def solve_matrix_product(self, qubo_problem: QuboProblem, seed=None):
        rng = np.random.default_rng(seed)
        size = qubo_problem.matrix.shape[0]
        current = rng.integers(0, 2, size=size)
        current_cost = qubo_problem.eval(self._to_dict(current))

        best = current.copy()
        best_cost = current_cost
        temperature = self.initial_temp

        while temperature > self.min_temp:
            for _ in range(self.steps_per_temp):
                flip_index = int(rng.integers(0, size))
                neighbor = current.copy()
                neighbor[flip_index] = 1 - neighbor[flip_index]
                neighbor_cost = qubo_problem.eval(self._to_dict(neighbor))
                delta_energy = neighbor_cost - current_cost

                if self._accept(delta_energy, temperature, rng):
                    current = neighbor
                    current_cost = neighbor_cost
                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost

            temperature = self._cool(temperature)

        return self._to_dict(best), float(best_cost)

    def _calculate_delta_energy(self, matrix, current, flip_index):
        bit_value = current[flip_index]
        delta = matrix[flip_index, flip_index]

        if sparse.issparse(matrix):
            row = matrix.getrow(flip_index)
            for index, value in zip(row.indices, row.data):
                if index != flip_index and current[index] == 1:
                    delta += value

            col = matrix.getcol(flip_index)
            for index, value in zip(col.indices, col.data):
                if index != flip_index and current[index] == 1:
                    delta += value
        else:
            for index, value in enumerate(matrix[flip_index, :]):
                if index != flip_index and current[index] == 1:
                    delta += value
            for index, value in enumerate(matrix[:, flip_index]):
                if index != flip_index and current[index] == 1:
                    delta += value

        return float((1 - 2 * bit_value) * delta)

    def solve_efficient(self, qubo_problem: QuboProblem, seed=None):
        rng = np.random.default_rng(seed)
        size = qubo_problem.matrix.shape[0]
        current = rng.integers(0, 2, size=size)
        current_cost = qubo_problem.eval(self._to_dict(current))

        best = current.copy()
        best_cost = current_cost
        temperature = self.initial_temp

        while temperature > self.min_temp:
            for _ in range(self.steps_per_temp):
                flip_index = int(rng.integers(0, size))
                delta_energy = self._calculate_delta_energy(qubo_problem.matrix, current, flip_index)

                if self._accept(delta_energy, temperature, rng):
                    current[flip_index] = 1 - current[flip_index]
                    current_cost += delta_energy
                    if current_cost < best_cost:
                        best = current.copy()
                        best_cost = current_cost

            temperature = self._cool(temperature)

        return self._to_dict(best), float(best_cost)

    def solve(self, qubo_problem: QuboProblem, seed=None, efficient=True):
        if efficient:
            return self.solve_efficient(qubo_problem, seed=seed)
        return self.solve_matrix_product(qubo_problem, seed=seed)
