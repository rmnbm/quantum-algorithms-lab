from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

import numpy as np
import scipy.sparse as sparse
import sympy as sp


class QuboProblem:
    """Represent a quadratic unconstrained binary optimization problem."""

    objective_sense = "min"

    def __init__(self, matrix):
        if not (sparse.issparse(matrix) or isinstance(matrix, np.ndarray)):
            matrix = np.array(matrix, dtype=float)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The QUBO matrix must be square.")

        if not (self._is_symmetric(matrix) or self._is_triangular_upper(matrix)):
            raise ValueError("The matrix must be either symmetric or upper triangular.")

        self.matrix = matrix.astype(float) if isinstance(matrix, np.ndarray) else matrix.astype(float)
        self._sympy_expr = None

    @property
    def variables(self):
        return tuple(sp.Symbol(f"x{i}") for i in range(self.matrix.shape[0]))

    def _is_symmetric(self, matrix):
        if sparse.issparse(matrix):
            return (matrix != matrix.T).nnz == 0
        return np.allclose(matrix, matrix.T)

    def _is_triangular_upper(self, matrix):
        if sparse.issparse(matrix):
            return sparse.tril(matrix, k=-1).nnz == 0
        return np.allclose(matrix, np.triu(matrix))

    def to_upper_triangular(self):
        if sparse.issparse(self.matrix):
            diagonal = sparse.diags(self.matrix.diagonal())
            strict_upper = sparse.triu(self.matrix, k=1)
            strict_lower = sparse.tril(self.matrix, k=-1).T
            return (diagonal + strict_upper + strict_lower).tocsr()

        diagonal = np.diag(np.diag(self.matrix))
        strict_upper = np.triu(self.matrix, k=1)
        strict_lower = np.tril(self.matrix, k=-1).T
        return diagonal + strict_upper + strict_lower

    def to_sympy_expr(self):
        if self._sympy_expr is not None:
            return self._sympy_expr

        matrix = self.to_upper_triangular()
        symbols = self.variables
        terms = []

        if sparse.issparse(matrix):
            rows, cols, values = sparse.find(matrix)
            for i, j, value in zip(rows, cols, values):
                if value:
                    terms.append(value * symbols[i] * symbols[j])
        else:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] != 0:
                        terms.append(matrix[i, j] * symbols[i] * symbols[j])

        self._sympy_expr = sp.Add(*terms) if terms else sp.Integer(0)
        return self._sympy_expr

    def vector_from_solution(self, solution: Mapping):
        vector = np.zeros(self.matrix.shape[0], dtype=float)
        for index in range(len(vector)):
            vector[index] = int(solution.get(f"x{index}", 0))
        return vector

    def evaluate(self, solution: Mapping):
        vector = self.vector_from_solution(solution)
        cost = vector.T @ self.matrix @ vector
        return float(np.asarray(cost).item())

    def eval(self, solution: Mapping):
        return self.evaluate(solution)

    def random_solution(self, rng):
        return {f"x{i}": int(rng.integers(0, 2)) for i in range(self.matrix.shape[0])}

    def generate_neighbor_sol(self, solution: Mapping, rng):
        if self.matrix.shape[0] == 0:
            return {}

        neighbor = {f"x{i}": int(solution.get(f"x{i}", 0)) for i in range(self.matrix.shape[0])}
        index = int(rng.integers(0, self.matrix.shape[0]))
        key = f"x{index}"
        neighbor[key] = 1 - neighbor[key]
        return neighbor

    def generate_complete_search_space(self) -> Iterable[dict[str, int]]:
        size = self.matrix.shape[0]
        for assignment in product([0, 1], repeat=size):
            yield {f"x{i}": bit for i, bit in enumerate(assignment)}
