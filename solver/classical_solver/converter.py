from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import sympy as sp

from problem.ising_problem import IsingProblem
from problem.qubo_problem import QuboProblem


def knapsack_to_qubo(values, weights, capacity, penalty_factor=None, **legacy_kwargs):
    """Convert a 0/1 knapsack instance into an upper-triangular QUBO matrix."""

    # Legacy French keyword aliases are preserved for notebook compatibility.
    if "valeurs" in legacy_kwargs:
        values = legacy_kwargs["valeurs"]
    if "poids" in legacy_kwargs:
        weights = legacy_kwargs["poids"]
    if "poids_max" in legacy_kwargs:
        capacity = legacy_kwargs["poids_max"]

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_items = len(values)

    if n_items != len(weights):
        raise ValueError("values and weights must have the same length.")
    if capacity <= 0:
        raise ValueError("capacity must be strictly positive.")

    num_slack_bits = int(np.floor(np.log2(capacity))) + 1
    qubo_size = n_items + num_slack_bits
    matrix = np.zeros((qubo_size, qubo_size), dtype=float)

    if penalty_factor is None:
        penalty_factor = float(np.sum(np.abs(values)) + 1.0)

    for i in range(n_items):
        matrix[i, i] -= values[i]
        matrix[i, i] += penalty_factor * (weights[i] ** 2 - 2 * capacity * weights[i])
        for j in range(i + 1, n_items):
            matrix[i, j] += 2 * penalty_factor * weights[i] * weights[j]

    for bit in range(num_slack_bits):
        slack_weight = 2**bit
        index = n_items + bit
        matrix[index, index] += penalty_factor * (slack_weight**2 - 2 * capacity * slack_weight)

        for other_bit in range(bit + 1, num_slack_bits):
            other_index = n_items + other_bit
            matrix[index, other_index] += 2 * penalty_factor * slack_weight * (2**other_bit)

    for i in range(n_items):
        for bit in range(num_slack_bits):
            matrix[i, n_items + bit] += 2 * penalty_factor * weights[i] * (2**bit)

    return matrix


def qubo_to_ising(qubo_problem: QuboProblem | np.ndarray | sparse.spmatrix):
    """Convert a QUBO problem into an equivalent Ising problem."""

    qubo = qubo_problem if isinstance(qubo_problem, QuboProblem) else QuboProblem(qubo_problem)
    upper = qubo.to_upper_triangular()
    dense_upper = upper.toarray() if sparse.issparse(upper) else np.asarray(upper, dtype=float)

    symbols = [sp.Symbol(f"s{i}") for i in range(dense_upper.shape[0])]
    expr = 0

    for i in range(dense_upper.shape[0]):
        coeff = dense_upper[i, i]
        if coeff:
            expr += (-coeff / 2.0) * symbols[i]
        expr += coeff / 2.0

        for j in range(i + 1, dense_upper.shape[1]):
            coeff = dense_upper[i, j]
            if coeff:
                expr += (coeff / 4.0) * symbols[i] * symbols[j]
                expr += (-coeff / 4.0) * symbols[i]
                expr += (-coeff / 4.0) * symbols[j]
                expr += coeff / 4.0

    return IsingProblem(expression=sp.expand(expr))


def ising_to_qubo(ising_problem: IsingProblem | sp.Expr):
    """Convert an Ising problem into an equivalent upper-triangular QUBO matrix."""

    problem = ising_problem if isinstance(ising_problem, IsingProblem) else IsingProblem(expression=ising_problem)
    expr = sp.expand(problem.to_sympy_expr())
    variables = list(problem.variables)
    index_by_symbol = {symbol: index for index, symbol in enumerate(variables)}
    matrix = np.zeros((len(variables), len(variables)), dtype=float)

    for term, coeff in expr.as_coefficients_dict().items():
        coeff = float(coeff)
        if term == 1:
            continue

        symbols = sorted(term.free_symbols, key=str)
        if len(symbols) == 1:
            i = index_by_symbol[symbols[0]]
            matrix[i, i] += -2.0 * coeff
        elif len(symbols) == 2:
            i, j = sorted((index_by_symbol[symbols[0]], index_by_symbol[symbols[1]]))
            matrix[i, i] += -2.0 * coeff
            matrix[j, j] += -2.0 * coeff
            matrix[i, j] += 4.0 * coeff
        else:
            raise ValueError("Only linear and quadratic Ising terms are supported.")

    return QuboProblem(matrix)
