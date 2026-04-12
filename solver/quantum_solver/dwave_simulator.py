from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.linalg import eigh


def _coerce_expression(problem):
    if hasattr(problem, "to_sympy_expr"):
        return sp.expand(problem.to_sympy_expr())
    return sp.expand(sp.sympify(problem))


class DwaveSimulator:
    """Toy simulator for adiabatic evolution on an Ising Hamiltonian."""

    def __init__(self):
        self.annealing_schedule = {
            "A": np.linspace(1.0, 0.0, 101),
            "B": np.linspace(0.0, 1.0, 101),
        }
        self.A = self.annealing_schedule["A"]
        self.B = self.annealing_schedule["B"]
        self.sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.identity = np.eye(2)

    def sigma_i(self, qubit_count, target_qubit, op_type="z"):
        base_op = self.sigma_z if op_type == "z" else self.sigma_x
        result = base_op if target_qubit == 0 else self.identity
        for qubit in range(1, qubit_count):
            current = base_op if qubit == target_qubit else self.identity
            result = np.kron(result, current)
        return result

    def build_Hfinal(self, ising_problem, qubit_count=None):
        expression = _coerce_expression(ising_problem)
        variables = sorted(expression.free_symbols, key=str)
        qubit_count = len(variables) if qubit_count is None else qubit_count
        var_to_index = {var: index for index, var in enumerate(variables)}

        h_final = np.zeros((2**qubit_count, 2**qubit_count))
        for term, coeff in expression.as_coefficients_dict().items():
            coeff = float(coeff)
            if term == 1:
                h_final += coeff * np.eye(2**qubit_count)
                continue

            symbols = list(term.free_symbols)
            if len(symbols) == 1:
                index = var_to_index[symbols[0]]
                h_final += coeff * self.sigma_i(qubit_count, index, "z")
            elif len(symbols) == 2:
                i = var_to_index[symbols[0]]
                j = var_to_index[symbols[1]]
                h_final += coeff * (self.sigma_i(qubit_count, i, "z") @ self.sigma_i(qubit_count, j, "z"))
            else:
                raise ValueError("Only linear and quadratic Ising terms are supported.")

        return h_final

    def build_Hinit(self, qubit_count):
        h_init = np.zeros((2**qubit_count, 2**qubit_count))
        for qubit in range(qubit_count):
            h_init += self.sigma_i(qubit_count, qubit, "x")
        return h_init

    def simulate_evolution(self, ising_problem, nb_eigenvalues):
        expression = _coerce_expression(ising_problem)
        variables = sorted(expression.free_symbols, key=str)
        qubit_count = len(variables)

        h_final = self.build_Hfinal(expression, qubit_count)
        h_init = self.build_Hinit(qubit_count)
        spectrum_history = []

        for index in range(len(self.A)):
            h_s = self.A[index] * h_init + self.B[index] * h_final
            eigenvalues = eigh(h_s, eigvals_only=True, subset_by_index=[0, nb_eigenvalues - 1])
            spectrum_history.append(eigenvalues)

        return spectrum_history

    def plot_eigenvalues(self, spectrum_history):
        values = np.array(spectrum_history)
        figure, axis = plt.subplots(figsize=(8, 5))
        for index in range(values.shape[1]):
            axis.plot(values[:, index], label=f"Eigenvalue {index}")
        axis.set_title("Evolution of the lowest eigenvalues")
        axis.set_xlabel("Annealing step")
        axis.set_ylabel("Energy")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
        return axis

    def plot_spectral_gap(self, spectrum_history):
        values = np.array(spectrum_history)
        if values.shape[1] < 2:
            raise ValueError("At least two eigenvalues are required to compute the spectral gap.")

        gap = values[:, 1] - values[:, 0]
        minimum_index = int(np.argmin(gap))

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(gap, label="Spectral gap")
        axis.scatter([minimum_index], [gap[minimum_index]], color="#e74c3c", label="Minimum gap")
        axis.set_title("Spectral gap along the annealing schedule")
        axis.set_xlabel("Annealing step")
        axis.set_ylabel("Gap")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
        return axis
