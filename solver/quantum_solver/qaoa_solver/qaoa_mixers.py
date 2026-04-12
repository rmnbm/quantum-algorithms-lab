from __future__ import annotations

import sympy as sp
from qiskit.circuit import Parameter, QuantumCircuit


def _coerce_expression(problem):
    if hasattr(problem, "to_sympy_expr"):
        return sp.expand(problem.to_sympy_expr())
    return sp.expand(sp.sympify(problem))


def add_ising_mixer_ham(qc: QuantumCircuit, ising_problem, n: int, layer_index: int | None = None):
    """Append the standard X mixer Hamiltonian layer."""

    layer_id = len(qc.parameters) if layer_index is None else layer_index
    beta = Parameter(f"beta_{layer_id}")
    for qubit in range(n):
        qc.rx(2 * beta, qubit)
    return qc, [beta]


def add_ising_problem_ham(qc: QuantumCircuit, ising_problem, n: int, layer_index: int | None = None):
    """Append the Ising cost Hamiltonian e^(-i gamma H_C) to a circuit."""

    expression = _coerce_expression(ising_problem)
    layer_id = len(qc.parameters) if layer_index is None else layer_index
    gamma = Parameter(f"gamma_{layer_id}")

    variables = sorted(expression.free_symbols, key=str)
    if len(variables) != n:
        raise ValueError("The number of qubits must match the number of Ising variables.")

    var_to_qubit = {var: index for index, var in enumerate(variables)}

    for term, coeff in expression.as_coefficients_dict().items():
        coeff = float(coeff)
        if term == 1:
            continue

        free_symbols = list(term.free_symbols)
        if len(free_symbols) == 1:
            qubit_index = var_to_qubit[free_symbols[0]]
            qc.rz(2 * coeff * gamma, qubit_index)
        elif len(free_symbols) == 2:
            first = var_to_qubit[free_symbols[0]]
            second = var_to_qubit[free_symbols[1]]
            qc.cx(first, second)
            qc.rz(2 * coeff * gamma, second)
            qc.cx(first, second)
        else:
            raise ValueError("Only linear and quadratic Ising terms are supported.")

    return qc, [gamma]
