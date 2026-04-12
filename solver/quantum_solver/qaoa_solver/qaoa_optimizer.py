from __future__ import annotations

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham, add_ising_problem_ham


def _coerce_expression(problem):
    if hasattr(problem, "to_sympy_expr"):
        return sp.expand(problem.to_sympy_expr())
    return sp.expand(sp.sympify(problem))


class QAOALocalOptimizer:
    """Optimize QAOA angles for a quadratic Ising cost Hamiltonian."""

    def __init__(self, simulator, gamma_bounds, beta_bounds, p, shots, opt_method):
        self.simulator = simulator
        self.gamma_bounds = gamma_bounds
        self.beta_bounds = beta_bounds
        self.p = p
        self.shots = shots
        self.opt_method = opt_method

    def _build_parametrized_circuit(self, problem, p):
        expression = _coerce_expression(problem)
        variables = sorted(expression.free_symbols, key=str)
        circuit = QuantumCircuit(len(variables))
        circuit.h(range(len(variables)))

        parameter_order = []
        for layer in range(p):
            circuit, gamma_params = add_ising_problem_ham(circuit, expression, len(variables), layer_index=layer)
            circuit, beta_params = add_ising_mixer_ham(circuit, expression, len(variables), layer_index=layer)
            parameter_order.extend(gamma_params + beta_params)

        measured_circuit = circuit.copy()
        measured_circuit.measure_all()
        return expression, variables, measured_circuit, parameter_order

    def _run_circuit(self, angles, qc, parameter_order):
        bindings = {param: value for param, value in zip(parameter_order, angles)}
        bound_circuit = qc.assign_parameters(bindings)
        job = self.simulator.run(bound_circuit, shots=self.shots)
        return job.result().get_counts()

    def get_expectation_value(self, angles, qc, parameter_order, problem, variables):
        counts = self._run_circuit(angles, qc, parameter_order)
        total_cost = 0.0

        for bitstring, count in counts.items():
            spins = {
                variable: (1 if bit == "0" else -1)
                for variable, bit in zip(variables, reversed(bitstring))
            }
            total_cost += float(problem.subs(spins)) * count

        return total_cost / self.shots

    def optimize(self, problem, p=None):
        depth = self.p if p is None else p
        expression, variables, measured_circuit, parameter_order = self._build_parametrized_circuit(problem, depth)

        def objective(angles):
            return self.get_expectation_value(angles, measured_circuit, parameter_order, expression, variables)

        bounds = []
        initial_angles = []
        for _ in range(depth):
            bounds.append(self.gamma_bounds)
            initial_angles.append(np.random.uniform(*self.gamma_bounds))
            bounds.append(self.beta_bounds)
            initial_angles.append(np.random.uniform(*self.beta_bounds))
        initial_angles = np.array(initial_angles, dtype=float)
        result = minimize(objective, initial_angles, method=self.opt_method, bounds=bounds if self.opt_method != "COBYLA" else None)
        return float(result.fun), result.x
