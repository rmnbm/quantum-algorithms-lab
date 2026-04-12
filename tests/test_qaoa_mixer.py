import numpy as np
from qiskit_aer import AerSimulator

from problem.ising_problem import IsingProblem
from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham, add_ising_problem_ham
from solver.quantum_solver.qaoa_solver.qaoa_optimizer import QAOALocalOptimizer


def test_add_ising_mixer_ham_adds_one_rx_per_qubit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    updated_qc, beta_params = add_ising_mixer_ham(qc, ising_problem=None, n=3, layer_index=0)

    assert len(beta_params) == 1
    assert len(updated_qc.data) == 3
    assert all(instruction.operation.name == "rx" for instruction in updated_qc.data)


def test_add_ising_problem_ham_builds_expected_gate_pattern():
    from qiskit import QuantumCircuit

    problem = IsingProblem(string="+2*s0*s1 -1*s0 +3*s1*s2 +4*s1 -5*s2*s0 + 8*s2")
    qc = QuantumCircuit(3)
    updated_qc, gamma_params = add_ising_problem_ham(qc, problem, 3, layer_index=0)

    op_counts = updated_qc.count_ops()
    assert len(gamma_params) == 1
    assert op_counts["rz"] == 6
    assert op_counts["cx"] == 6


def test_qaoa_optimizer_runs_end_to_end():
    problem = IsingProblem(string="s0*s1 - s0 + 0.5*s1")
    optimizer = QAOALocalOptimizer(
        simulator=AerSimulator(),
        gamma_bounds=(0.0, np.pi),
        beta_bounds=(0.0, np.pi),
        p=1,
        shots=128,
        opt_method="COBYLA",
    )

    best_cost, angles = optimizer.optimize(problem)

    assert isinstance(best_cost, float)
    assert len(angles) == 2
