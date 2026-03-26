from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def add_ising_mixer_ham(qc, ising_problem, n):
    layer_id = len(qc.parameters)
    beta = Parameter(f'β_{layer_id}')

    for qubit in range(n):
        qc.rx(2 * beta, qubit)
    
    return qc, [beta]






def add_ising_problem_ham(qc, ising_problem, n):
    layer_id = len(qc.parameters)
    gamma = Parameter(f'γ_{layer_id}')
    
    variables = sorted(list(ising_problem.free_symbols), key=lambda x: str(x))
    var_to_qubit = {var: idx for idx, var in enumerate(variables)}
    
    #We get the coefficients directly from the math expression
    coeff_dict = ising_problem.as_coefficients_dict()

    for term, coeff in coeff_dict.items():
        coeff = float(coeff)
        if term == 1: continue # Skip empty terms
            
        free_syms = list(term.free_symbols)
        
        #We apply the Rz rotations for single variables
        if len(free_syms) == 1:
            qubit_idx = var_to_qubit[free_syms[0]]
            qc.rz(2 * coeff * gamma, qubit_idx)
            
        #We apply CNOT-Rz-CNOT for interactions between two variables
        elif len(free_syms) == 2:
            qubit_1 = var_to_qubit[free_syms[0]]
            qubit_2 = var_to_qubit[free_syms[1]]
            qc.cx(qubit_1, qubit_2)
            qc.rz(2 * coeff * gamma, qubit_2)
            qc.cx(qubit_1, qubit_2)
            
    return qc, [gamma]