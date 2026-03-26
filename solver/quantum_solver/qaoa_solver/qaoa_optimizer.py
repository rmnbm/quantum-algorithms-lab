import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham, add_ising_problem_ham

class QAOALocalOptimizer:
    def __init__(self, simulator, gamma_bounds, beta_bounds, p, shots, opt_method):
        self.simulator = simulator
        self.gamma_bounds = gamma_bounds
        self.beta_bounds = beta_bounds
        self.p = p # Number of layers
        self.shots = shots
        self.opt_method = opt_method # The math method used to find the best angles 

    def _run_circuit(self, angles, qc): 
        parameters = {param: [val] for param, val in zip(qc.parameters, angles)}
        
        #We run the circuit on the simulator
        job = self.simulator.run(qc, parameter_binds=[parameters], shots=self.shots)
        
        return job.result().get_counts()[0]
    

    #We created the QAOALocalOptimizer class to automate the search for optimal gamma and beta angles. 
    # The class stores our hardware settings (simulator and shots) and the mathematical rules for the search. 
    # We added a private method _run_circuit which acts as a bridge : it takes a list of trial angles, plugs 
    # them into our quantum circuit, and returns the measurement results from the simulator.



    def get_expectation_value(self, angles, qc, problem):
        #We run the circuit to get bitstring counts 
        counts = self._run_circuit(angles, qc)
        total_cost = 0
        
        #Mapping bitstrings to +1/-1 and calculate cost
        for bitstring, count in counts.items():
            # We map '0' -> 1 and '1' -> -1
            spins = {f's{i+1}': (1 if b == '0' else -1) for i, b in enumerate(reversed(bitstring))}
            
            #We calculate cost 
            cost = float(problem.subs(spins))
            total_cost += cost * count
            
        return total_cost / self.shots

    def optimize(self, problem, p):
        #We build the full p-layer circuit
        n = len(problem.free_symbols)
        qc = QuantumCircuit(n)
        for i in range(n): qc.h(i) #Uniform superposition
        
        for _ in range(p):
            add_ising_problem_ham(qc, problem, n)
            add_ising_mixer_ham(qc, problem, n)
            
        #We define the objective function for the optimizer
        def objective(angles):
            return self.get_expectation_value(angles, qc, problem)

        #We run the automated search (minimize the cost) 
        init_angles = np.random.uniform(0, np.pi, 2 * p)
        res = minimize(objective, init_angles, method=self.opt_method)
        
        # Return results: best cost, best angles 
        return res.fun, res.x