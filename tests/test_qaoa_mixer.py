import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_mixer_ham
from solver.quantum_solver.qaoa_solver.qaoa_mixers import add_ising_problem_ham
from solver.classical_solver.ising_problem import IsingProblem
from qiskit import QuantumCircuit

def test_add_ising_mixer_ham():
    print("Lancement du test...")
    
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
  
    updated_qc, beta_params = add_ising_mixer_ham(qc, ising_problem=None, n=n_qubits)

    assert len(beta_params) == 1, "Erreur : Il doit y avoir 1 seul paramètre Beta"
    assert len(updated_qc.data) == n_qubits, "Erreur : Il doit y avoir 3 portes ajoutées"

    for instruction in updated_qc.data:
        assert instruction.operation.name == 'rx', "Erreur : La porte doit être RX"
        
    print("Le test a réussi avec succès. ")



def test_add_ising_problem_ham():
    print("⏳ Test de l'Hamiltonien de problème avec SymPy...")
    

    string_test = "+2*s1*s2 -1*s1 +3*s2*s3 +4*s2 -5*s3*s1+ 8*s3"
    mon_probleme = IsingProblem(string=string_test)

    n_qubits = len(mon_probleme.variables)
    qc = QuantumCircuit(n_qubits)
    

    updated_qc, gamma_params = add_ising_problem_ham(qc, mon_probleme, n_qubits)
    

    print("\nCircuit généré :")
    print(updated_qc.draw(output='text'))
    
    assert len(gamma_params) == 1, "Il devrait y avoir 1 paramètre Gamma"
    

    op_counts = updated_qc.count_ops()
    assert op_counts['rz'] == 6, "Erreur sur le nombre de portes RZ"
    assert op_counts['cx'] == 6, "Erreur sur le nombre de portes CNOT"
    
    print("\n✅ Test réussi ! Ton analyseur SymPy marche parfaitement avec Qiskit.")


if __name__ == "__main__":
    test_add_ising_mixer_ham()
    test_add_ising_problem_ham()