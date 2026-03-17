import numpy as np
import sympy as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class DwaveSimulator:
    def __init__(self):
        
        # TODO1
        self.A = np.linspace(1.0, 0.0, 101)
        self.B = np.linspace(0.0, 1.0, 101)

        # Matrices de Pauli de base
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.identity = np.eye(2)

    def sigma_i(self, n, i, op_type='z'):
        
        base_op = self.sigma_z if op_type == 'z' else self.sigma_x
        
        # Initialisation avec le qubit 0
        res = base_op if i == 0 else self.identity
        
        # Produit tensoriel avec tous les qubits suivants
        for j in range(1, n):
            current_op = base_op if j == i else self.identity
            res = np.kron(res, current_op)
            
        return res

    def build_Hfinal(self, ising_problem, n):
        # TODO2
        
        dim = 2**n
        Hfinal = np.zeros((dim, dim))
        
        dictionnaire_termes = ising_problem.as_coefficients_dict()

        for terme, poids in dictionnaire_termes.items():
            
            spins_presents = list(terme.atoms(sp.Symbol))
            indices = [int(str(s).replace('s', '')) for s in spins_presents]

            if len(indices) == 1: 
                i = indices[0]
                Hfinal += float(poids) * self.sigma_i(n, i, 'z')
                
            elif len(indices) == 2: 
                i, j = indices[0], indices[1]
                op_i = self.sigma_i(n, i, 'z')
                op_j = self.sigma_i(n, j, 'z')
                Hfinal += float(poids) * (op_i @ op_j)
                
        return Hfinal

    def build_Hinit(self, n):
        # TODO3
        dim = 2**n
        Hinit = np.zeros((dim, dim))
        
        for i in range(n):
            Hinit += self.sigma_i(n, i, 'x')
            
        return Hinit

    def simulate_evolution(self, ising_problem, nb_eigenvalues):
        # TODO4
        spins_presents = list(ising_problem.atoms(sp.Symbol))
        indices = [int(str(s).replace('s', '')) for s in spins_presents]
        n = max(indices) + 1 if indices else 0

        Hfinal = self.build_Hfinal(ising_problem, n)
        Hinit = self.build_Hinit(n)

        historique_valeurs_propres = []

        # Boucle sur le temps s
        for s in range(len(self.A)):
            # H(s) = A(s)H_init + B(s)H_final
            H_s = self.A[s] * Hinit + self.B[s] * Hfinal
            
            # Diagonalisation avec scipy
            val_propres = eigh(H_s, eigvals_only=True, subset_by_index=[0, nb_eigenvalues - 1])
            historique_valeurs_propres.append(val_propres)

        return historique_valeurs_propres

    def plot_eigenvalues(self, historique_valeurs_propres):
        #TODO5
        data = np.array(historique_valeurs_propres)
            
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=f'Eigen {i}')
                
        plt.title('Evolution des énergies')
        plt.xlabel('Temps de recuit (steps)')
        plt.ylabel('Energie')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    def plot_spectral_gap(self, historique_valeurs_propres):
        #TODO5
        data = np.array(historique_valeurs_propres)
        
        if data.shape[1] < 2:
            print("Besoin de 2 valeurs propres minimum.")
            return

        gap = data[:, 1] - data[:, 0]
        g_min = np.min(gap)
        s_min = np.argmin(gap)

        plt.plot(gap, label='Gap spectral')
        plt.plot(s_min, g_min, 'ro')
        
        plt.title(f'Gap Spectral (Minimum = {g_min:.4f} au pas {s_min})')
        plt.xlabel('Temps de recuit (steps)')
        plt.ylabel('Différence d\'énergie')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        