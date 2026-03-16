import numpy as np
import scipy.sparse as sp


class QuboProblem():
    def __init__(self, M):

        if not (sp.issparse(M) or isinstance(M, np.ndarray)):
            M = np.array(M)

        if self._is_symmetric(M) or self._is_triangular_upper(M):
            self.matrix = M
        else:
            raise Exception("The matrix must be either symmetric or triangular upper.")

    def _is_symmetric(self,M):
        if sp.issparse(M):
            return (M != M.T).nnz == 0
        return np.allclose(M,M.T)

    def _is_triangular_upper(self,M):
        if sp.issparse(M):
            return sp.tril(M, k= -1).nnz == 0
        return np.allclose(M, np.triu(M))
        
    def to_sympy_expr(self):
        expr = ""
        n_rows, n_cols = self.matrix.shape
        x = symbols(f'x0:{n_rows}')
        terms = []
        if sp.issparse(self.matrix):
            rows, cols, values = sp.find(self.matrix)
            for i, j, v in zip(rows, cols, values):
                terms.append(v*x[i]*x[j])
        else:
            for i in range(n_rows):
                for j in range(n_cols):
                    if self.matrix[i,j] != 0:
                        terms.append(self.matrix[i, j] * x[i] * x[j])
        
        self._sympy_expr = Add(*terms)
        return self._sympy_expr

    def eval(self, solution):
            n = self.matrix.shape[0]
            x = np.zeros(n)
            for i in range(n):
                x[i] = solution.get(f'x{i}', 0)
                
            cost = x.T @ self.matrix @ x
            
            return float(np.asarray(cost).item())


if __name__ == "__main__":
    Q_data = sp.csr_matrix([
        [5, -10],
        [0,  7]
    ])

    qubo = QuboProblem(Q_data)
    
    print("--- Testing TODO 9 (Sympy Expression) ---")
    print(f"Expression: {qubo.to_sympy_expr()}")
    
    print("\n--- Testing TODO 10 (Evaluation) ---")
    sol = {'x0': 1, 'x1': 1}
    print(f"Cost of {sol}: {qubo.eval(sol)}")