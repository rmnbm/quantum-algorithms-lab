import sympy as sp
import networkx as nx


class IsingProblem:
    def __init__(self, string=None, graph=None):
        if string is not None:
            self.expr = sp.sympify(string)
            
        elif graph is not None:
            
            expr = 0
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0) 
                s_u = sp.Symbol(str(u))
                s_v = sp.Symbol(str(v))
                expr += weight * s_u * s_v
                
            for node, data in graph.nodes(data=True):
                weight = data.get('weight', 0.0) 
                if weight != 0.0:
                    s_n = sp.Symbol(str(node))
                    expr += weight * s_n
                    
            self.expr = expr
            
        else:
            raise ValueError("Tu dois fournir soit 'cost_string' soit 'graph'.")
            

        self.variables = list(self.expr.free_symbols)

    def to_sympy_expr(self):
        return self.expr


    def get_cost(self, solution):
        subs_dict = {}
        for var, val in solution.items():
            if isinstance(var, str):
                subs_dict[sp.Symbol(var)] = val
            else:
                subs_dict[var] = val
        cost = self.expr.subs(subs_dict)
        return float(cost)
