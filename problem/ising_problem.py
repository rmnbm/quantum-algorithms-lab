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


# # TODO 14: Create a toy example and check that your code is working as expected.

# G_test = nx.Graph()
# G_test.add_node('s1', weight=-1.0)
# G_test.add_node('s2') 
# G_test.add_node('s3')
# G_test.add_edge('s1', 's2', weight=2.0)
# G_test.add_edge('s2', 's3', weight=-1.0)
# G_test.add_edge('s1', 's3', weight=1.5)

# string_test = "+2*s1*s2 -1*s1 +3*s2*s3 +4*s2 -5*s3*s1+ 8*s3"

# mon_probleme_1 = IsingProblem(string= string_test)
# mon_probleme_2 = IsingProblem(graph= G_test)

# ma_solution = {
#     's1': 1,
#     's2': -1,
#     's3': 1
# }

# print(f"1.   La formule est {mon_probleme_1.to_sympy_expr()} et contient {mon_probleme_1.variables}")
# print(f"Pour la configuration {ma_solution},le coût total calculé est : {mon_probleme_1.get_cost(ma_solution)} ")

# print(f"2.   La formule est {mon_probleme_2.to_sympy_expr()} et contient {mon_probleme_2.variables}")
# print(f"Pour la configuration {ma_solution},le coût total calculé est : {mon_probleme_2.get_cost(ma_solution)} ")

