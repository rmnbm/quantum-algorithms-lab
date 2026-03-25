import sympy as sp
import networkx as nx

# TODO 1: Create a class MaxCutProblem that can be instanciated with a networkx graph.
class MaxCutProblem:
    def __init__(self, graph):
        self.graph = graph
        self._sympy_expr = None

# TODO 2: Create a method to_sympy_expr() that converts the Maxcut graph to a sympy expression if the expression has not yet been computed.
#         Once computed, store it in a class attribute and check if this attribute exists before the computation (it can be costly to do it several times)
#         You can use the sympify function of the sympy library to create the expression
#         See the documentation here: https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html
    def to_sympy_expr(self):
        if self._sympy_expr is None:
            expr_str = ""

            for u, v, data in self.graph.edges(data=True):
                weight = data.get("weight", 1)
                term = f"-{weight}*(x_{u}*x_{v})"
                expr_str += term + " + "

            if expr_str.endswith(" + "):
                expr_str = expr_str[:-3]

            if expr_str == "":
                expr_str = "0"

            self._sympy_expr = sp.sympify(expr_str)

        return self._sympy_expr


#
# TODO 3: Create a method eval(solution) that evaluates the cost associated to a solution. The solution could be a dictionnary with string keys
# (variables of the cost function) and integer values. You can use the sympy subs method (do not forget to cast your strings to symbols).
    def eval(self, solution):
        expr = self.to_sympy_expr()
        subs_dict = {sp.Symbol(var): value for var, value in solution.items()}
        return expr.subs(subs_dict)
        

