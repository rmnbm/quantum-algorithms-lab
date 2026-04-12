from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping

import networkx as nx
import sympy as sp


class IsingProblem:
    """Represent an Ising optimization problem on spins in {-1, +1}."""

    objective_sense = "min"

    def __init__(self, string: str | None = None, graph: nx.Graph | None = None, expression=None):
        provided = [string is not None, graph is not None, expression is not None]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of 'string', 'graph', or 'expression'.")

        if string is not None:
            expr = sp.sympify(string)
        elif graph is not None:
            expr = self._graph_to_expr(graph)
        else:
            expr = sp.sympify(expression)

        self.expr = sp.expand(expr)
        self.variables = tuple(sorted(self.expr.free_symbols, key=str))

    @staticmethod
    def _graph_to_expr(graph: nx.Graph):
        expr = 0
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            expr += weight * sp.Symbol(str(u)) * sp.Symbol(str(v))

        for node, data in graph.nodes(data=True):
            weight = data.get("weight", 0.0)
            if weight:
                expr += weight * sp.Symbol(str(node))

        return expr

    @property
    def free_symbols(self):
        return self.expr.free_symbols

    def as_coefficients_dict(self):
        return self.expr.as_coefficients_dict()

    def subs(self, *args, **kwargs):
        return self.expr.subs(*args, **kwargs)

    def to_sympy_expr(self):
        return self.expr

    def _normalize_solution(self, solution: Mapping):
        normalized: Dict[sp.Symbol, int] = {}
        for key, value in solution.items():
            symbol = key if isinstance(key, sp.Symbol) else sp.Symbol(str(key))
            normalized[symbol] = int(value)

        missing = [var for var in self.variables if var not in normalized]
        if missing:
            missing_names = ", ".join(str(var) for var in missing)
            raise ValueError(f"Missing assignments for variables: {missing_names}")

        return normalized

    def evaluate(self, solution: Mapping):
        return float(self.expr.subs(self._normalize_solution(solution)))

    def get_cost(self, solution: Mapping):
        return self.evaluate(solution)

    def eval(self, solution: Mapping):
        return self.evaluate(solution)

    def random_solution(self, rng):
        return {str(var): rng.choice([-1, 1]) for var in self.variables}

    def generate_neighbor_sol(self, solution: Mapping, rng):
        if not self.variables:
            return {}

        neighbor = {str(k): int(v) for k, v in solution.items()}
        target = str(rng.choice(self.variables))
        neighbor[target] = -neighbor[target]
        return neighbor

    def generate_complete_search_space(self) -> Iterable[dict[str, int]]:
        names = [str(var) for var in self.variables]
        for assignment in product([-1, 1], repeat=len(names)):
            yield dict(zip(names, assignment))
