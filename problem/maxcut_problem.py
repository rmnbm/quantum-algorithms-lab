from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp

from problem.ising_problem import IsingProblem


class MaxCutProblem:
    """Represent a weighted MaxCut instance on an undirected graph."""

    objective_sense = "max"

    def __init__(self, graph: nx.Graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("MaxCutProblem expects a networkx.Graph instance.")

        self.graph = graph.copy()
        self.nodes = tuple(self.graph.nodes())
        self._node_to_symbol = {node: sp.Symbol(f"s{index}") for index, node in enumerate(self.nodes)}
        self._sympy_expr = None

    @property
    def variables(self):
        return tuple(self._node_to_symbol[node] for node in self.nodes)

    def to_sympy_expr(self):
        if self._sympy_expr is None:
            expr = 0
            for u, v, data in self.graph.edges(data=True):
                weight = data.get("weight", 1.0)
                expr += -weight * self._node_to_symbol[u] * self._node_to_symbol[v]
            self._sympy_expr = sp.expand(expr)

        return self._sympy_expr

    def to_ising_problem(self):
        return IsingProblem(expression=self.to_sympy_expr())

    def _normalize_solution(self, solution: Mapping):
        normalized = {}
        str_to_node = {str(node): node for node in self.nodes}

        for key, value in solution.items():
            if isinstance(key, sp.Symbol):
                symbol = key
            elif key in self._node_to_symbol:
                symbol = self._node_to_symbol[key]
            elif str(key) in str_to_node:
                symbol = self._node_to_symbol[str_to_node[str(key)]]
            else:
                symbol = sp.Symbol(str(key))
            normalized[symbol] = int(value)

        missing = [symbol for symbol in self.variables if symbol not in normalized]
        if missing:
            missing_names = ", ".join(str(symbol) for symbol in missing)
            raise ValueError(f"Missing assignments for variables: {missing_names}")

        return normalized

    def evaluate(self, solution: Mapping):
        return float(self.to_sympy_expr().subs(self._normalize_solution(solution)))

    def eval(self, solution: Mapping):
        return self.evaluate(solution)

    def random_solution(self, rng):
        return {str(symbol): rng.choice([-1, 1]) for symbol in self.variables}

    def generate_neighbor_sol(self, solution: Mapping, rng):
        if not self.variables:
            return {}

        neighbor = {str(key): int(value) for key, value in solution.items()}
        symbol = str(rng.choice(self.variables))
        neighbor[symbol] = -neighbor[symbol]
        return neighbor

    def generate_complete_search_space(self) -> Iterable[dict[str, int]]:
        symbols = [str(symbol) for symbol in self.variables]
        for assignment in product([-1, 1], repeat=len(symbols)):
            yield dict(zip(symbols, assignment))

    def display_graph(self, ax=None):
        ax = ax or plt.subplots(figsize=(6, 4))[1]
        pos = nx.spring_layout(self.graph, seed=42)
        labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx(self.graph, pos=pos, ax=ax, node_color="#d9eaf4", edge_color="#34495e")
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels, ax=ax)
        ax.set_axis_off()
        return ax

    def display_solution(self, solution: Mapping, ax=None):
        normalized = self._normalize_solution(solution)
        ax = ax or plt.subplots(figsize=(6, 4))[1]
        pos = nx.spring_layout(self.graph, seed=42)
        colors = []
        for node in self.nodes:
            symbol = self._node_to_symbol[node]
            colors.append("#e74c3c" if normalized[symbol] == 1 else "#3498db")

        nx.draw_networkx(self.graph, pos=pos, ax=ax, node_color=colors, edge_color="#2c3e50")
        labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels, ax=ax)
        ax.set_axis_off()
        return ax
