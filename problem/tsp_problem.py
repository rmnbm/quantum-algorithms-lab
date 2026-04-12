from __future__ import annotations

from itertools import permutations
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TspProblem:
    """Represent a travelling salesman problem on a complete weighted graph."""

    objective_sense = "min"

    def __init__(self, distance_matrix, city_names: Sequence[str] | None = None, coordinates=None):
        matrix = np.asarray(distance_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("distance_matrix must be a square matrix.")

        self.distance_matrix = matrix
        self.city_names = list(city_names) if city_names is not None else [f"city_{i}" for i in range(matrix.shape[0])]
        self.coordinates = coordinates

    def evaluate(self, route: Sequence[int]):
        if len(route) != len(self.city_names):
            raise ValueError("A TSP route must visit every city exactly once.")

        total = 0.0
        for start, end in zip(route, route[1:]):
            total += self.distance_matrix[start, end]
        total += self.distance_matrix[route[-1], route[0]]
        return float(total)

    def eval(self, route: Sequence[int]):
        return self.evaluate(route)

    def random_solution(self, rng):
        if len(self.city_names) <= 1:
            return tuple(range(len(self.city_names)))
        tail = list(range(1, len(self.city_names)))
        rng.shuffle(tail)
        return (0, *tail)

    def generate_neighbor_sol(self, route: Sequence[int], rng):
        route = list(route)
        if len(route) <= 2:
            return tuple(route)
        i, j = sorted(rng.choice(np.arange(1, len(route)), size=2, replace=False))
        route[i], route[j] = route[j], route[i]
        return tuple(route)

    def generate_complete_search_space(self) -> Iterable[tuple[int, ...]]:
        if not self.city_names:
            yield tuple()
            return

        start = 0
        for suffix in permutations(range(1, len(self.city_names))):
            yield (start, *suffix)

    def display_solution(self, route: Sequence[int], ax=None):
        graph = nx.complete_graph(len(self.city_names))
        labels = {index: name for index, name in enumerate(self.city_names)}

        if self.coordinates is not None:
            pos = {index: self.coordinates[index] for index in range(len(self.city_names))}
        else:
            pos = nx.circular_layout(graph)

        ax = ax or plt.subplots(figsize=(6, 4))[1]
        nx.draw_networkx(graph, pos=pos, labels=labels, node_color="#d9eaf4", ax=ax)

        cycle_edges = list(zip(route, route[1:])) + [(route[-1], route[0])]
        nx.draw_networkx_edges(graph, pos=pos, edgelist=cycle_edges, edge_color="#e74c3c", width=2.5, ax=ax)
        ax.set_axis_off()
        return ax
