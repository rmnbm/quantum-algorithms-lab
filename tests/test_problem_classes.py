import numpy as np
import networkx as nx
import sympy as sp

from problem.ising_problem import IsingProblem
from problem.knapsack_problem import KnapsackProblem
from problem.maxcut_problem import MaxCutProblem
from problem.qubo_problem import QuboProblem
from problem.tsp_problem import TspProblem


def test_ising_problem_from_graph():
    graph = nx.Graph()
    graph.add_edge("s0", "s1", weight=2.0)
    graph.add_node("s0", weight=1.0)
    graph.add_node("s1", weight=-1.0)

    problem = IsingProblem(graph=graph)
    assert np.isclose(problem.eval({"s0": 1, "s1": -1}), 0.0)


def test_qubo_problem_to_sympy_expr():
    problem = QuboProblem(np.array([[5.0, -10.0], [0.0, 7.0]]))
    expr = problem.to_sympy_expr()
    expected = 5.0 * problem.variables[0] ** 2 - 10.0 * problem.variables[0] * problem.variables[1] + 7.0 * problem.variables[1] ** 2
    assert sp.simplify(expr - expected) == 0


def test_maxcut_problem_eval():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=3.0)
    graph.add_edge(1, 2, weight=2.0)
    problem = MaxCutProblem(graph)

    assert np.isclose(problem.eval({"s0": 1, "s1": -1, "s2": 1}), 5.0)


def test_knapsack_problem_rejects_infeasible_solution():
    problem = KnapsackProblem(values=[4, 5], weights=[3, 4], capacity=3)
    try:
        problem.eval({"x0": 1, "x1": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("An infeasible solution should raise ValueError.")


def test_tsp_problem_eval():
    matrix = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=float)
    problem = TspProblem(matrix, city_names=["A", "B", "C"])

    assert np.isclose(problem.eval((0, 1, 2)), 7.0)
