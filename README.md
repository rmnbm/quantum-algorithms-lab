# Quantum Algorithms Lab Project

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-17%20passed-2EA44F)
![Qiskit](https://img.shields.io/badge/Qiskit-enabled-6929C4)

A cleaned and reusable implementation of the ESILV quantum algorithms lab sessions, organized as a compact Python toolkit for combinatorial optimization and introductory quantum optimization workflows.

## Overview

This repository combines:

- reusable problem models for MaxCut, QUBO, Ising, Knapsack, and TSP
- classical baselines such as exhaustive search, random search, local search, branch and bound, and simulated annealing
- quantum-oriented helpers for QAOA and adiabatic-style Ising simulations
- the original exploratory notebooks used throughout the lab sessions

The maintained library code lives in `problem/`, `solver/`, `utils/`, and `tests/`. The notebooks remain available as learning and experimentation material.

## Highlights

- Consistent Python APIs across optimization problem classes
- QUBO <-> Ising conversion utilities for classical and quantum workflows
- Lightweight QAOA helpers built on top of Qiskit
- Classical reference solvers for benchmarking and validation

## Project Structure

```text
problem/
  ising_problem.py
  knapsack_problem.py
  maxcut_problem.py
  qubo_problem.py
  tsp_problem.py
solver/
  classical_solver/
  quantum_solver/
tests/
notebooks/
  README.md
```

## Quick Start

### Option 1: install from requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running the Tests

```bash
pytest
```

## Repository Notes

- The reusable project code lives in `problem/`, `solver/`, `utils/`, and `tests/`.
- The notebooks remain in the repository as course-facing exploration material.
- Some notebooks intentionally preserve a worksheet-style structure, while the production-ready implementations are maintained in the Python modules.

## Minimal Example

```python
import numpy as np

from problem.qubo_problem import QuboProblem
from solver.classical_solver.local_search import local_search

qubo = QuboProblem(np.array([[-1.0, 0.5], [0.0, -2.0]]))
best_solution, best_value = local_search(qubo, iterations=100, seed=0)

print(best_solution)
print(best_value)
```

## Notebook Policy

The notebooks are intentionally kept as the original lab exploration material. For reusable project code, use the modules under `problem/`, `solver/`, and `utils/`.

See [notebooks/README.md](/home/rmnbm/quant_algo/notebooks/README.md) for a short guide to the notebook collection.

## Dependencies

Core runtime dependencies:

- numpy
- scipy
- sympy
- networkx
- matplotlib
- pandas
- qiskit
- qiskit-aer

Developer tooling:

- pytest
