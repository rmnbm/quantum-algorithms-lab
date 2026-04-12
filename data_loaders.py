from __future__ import annotations

import pickle
from pathlib import Path


def save_instances(instances, filename="instances_pool.pkl"):
    """Persist a pool of benchmark instances to disk."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(instances, handle)
    return path


def load_instances(filename="instances_pool.pkl"):
    """Load a pool of benchmark instances from disk."""

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    with path.open("rb") as handle:
        return pickle.load(handle)
