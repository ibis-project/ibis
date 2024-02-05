from __future__ import annotations

from pathlib import Path

import pandas as pd

import ibis
import ibis.expr.operations as ops


def get_backends(exclude=()):
    entry_points = sorted(ep.name for ep in ibis.util.backend_entry_points())
    return [
        (backend, getattr(ibis, backend))
        for backend in entry_points
        if backend not in exclude
    ]


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        if not child_class.__subclasses__():
            yield child_class
        else:
            yield from get_leaf_classes(child_class)


def main():
    public_ops = frozenset(get_leaf_classes(ops.Value))
    support = {"operation": [f"{op.__module__}.{op.__name__}" for op in public_ops]}
    support.update(
        (name, list(map(backend.has_operation, public_ops)))
        for name, backend in get_backends()
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()

    with Path(ibis.__file__).parents[1].joinpath(
        "docs", "backends", "raw_support_matrix.csv"
    ).open(mode="w") as f:
        df.to_csv(f, index_label="Operation")


if __name__ == "__main__":
    main()
