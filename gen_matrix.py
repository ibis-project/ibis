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
    internal_ops = {
        # Never translates into anything
        ops.UnresolvedExistsSubquery,
        ops.UnresolvedNotExistsSubquery,
        ops.ScalarParameter,
    }

    public_ops = frozenset(get_leaf_classes(ops.Value)) - internal_ops
    support = {"operation": [f"{op.__module__}.{op.__name__}" for op in public_ops]}
    support.update(
        (name, list(map(backend.has_operation, public_ops)))
        # exclude flink until https://github.com/apache/flink/pull/23141 is
        # merged and released we also need to roll it into poetry
        for name, backend in get_backends(exclude=("flink",))
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()

    with Path(ibis.__file__).parents[1].joinpath(
        "docs", "backends", "raw_support_matrix.csv"
    ).open(mode="w") as f:
        df.to_csv(f, index_label="FullOperation")


if __name__ == "__main__":
    main()
