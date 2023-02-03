from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files
import pandas as pd

import ibis
import ibis.expr.operations as ops


def get_backends():
    entry_points = sorted(
        name for ep in ibis.util.backend_entry_points() if (name := ep.name) != "spark"
    )
    return [(backend, getattr(ibis, backend)) for backend in entry_points]


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        if not child_class.__subclasses__():
            yield child_class
        else:
            yield from get_leaf_classes(child_class)


INTERNAL_OPS = {
    # Never translates into anything
    ops.UnresolvedExistsSubquery,
    ops.UnresolvedNotExistsSubquery,
    ops.ScalarParameter,
}

PUBLIC_OPS = (frozenset(get_leaf_classes(ops.Value))) - INTERNAL_OPS


def main():
    support = {"operation": [f"{op.__module__}.{op.__name__}" for op in PUBLIC_OPS]}
    support.update(
        (name, list(map(backend.has_operation, PUBLIC_OPS)))
        for name, backend in get_backends()
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()

    with mkdocs_gen_files.open(Path("backends", "raw_support_matrix.csv"), "w") as f:
        df.to_csv(f, index_label="FullOperation")


main()
