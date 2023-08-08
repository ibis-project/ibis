from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mkdocs_gen_files
import pandas as pd

import ibis
import ibis.expr.operations as ops

if TYPE_CHECKING:
    from collections.abc import Container, Sequence

    from ibis.backends.base import BaseBackend


def get_backends(exclude: Container[str] = ()) -> Sequence[tuple[str, BaseBackend]]:
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


INTERNAL_OPS = {
    # Never translates into anything
    ops.UnresolvedExistsSubquery,
    ops.UnresolvedNotExistsSubquery,
    ops.ScalarParameter,
}

PUBLIC_OPS = frozenset(get_leaf_classes(ops.Value)) - INTERNAL_OPS


def main():
    support = {"operation": [f"{op.__module__}.{op.__name__}" for op in PUBLIC_OPS]}
    support.update(
        (name, list(map(backend.has_operation, PUBLIC_OPS)))
        # exclude flink until https://github.com/apache/flink/pull/23141 is
        # merged and released we also need to roll it into poetry
        for name, backend in get_backends(exclude=("flink",))
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()

    with mkdocs_gen_files.open(Path("backends", "raw_support_matrix.csv"), "w") as f:
        df.to_csv(f, index_label="FullOperation")


main()
