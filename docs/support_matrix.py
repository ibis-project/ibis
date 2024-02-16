from __future__ import annotations

import pandas as pd

import ibis
import ibis.expr.operations as ops


def make_support_matrix():
    """Construct the backend operation support matrix data."""

    from ibis.backends.base.sqlglot.compiler import ALL_OPERATIONS

    support_matrix_ignored_operations = (ops.ScalarParameter,)

    public_ops = ALL_OPERATIONS.difference(support_matrix_ignored_operations)

    assert public_ops

    support = {"Operation": [f"{op.__module__}.{op.__name__}" for op in public_ops]}
    support.update(
        (backend, list(map(getattr(ibis, backend).has_operation, public_ops)))
        for backend in sorted(ep.name for ep in ibis.util.backend_entry_points())
    )

    support_matrix = (
        pd.DataFrame(support)
        .assign(splits=lambda df: df.Operation.str.findall("[a-zA-Z_][a-zA-Z_0-9]*"))
        .assign(
            Category=lambda df: df.splits.str[-2],
            Operation=lambda df: df.splits.str[-1],
        )
        .drop(["splits"], axis=1)
        .set_index(["Category", "Operation"])
        .sort_index()
    )
    all_visible_ops_count = len(support_matrix)
    assert all_visible_ops_count

    coverage = pd.Index(
        support_matrix.sum()
        .map(lambda n: f"{n} ({round(100 * n / all_visible_ops_count)}%)")
        .T
    )
    support_matrix.columns = pd.MultiIndex.from_tuples(
        list(zip(support_matrix.columns, coverage)), names=("Backend", "API coverage")
    )
    return support_matrix


if __name__ == "__main__":
    print(make_support_matrix())  # noqa: T201
