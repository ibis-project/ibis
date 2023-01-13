from __future__ import annotations

import io
from pathlib import Path

import mkdocs_gen_files
import pandas as pd
import tomli

import ibis
import ibis.expr.operations as ops


def get_backends():
    pyproject = tomli.loads(Path("pyproject.toml").read_text())
    backends = pyproject["tool"]["poetry"]["plugins"]["ibis.backends"]
    del backends["spark"]
    return [(backend, getattr(ibis, backend)) for backend in sorted(backends.keys())]


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

    file_path = Path("backends", "raw_support_matrix.csv")
    local_path = Path(__file__).parent / "docs" / file_path

    buf = io.StringIO()
    df.to_csv(buf, index_label="FullOperation")

    local_path.write_text(buf.getvalue())
    with mkdocs_gen_files.open(file_path, "w") as f:
        f.write(buf.getvalue())


main()
