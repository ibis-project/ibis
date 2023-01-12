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

GEOSPATIAL_OPS = frozenset(get_leaf_classes(ops.GeoSpatialUnOp)) | frozenset(
    get_leaf_classes(ops.GeoSpatialBinOp)
)

CORE_OPS = (frozenset(get_leaf_classes(ops.Value))) - INTERNAL_OPS - GEOSPATIAL_OPS

ICONS = {
    True: ":material-check-decagram:{ .verified }",
    False: ":material-cancel:{ .cancel }",
}


def gen_matrix(basename, possible_ops, raw_data_format=False):
    if raw_data_format:
        support = {
            "operation": [f"{op.__module__}.{op.__name__}" for op in possible_ops]
        }
    else:
        support = {"operation": [f"`{op.__name__}`" for op in possible_ops]}

    support.update(
        (name, list(map(backend.has_operation, possible_ops)))
        for name, backend in get_backends()
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()
    if not raw_data_format:
        counts = df.sum().sort_values(ascending=False)
        counts = counts[counts > 0]
        num_ops = len(possible_ops)
        coverage = (
            counts.map(lambda n: f"_{n} ({round(100 * n / num_ops)}%)_")
            .to_frame(name="**API Coverage**")
            .T
        )

        ops_table = df.loc[:, counts.index].replace(ICONS)
        table = pd.concat([coverage, ops_table])
    else:
        table = df

    file_path = Path("backends", f"{basename}_support_matrix.csv")
    local_pth = Path(__file__).parent / "docs" / file_path
    buf = io.StringIO()
    table.to_csv(buf, index_label="Backends")
    local_pth.write_text(buf.getvalue())
    with mkdocs_gen_files.open(file_path, "w") as f:
        f.write(buf.getvalue())


def main():
    gen_matrix(basename="core", possible_ops=CORE_OPS)
    gen_matrix(
        basename="geospatial",
        possible_ops=GEOSPATIAL_OPS,
    )
    gen_matrix(
        basename="raw", possible_ops=CORE_OPS | GEOSPATIAL_OPS, raw_data_format=True
    )


main()
