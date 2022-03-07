import io
from pathlib import Path

import pandas as pd
import tomli

import ibis
import ibis.expr.operations as ops


def get_backends():
    pyproject = tomli.loads(Path("pyproject.toml").read_text())
    backends = pyproject["tool"]["poetry"]["plugins"]["ibis.backends"]
    return [
        (backend, getattr(ibis, backend))
        for backend in sorted(backends.keys())
    ]


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        if not child_class.__subclasses__():
            yield child_class
        else:
            yield from get_leaf_classes(child_class)


ICONS = {
    True: ":material-check-decagram:{ .verified }",
    False: ":material-cancel:{ .cancel }",
}


def main():
    dst = Path(__file__).parent.joinpath(
        "docs",
        "backends",
        "support_matrix.csv",
    )
    possible_ops = set(get_leaf_classes(ops.ValueOp))

    support = {
        "operation": [f"`{op.__name__}`" for op in possible_ops],
    }
    for name, backend in get_backends():
        support[name] = list(map(backend.has_operation, possible_ops))

    df = pd.DataFrame(support).set_index("operation").sort_index()
    counts = df.sum().sort_values(ascending=False)
    df = df.loc[:, counts.index].replace(ICONS)
    out = io.BytesIO()
    df.to_csv(out)
    ops_bytes = out.getvalue()

    if not dst.exists() or ops_bytes != dst.read_bytes():
        dst.write_bytes(ops_bytes)


main()
