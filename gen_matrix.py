from pathlib import Path

import pandas as pd
import tomli

import ibis
import ibis.expr.operations as ops


def get_backends():
    pyproject = tomli.loads(Path("pyproject.toml").read_text())
    backends = pyproject["tool"]["poetry"]["plugins"]["ibis.backends"]
    del backends["spark"]
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
    possible_ops = frozenset(get_leaf_classes(ops.Value))

    support = {
        "operation": [f"`{op.__name__}`" for op in possible_ops],
    }
    support.update(
        (name, list(map(backend.has_operation, possible_ops)))
        for name, backend in get_backends()
    )

    df = pd.DataFrame(support).set_index("operation").sort_index()

    counts = df.sum().sort_values(ascending=False)
    num_ops = len(possible_ops)
    coverage = (
        counts.map(lambda n: f"_{n} ({round(100 * n / num_ops)}%)_")
        .to_frame(name="**API Coverage**")
        .T
    )

    ops_table = df.loc[:, counts.index].replace(ICONS)
    table = pd.concat([coverage, ops_table])
    dst = Path(__file__).parent.joinpath(
        "docs",
        "backends",
        "support_matrix.csv",
    )
    table.to_csv(dst, index_label="Backends")


main()
