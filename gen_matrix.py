import collections
import importlib
import operator
from pathlib import Path

import mkdocs_gen_files
import tabulate
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


with mkdocs_gen_files.open(
    Path("backends") / "99_support_matrix.md",
    "w",
) as f:
    print("# Operation Support Matrix", file=f)
    print('<div class="support-matrix" markdown>', file=f)

    support = collections.defaultdict(list)

    possible_ops = sorted(
        set(get_leaf_classes(ops.ValueOp)), key=operator.attrgetter("__name__")
    )

    for op in possible_ops:
        support["operation"].append(f"`{op.__name__}`")
        for name, backend in get_backends():
            try:
                translator = backend.compiler.translator_class
                ops = translator._registry.keys() | translator._rewrites.keys()
                supported = op in ops
            except AttributeError:
                if name in ("dask", "pandas"):
                    execution = importlib.import_module(
                        f"ibis.backends.{name}.execution"
                    )
                    execute_node = execution.execute_node
                    ops = {op for op, *_ in execute_node.funcs.keys()}
                    supported = op in ops or any(
                        issubclass(op, other) for other in ops
                    )
                else:
                    continue
            if supported:
                support[name].append(":material-check-decagram:{ .verified }")
            else:
                support[name].append(":material-cancel:{ .cancel }")

    table = tabulate.tabulate(support, headers="keys", tablefmt="pipe")
    print(table, file=f)
    print('</div>', file=f)
