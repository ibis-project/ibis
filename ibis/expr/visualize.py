from __future__ import annotations

import contextlib
import sys
import tempfile
from html import escape

import graphviz as gv

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.common.graph import Graph


def get_type(node):
    with contextlib.suppress(AttributeError, NotImplementedError):
        return escape(str(node.dtype))

    try:
        schema = node.schema
    except (AttributeError, NotImplementedError):
        # TODO(kszucs): this branch should be removed
        try:
            # As a last resort try get the name of the output_type class
            return node.output_type.__name__
        except (AttributeError, NotImplementedError):
            return "\u2205"  # empty set character
    except com.IbisError:
        assert isinstance(node, ops.Join)
        left_table_name = getattr(node.left, "name", None) or ops.genname()
        left_schema = node.left.schema
        right_table_name = getattr(node.right, "name", None) or ops.genname()
        right_schema = node.right.schema
        pairs = [
            (f"{left_table_name}.{left_column}", type)
            for left_column, type in left_schema.items()
        ] + [
            (f"{right_table_name}.{right_column}", type)
            for right_column, type in right_schema.items()
        ]
        schema = ibis.schema(pairs)

    return '<BR ALIGN="LEFT" />' + '<BR ALIGN="LEFT" />'.join(
        f"<I>{escape(name)}</I>: {escape(str(type))}"
        for name, type in zip(schema.names, schema.types)
    )


def get_label(node):
    typename = get_type(node)  # Already an escaped string
    name = type(node).__name__
    nodename = (
        node.name
        if isinstance(
            node,
            (
                ops.Literal,
                ops.TableColumn,
                ops.Alias,
                ops.PhysicalTable,
                ops.window.RangeWindowFrame,
            ),
        )
        else None
    )
    if nodename is not None:
        # [TODO] Don't show nodename because it's too long and ruins the image
        if isinstance(node, ops.window.RangeWindowFrame):
            label_fmt = "<<B>{}</B>>"
            label = label_fmt.format(escape(name))
        else:
            if isinstance(node, ops.TableNode):
                label_fmt = "<<I>{}</I>: <B>{}</B>{}>"
            else:
                label_fmt = '<<I>{}</I>: <B>{}</B><BR ALIGN="LEFT" />:: {}>'
            # typename is already escaped
            label = label_fmt.format(escape(nodename), escape(name), typename)
    else:
        if isinstance(node, ops.TableNode):
            label_fmt = "<<B>{}</B>{}>"
        else:
            label_fmt = '<<B>{}</B><BR ALIGN="LEFT" />:: {}>'
        label = label_fmt.format(escape(name), typename)
    return label


DEFAULT_NODE_ATTRS = {"shape": "box", "fontname": "Deja Vu Sans Mono"}
DEFAULT_EDGE_ATTRS = {"fontname": "Deja Vu Sans Mono"}


def to_graph(expr, node_attr=None, edge_attr=None, label_edges: bool = False):
    graph = Graph.from_bfs(expr.op(), filter=ops.Node)

    g = gv.Digraph(
        node_attr=node_attr or DEFAULT_NODE_ATTRS,
        edge_attr=edge_attr or DEFAULT_EDGE_ATTRS,
    )

    g.attr(rankdir="BT")

    seen = set()
    edges = set()

    for v, us in graph.items():
        vhash = str(hash(v))
        if v not in seen:
            g.node(vhash, label=get_label(v))
            seen.add(v)

        for u in us:
            uhash = str(hash(u))
            if u not in seen:
                g.node(uhash, label=get_label(u))
                seen.add(u)
            if (edge := (u, v)) not in edges:
                if not label_edges:
                    label = None
                else:
                    for name, arg in zip(v.argnames, v.args):
                        if isinstance(arg, tuple) and u in arg:
                            index = arg.index(u)
                            name = f"{name}[{index}]"
                            break
                        elif arg == u:
                            break
                    else:
                        name = None
                    label = f"<.{name}>"

                g.edge(uhash, vhash, label=label)
                edges.add(edge)
    return g


def draw(graph, path=None, format="png", verbose: bool = False):
    if verbose:
        print(graph.source, file=sys.stderr)  # noqa: T201

    piped_source = graph.pipe(format=format)

    if path is None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{format}", mode="wb"
        ) as f:
            f.write(piped_source)
        return f.name
    else:
        with open(path, mode="wb") as f:
            f.write(piped_source)
        return path


if __name__ == "__main__":
    from argparse import ArgumentParser

    from ibis import _

    p = ArgumentParser(
        description="Render a GraphViz SVG of an example ibis expression."
    )

    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print GraphViz DOT code to stderr.",
    )
    p.add_argument(
        "-l",
        "--label-edges",
        action="store_true",
        help="Show operation inputs as edge labels.",
    )

    args = p.parse_args()

    left = ibis.table(dict(a="int64", b="string"), name="left")
    right = ibis.table(dict(b="string", c="int64", d="string"), name="right")
    expr = (
        left.inner_join(right, "b")
        .select(left.a, b=right.c, c=right.d)
        .filter((_.a + _.b * 2 * _.b / _.b**3 > 4) & (_.b > 5))
        .group_by(_.c)
        .having(_.a.mean() > 0.0)
        .aggregate(a_mean=_.a.mean(), b_sum=_.b.sum())
        .mutate(
            arrays=ibis.array([1, 2, 3]),
            maps=ibis.map({"a": 1, "b": 2}),
            structs=ibis.struct({"a": [1, 2, 3], "b": {"c": 1, "d": 2}}),
        )
    )
    expr.visualize(verbose=args.verbose > 0, label_edges=args.label_edges)
