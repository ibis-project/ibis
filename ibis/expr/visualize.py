import tempfile
from html import escape

import graphviz as gv

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir


def get_type(node):
    try:
        return str(node.output_dtype)
    except (AttributeError, NotImplementedError):
        pass

    try:
        schema = node.schema
    except (AttributeError, NotImplementedError):
        try:
            # As a last resort try get the name of the output_type class
            return node.output_type.__name__
        except (AttributeError, NotImplementedError):
            return '\u2205'  # empty set character
    except com.IbisError:
        assert isinstance(node, ops.Join)
        left_table_name = getattr(node.left, 'name', None) or ops.genname()
        left_schema = node.left.schema
        right_table_name = getattr(node.right, 'name', None) or ops.genname()
        right_schema = node.right.schema
        pairs = [
            (f'{left_table_name}.{left_column}', type)
            for left_column, type in left_schema.items()
        ] + [
            (f'{right_table_name}.{right_column}', type)
            for right_column, type in right_schema.items()
        ]
        schema = ibis.schema(pairs)

    return (
        ''.join(
            '<BR ALIGN="LEFT" />  <I>{}</I>: {}'.format(
                escape(name), escape(str(type))
            )
            for name, type in zip(schema.names, schema.types)
        )
        + '<BR ALIGN="LEFT" />'
    )


def get_label(node, argname=None):
    typename = get_type(node)  # Already an escaped string
    name = type(node).__name__
    nodename = getattr(node, 'name', argname)
    if nodename is not None:
        if isinstance(node, ops.TableNode):
            label_fmt = '<<I>{}</I>: <B>{}</B>{}>'
        else:
            label_fmt = '<<I>{}</I>: <B>{}</B> \u27f6 {}>'
        label = label_fmt.format(escape(nodename), escape(name), typename)
    else:
        if isinstance(node, ops.TableNode):
            label_fmt = '<<B>{}</B>{}>'
        else:
            label_fmt = '<<B>{}</B> \u27f6 {}>'
        label = label_fmt.format(escape(name), typename)
    return label


DEFAULT_NODE_ATTRS = {'shape': 'box', 'fontname': 'Deja Vu Sans Mono'}


def to_graph(op, node_attr=None, edge_attr=None):
    if isinstance(op, ir.Expr):
        return to_graph(op.op())
    elif not isinstance(op, ops.Node):
        raise TypeError(op)

    stack = [(op, op.resolve_name() if op.has_resolved_name() else None)]
    seen = set()
    g = gv.Digraph(
        node_attr=node_attr or DEFAULT_NODE_ATTRS, edge_attr=edge_attr or {}
    )

    g.attr(rankdir='BT')

    while stack:
        op, name = stack.pop()

        if op not in seen:
            seen.add(op)

            vlabel = get_label(op, argname=name)
            vhash = str(hash(op))
            g.node(vhash, label=vlabel)

            for name, arg in zip(op.argnames, op.args):
                if isinstance(arg, ops.Node):
                    uhash = str(hash(arg))
                    ulabel = get_label(arg, argname=name)
                    g.node(uhash, label=ulabel)
                    g.edge(uhash, vhash)
                    stack.append((arg, name))
    return g


def draw(graph, path=None, format='png'):
    piped_source = graph.pipe(format=format)

    if path is None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f'.{format}', mode='wb'
        ) as f:
            f.write(piped_source)
        return f.name
    else:
        with open(path, mode='wb') as f:
            f.write(piped_source)
        return path


if __name__ == '__main__':
    t = ibis.table(
        [('a', 'int64'), ('b', 'double'), ('c', 'string')], name='t'
    )
    left = ibis.table([('a', 'int64'), ('b', 'string')])
    right = ibis.table([('b', 'string'), ('c', 'int64'), ('d', 'string')])
    joined = left.inner_join(right, left.b == right.b)
    df = joined[left.a, right.c.name('b'), right.d.name('c')]
    a = df.a
    b = df.b
    filt = df[(a + b * 2 * b / b**3 > 4) & (b > 5)]
    expr = filt.groupby(filt.c).aggregate(
        amean=filt.a.mean(), bsum=filt.b.sum()
    )
    expr.visualize()
