import tempfile
from html import escape

import graphviz as gv

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir


def get_type(expr):
    try:
        return str(expr.type())
    except (AttributeError, NotImplementedError):
        pass

    try:
        schema = expr.schema()
    except (AttributeError, NotImplementedError):
        try:
            # As a last resort try get the name of the output_type class
            return expr.op().output_type().__name__
        except (AttributeError, NotImplementedError):
            return '\u2205'  # empty set character
    except com.IbisError:
        op = expr.op()
        assert isinstance(op, ops.Join)
        left_table_name = getattr(op.left.op(), 'name', None) or ops.genname()
        left_schema = op.left.schema()
        right_table_name = (
            getattr(op.right.op(), 'name', None) or ops.genname()
        )
        right_schema = op.right.schema()
        pairs = [
            ('{}.{}'.format(left_table_name, left_column), type)
            for left_column, type in left_schema.items()
        ] + [
            ('{}.{}'.format(right_table_name, right_column), type)
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


def get_label(expr, argname=None):
    import ibis.expr.operations as ops

    node = expr.op()
    typename = get_type(expr)  # Already an escaped string
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


def to_graph(expr, node_attr=None, edge_attr=None):
    stack = [(expr, expr._safe_name)]
    seen = set()
    g = gv.Digraph(
        node_attr=node_attr or DEFAULT_NODE_ATTRS, edge_attr=edge_attr or {}
    )

    g.attr(rankdir='BT')

    while stack:
        e, ename = stack.pop()
        vkey = e._key, ename

        if vkey not in seen:
            seen.add(vkey)

            vlabel = get_label(e, argname=ename)
            vhash = str(hash(vkey))
            g.node(vhash, label=vlabel)

            node = e.op()
            args = node.args
            for arg, name in zip(args, node.signature.names()):
                if isinstance(arg, ir.Expr):
                    u = arg, name
                    ukey = arg._key, name
                    uhash = str(hash(ukey))
                    ulabel = get_label(arg, argname=name)
                    g.node(uhash, label=ulabel)
                    g.edge(uhash, vhash)
                    stack.append(u)
    return g


def draw(graph, path=None, format='png'):
    piped_source = graph.pipe(format=format)

    if path is None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.{}'.format(format), mode='wb'
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
    filt = df[(a + b * 2 * b / b ** 3 > 4) & (b > 5)]
    expr = filt.groupby(filt.c).aggregate(
        amean=filt.a.mean(), bsum=filt.b.sum()
    )
    expr.visualize()
