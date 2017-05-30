import itertools
import tempfile

import graphviz as g

from ibis.compat import zip_longest

import ibis.expr.types as ir
import ibis.expr.operations as ops


def get_args(node):

    if isinstance(node, (ops.Aggregation, ops.Selection)):
        return get_args_selection_aggregation(node)
    else:
        args = (arg for arg in node.args if isinstance(arg, ir.Expr))
        try:
            input_type = node.input_type
        except AttributeError:
            names = node._arg_names
        else:
            names = (arg.name for arg in input_type.types)
        return zip_longest(args, names)


def get_args_selection_aggregation(node):
    return zip_longest(
        itertools.chain(
            [node.table],
            itertools.chain.from_iterable(
                getattr(node, argname) or [None]
                for argname in node._arg_names if argname != 'table'
            )
        ),
        itertools.chain(
            ['table'],
            itertools.chain.from_iterable(
                [
                    '{}[{:d}]'.format(argname, i)
                    for i in range(len(getattr(node, argname)))
                ] or [None]
                for argname in node._arg_names if argname != 'table'
            )
        ),
    )


def get_type(expr):
    try:
        return str(expr.type())
    except AttributeError:
        pass

    try:
        schema = expr.schema()
    except AttributeError:
        pass
    else:
        return ''.join(
            '<BR ALIGN="LEFT" />  <I>{}</I>: {}'.format(name, type)
            for name, type in zip(schema.names, schema.types)
        ) + '<BR ALIGN="LEFT" />'

    try:
        # As a last resort try get the name of the output_type class
        return expr.op().output_type().__name__
    except AttributeError:
        return '\u2205'  # empty set character


def get_label(expr, argname=None):
    import ibis.expr.operations as ops

    node = expr.op()
    typename = get_type(expr)
    name = type(node).__name__
    nodename = getattr(node, 'name', argname)
    if nodename is not None:
        if isinstance(node, ops.TableNode):
            label_fmt = '<<I>{}</I>: <B>{}</B>{}>'
        else:
            label_fmt = '<<I>{}</I>: <B>{}</B> \u27f6 {}>'
        label = label_fmt.format(nodename, name, typename)
    else:
        if isinstance(node, ops.TableNode):
            label_fmt = '<<B>{}</B>{}>'
        else:
            label_fmt = '<{} \u27f6 {}>'
        label = label_fmt.format(name, typename)
    return label


def get_arg_names(node):
    try:
        return [arg.name for arg in node.input_type.types]
    except AttributeError:
        return node._arg_names


def to_graph(expr, node_attr=None, edge_attr=None):
    if node_attr is None:
        node_attr = {
            'shape': 'box',
            'fontname': 'Deja Vu Sans Mono',
        }

    if edge_attr is None:
        edge_attr = {
            'dir': 'back',
        }

    stack = [expr]
    seen = set()
    labeled = set()

    graph = g.Digraph(node_attr=node_attr, edge_attr=edge_attr)

    while stack:
        e = stack.pop()
        node = e.op()
        a = str(hash(repr(node)))

        if a not in seen:
            seen.add(a)

            if a not in labeled:
                label = get_label(e)
            else:
                label = None

            graph.node(a, label=label)

            for arg, arg_name in get_args(node):
                if arg is not None:
                    b = str(hash(repr(arg.op())))
                    label = get_label(arg, arg_name)
                    graph.node(b, label=label)
                    labeled.add(b)
                    graph.edge(a, b)
                    stack.append(arg)
    return graph


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
    import ibis

    t = ibis.table(
        [('a', 'int64'), ('b', 'double'), ('c', 'string')], name='t'
    )
    a = t.a
    b = t.b
    filt = t[(a + b * 2 * b / b ** 3 > 4) & (b > 5)]
    expr = filt.groupby(filt.c).aggregate(
        amean=filt.a.mean(),
        bsum=filt.b.sum(),
    )
    expr.visualize()
