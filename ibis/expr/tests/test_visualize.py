import os
import pytest

pytest.importorskip('graphviz')

import ibis  # noqa: E402
import ibis.expr.types as ir  # noqa: E402
import ibis.expr.rules as rlz  # noqa: E402
import ibis.expr.visualize as viz  # noqa: E402
import ibis.expr.operations as ops  # noqa: E402

from ibis.expr.signature import Argument as Arg  # noqa: E402

pytestmark = pytest.mark.skipif(
    int(os.environ.get('CONDA_BUILD', 0)) == 1, reason='CONDA_BUILD defined'
)


@pytest.mark.parametrize(
    'expr_func',
    [
        lambda t: t.a,
        lambda t: t.a + t.b,
        lambda t: t.a + t.b > 3 ** t.a,
        lambda t: t[(t.a + t.b * 2 * t.b / t.b ** 3 > 4) & (t.b > 5)],
        lambda t: t[(t.a + t.b * 2 * t.b / t.b ** 3 > 4) & (t.b > 5)].group_by(
            'c'
        ).aggregate(
            amean=lambda f: f.a.mean(),
            bsum=lambda f: f.b.sum(),
        )
    ]
)
def test_exprs(table, expr_func):
    expr = expr_func(table)
    graph = viz.to_graph(expr)
    assert str(hash(table.op())) in graph.source
    assert str(hash(expr.op())) in graph.source


def test_custom_expr():
    class MyExpr(ir.Expr):
        pass

    class MyExprNode(ops.Node):
        foo = Arg(rlz.string)
        bar = Arg(rlz.numeric)

        def output_type(self):
            return MyExpr

    op = MyExprNode('Hello!', 42.3)
    expr = op.to_expr()
    graph = viz.to_graph(expr)
    assert str(hash(op)) in graph.source


def test_custom_expr_with_not_implemented_type():

    class MyExpr(ir.Expr):
        def type(self):
            raise NotImplementedError

        def schema(self):
            raise NotImplementedError

    class MyExprNode(ops.Node):
        foo = Arg(rlz.string)
        bar = Arg(rlz.numeric)

        def output_type(self):
            return MyExpr

    op = MyExprNode('Hello!', 42.3)
    expr = op.to_expr()
    graph = viz.to_graph(expr)
    assert str(hash(op)) in graph.source


@pytest.mark.parametrize('how', ['inner', 'left', 'right', 'outer'])
def test_join(how):
    left = ibis.table([('a', 'int64'), ('b', 'string')])
    right = ibis.table([('b', 'string'), ('c', 'int64')])
    joined = left.join(right, left.b == right.b, how=how)
    result = joined[left.a, right.c]
    graph = viz.to_graph(result)
    assert str(hash(result.op())) in graph.source


def test_sort_by():
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.groupby(t.b).aggregate(
        sum_a=t.a.sum().cast('double')
    ).sort_by('c')
    graph = viz.to_graph(expr)
    assert str(hash(expr.op())) in graph.source


@pytest.mark.skipif(
    bool(os.environ.get('APPVEYOR', None)),
    reason='Not sure what the prerequisites for running this on Windows are'
)
def test_optional_graphviz_repr():
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.groupby(t.b).aggregate(
        sum_a=t.a.sum().cast('double')
    ).sort_by('c')

    # default behavior
    assert expr._repr_png_() is not None

    # turn it off
    ibis.options.graphviz_repr = False
    assert expr._repr_png_() is None

    # turn it back on
    ibis.options.graphviz_repr = True
    assert expr._repr_png_() is not None


def test_between():
    t = ibis.table([('a', 'int64'), ('b', 'string'), ('c', 'int32')])
    expr = t.a.between(1, 1)
    lower_bound, upper_bound = expr.op().args[1:]
    graph = viz.to_graph(expr)
    source = graph.source

    # one for the node itself and one for the edge to between
    assert source.count(str(hash(lower_bound.op()))) == 2
    assert source.count(str(hash(upper_bound.op()))) == 2
