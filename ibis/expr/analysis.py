from __future__ import annotations

from typing import TYPE_CHECKING

import toolz

import ibis.common.graph as g
import ibis.expr.operations as ops
from ibis import util
from ibis.common.deferred import _, deferred, var
from ibis.common.patterns import In, pattern
from ibis.util import Namespace

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

p = Namespace(pattern, module=ops)
c = Namespace(deferred, module=ops)

x = var("x")
y = var("y")

# ---------------------------------------------------------------------
# Some expression metaprogramming / graph transformations to support
# compilation later


def sub_immediate_parents(node: ops.Node, table: ops.TableNode) -> ops.Node:
    """Replace immediate parent tables in `op` with `table`."""
    parents = find_immediate_parent_tables(node)
    return node.replace(In(parents) >> table)


# TODO(kszucs): should be removed in favor of node.find_topmost(Relation)
def find_immediate_parent_tables(input_node, keep_input=True):
    """Find every first occurrence of a `ir.Table` object in `input_node`.

    This function does not traverse into `Table` objects. For example, the
    underlying `PhysicalTable` of a `Selection` will not be yielded.

    Parameters
    ----------
    input_node
        Input node
    keep_input
        Whether to keep the input when traversing

    Yields
    ------
    ir.Expr
        Parent table expression

    Examples
    --------
    >>> import ibis, toolz
    >>> t = ibis.table([("a", "int64")], name="t")
    >>> expr = t.mutate(foo=t.a + 1)
    >>> (result,) = find_immediate_parent_tables(expr.op())
    >>> result.equals(expr.op())
    True
    >>> (result,) = find_immediate_parent_tables(expr.op(), keep_input=False)
    >>> result.equals(t.op())
    True
    """
    assert all(isinstance(arg, ops.Node) for arg in util.promote_list(input_node))

    def finder(node):
        if isinstance(node, ops.Relation):
            if keep_input or node != input_node:
                return g.halt, node
            else:
                return g.proceed, None

        # HACK: special case ops.Contains to only consider the needle's base
        # table, since that's the only expression that matters for determining
        # cardinality
        elif isinstance(node, ops.InSubquery):
            # we allow InColumn.options to be a column from a foreign table
            return [node.value], None
        else:
            return g.proceed, None

    return list(toolz.unique(g.traverse(finder, input_node)))


# TODO(kszucs): remove it in favor of the rewrites
def windowize_function(expr, default_frame, merge_frames=False):
    func, frame = var("func"), var("frame")

    wrap_analytic = (p.Analytic | p.Reduction) >> c.WindowFunction(_, default_frame)
    merge_windows = p.WindowFunction(func, frame) >> c.WindowFunction(
        func,
        frame.copy(
            order_by=frame.order_by + default_frame.order_by,
            group_by=frame.group_by + default_frame.group_by,
        ),
    )

    node = expr.op()
    if merge_frames:
        # it only happens in ibis.expr.groupby.GroupedTable, but the projector
        # changes the windowization order to put everything here
        node = node.replace(merge_windows, filter=p.Value)
    node = node.replace(wrap_analytic, filter=p.Value & ~p.WindowFunction)

    return node.to_expr()


# TODO(kszucs): should be removed
def find_first_base_table(node):
    def predicate(node):
        if isinstance(node, ops.Relation):
            return g.halt, node
        else:
            return g.proceed, None

    try:
        return next(g.traverse(predicate, node))
    except StopIteration:
        return None


def flatten_predicates(node):
    """Yield the expressions corresponding to the `And` nodes of a predicate.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
    >>> filt = (t.a == 1) & (t.b == "foo")
    >>> predicates = flatten_predicate(filt.op())
    >>> len(predicates)
    2
    >>> predicates[0].to_expr().name("left")
    r0 := UnboundTable: t
      a int64
      b string
    left: r0.a == 1
    >>> predicates[1].to_expr().name("right")
    r0 := UnboundTable: t
      a int64
      b string
    right: r0.b == 'foo'
    """

    def predicate(node):
        if isinstance(node, ops.And):
            return g.proceed, None
        else:
            return g.halt, node

    return list(g.traverse(predicate, node))


def find_predicates(node, flatten=True):
    # TODO(kszucs): consider to remove flatten argument and compose with
    # flatten_predicates instead
    def predicate(node):
        assert isinstance(node, ops.Node), type(node)
        if isinstance(node, ops.Value) and node.dtype.is_boolean():
            if flatten and isinstance(node, ops.And):
                return g.proceed, None
            else:
                return g.halt, node
        return g.proceed, None

    return list(g.traverse(predicate, node))


def find_subqueries(node: ops.Node, min_dependents=1) -> tuple[ops.Node, ...]:
    raise NotImplementedError
    # subquery_dependents = defaultdict(set)
    # for n in filter(None, util.promote_list(node)):
    #     dependents = g.Graph.from_dfs(n).invert()
    #     for u, vs in dependents.toposort().items():
    #         # count the number of table-node dependents on the current node
    #         # but only if the current node is a selection or aggregation
    #         if isinstance(u, (rels.Projection, rels.Aggregation, rels.Limit)):
    #             subquery_dependents[u].update(vs)

    # return tuple(
    #     node
    #     for node, dependents in reversed(subquery_dependents.items())
    #     if len(dependents) >= min_dependents
    # )


def find_toplevel_unnest_children(nodes: Iterable[ops.Node]) -> Iterator[ops.Table]:
    def finder(node):
        return (
            isinstance(node, ops.Value),
            find_first_base_table(node) if isinstance(node, ops.Unnest) else None,
        )

    return g.traverse(finder, nodes)
