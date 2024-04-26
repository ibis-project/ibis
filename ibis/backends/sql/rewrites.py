"""Lower the ibis expression graph to a SQL-like relational algebra."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.deferred import var
from ibis.common.graph import Graph
from ibis.common.patterns import InstanceOf, Pattern, _, replace
from ibis.expr.rewrites import d, p, replace_parameter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

x = var("x")
y = var("y")


@public
class CTE(ops.Relation):
    """Common table expression."""

    parent: ops.Relation

    @attribute
    def schema(self):
        return self.parent.schema

    @attribute
    def values(self):
        return self.parent.values


@public
class FirstValue(ops.Analytic):
    """Retrieve the first element."""

    arg: ops.Column[dt.Any]

    @attribute
    def dtype(self):
        return self.arg.dtype


@public
class LastValue(ops.Analytic):
    """Retrieve the last element."""

    arg: ops.Column[dt.Any]

    @attribute
    def dtype(self):
        return self.arg.dtype


@replace(p.WindowFunction(p.First | p.Last))
def first_to_firstvalue(_, **kwargs):
    """Convert a First or Last node to a FirstValue or LastValue node."""
    if _.func.where is not None:
        raise com.UnsupportedOperationError(
            f"`{type(_.func).__name__.lower()}` with `where` is unsupported "
            "in a window function"
        )
    klass = FirstValue if isinstance(_.func, ops.First) else LastValue
    return _.copy(func=klass(_.func.arg))


def extract_ctes(node):
    result = []
    cte_types = (
        ops.Project,
        ops.Filter,
        ops.Sort,
        ops.Aggregate,
        ops.JoinChain,
        ops.Set,
        ops.Limit,
        ops.Sample,
    )
    dont_count = (ops.Field, ops.CountStar, ops.CountDistinctStar)

    g = Graph.from_bfs(node, filter=~InstanceOf(dont_count))
    for node, dependents in g.invert().items():
        if isinstance(node, ops.View) or (
            len(dependents) > 1 and isinstance(node, cte_types)
        ):
            result.append(node)

    return result


def sqlize(
    node: ops.Node,
    params: Mapping[ops.ScalarParameter, Any],
    rewrites: Sequence[Pattern] = (),
) -> tuple[ops.Node, list[ops.Node]]:
    """Lower the ibis expression graph to a SQL-like relational algebra.

    Parameters
    ----------
    node
        The root node of the expression graph.
    params
        A mapping of scalar parameters to their values.
    rewrites
        Supplementary rewrites to apply to the expression graph.

    Returns
    -------
    Tuple of the rewritten expression graph and a list of CTEs.

    """
    assert isinstance(node, ops.Relation)

    # apply the backend specific rewrites
    if rewrites:
        node = node.replace(reduce(operator.or_, rewrites))

    # lower the expression graph to a SQL-like relational algebra
    node = node.replace(
        first_to_firstvalue | replace_parameter, context={"params": params}
    )

    # extract common table expressions while wrapping them in a CTE node
    ctes = frozenset(extract_ctes(node))

    def wrap(node, _, **kwargs):
        new = node.__recreate__(kwargs)
        return CTE(new) if node in ctes else new

    result = node.replace(wrap)
    ctes = reversed([cte.parent for cte in result.find(CTE)])

    return result, ctes


# supplemental rewrites selectively used on a per-backend basis

"""Replace `log2` and `log10` with `log`."""
replace_log2 = p.Log2 >> d.Log(_.arg, base=2)
replace_log10 = p.Log10 >> d.Log(_.arg, base=10)


"""Add an ORDER BY clause to rank window functions that don't have one."""


@replace(p.WindowFunction(func=p.NTile(y), order_by=()))
def add_order_by_to_empty_ranking_window_functions(_, **kwargs):
    return _.copy(order_by=(y,))


"""Replace checks against an empty right side with `False`."""
empty_in_values_right_side = p.InValues(options=()) >> d.Literal(False, dtype=dt.bool)


@replace(
    p.WindowFunction(p.RankBase | p.NTile)
    | p.StringFind
    | p.FindInSet
    | p.ArrayPosition
)
def one_to_zero_index(_, **kwargs):
    """Subtract one from one-index functions."""
    return ops.Subtract(_, 1)


@replace(ops.NthValue)
def add_one_to_nth_value_input(_, **kwargs):
    if isinstance(_.nth, ops.Literal):
        nth = ops.Literal(_.nth.value + 1, dtype=_.nth.dtype)
    else:
        nth = ops.Add(_.nth, 1)
    return _.copy(nth=nth)


@replace(p.Capitalize)
def rewrite_capitalize(_, **kwargs):
    """Rewrite Capitalize in terms of substring, concat, upper, and lower."""
    first = ops.Uppercase(ops.Substring(_.arg, start=0, length=1))
    # use length instead of length - 1 to avoid backends complaining about
    # asking for negative length
    #
    # there are at most length - 1 characters, so asking for length is fine
    rest = ops.Lowercase(ops.Substring(_.arg, start=1, length=ops.StringLength(_.arg)))
    return ops.StringConcat((first, rest))


@replace(p.Sample)
def rewrite_sample_as_filter(_, **kwargs):
    """Rewrite Sample as `t.filter(random() <= fraction)`.

    Errors as unsupported if a `seed` is specified.
    """
    if _.seed is not None:
        raise com.UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )
    return ops.Filter(_.parent, (ops.LessEqual(ops.RandomScalar(), _.fraction),))


@replace(p.WindowFunction(order_by=()))
def rewrite_empty_order_by_window(_, **kwargs):
    return _.copy(order_by=(ops.NULL,))


@replace(p.WindowFunction(p.RowNumber | p.NTile))
def exclude_unsupported_window_frame_from_row_number(_, **kwargs):
    return ops.Subtract(_.copy(start=None, end=0), 1)


@replace(p.WindowFunction(p.MinRank | p.DenseRank, start=None))
def exclude_unsupported_window_frame_from_rank(_, **kwargs):
    return ops.Subtract(
        _.copy(start=None, end=0, order_by=_.order_by or (ops.NULL,)), 1
    )


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All, start=None
    )
)
def exclude_unsupported_window_frame_from_ops(_, **kwargs):
    return _.copy(start=None, end=0, order_by=_.order_by or (ops.NULL,))
