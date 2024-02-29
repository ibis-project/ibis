"""Lower the ibis expression graph to a SQL-like relational algebra."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, Optional

import toolz
from public import public

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.deferred import var
from ibis.common.graph import Graph
from ibis.common.patterns import Object, Pattern, _, replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import d, p, replace_parameter
from ibis.expr.schema import Schema

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
class Select(ops.Relation):
    """Relation modelled after SQL's SELECT statement."""

    parent: ops.Relation
    selections: FrozenDict[str, ops.Value] = {}
    predicates: VarTuple[ops.Value[dt.Boolean]] = ()
    sort_keys: VarTuple[ops.SortKey] = ()

    @attribute
    def values(self):
        return self.selections

    @attribute
    def schema(self):
        return Schema({k: v.dtype for k, v in self.selections.items()})


@public
class Window(ops.Value):
    """Window modelled after SQL's window statements."""

    how: Literal["rows", "range"]
    func: ops.Reduction | ops.Analytic
    start: Optional[ops.WindowBoundary] = None
    end: Optional[ops.WindowBoundary] = None
    group_by: VarTuple[ops.Column] = ()
    order_by: VarTuple[ops.SortKey] = ()

    shape = ds.columnar

    @attribute
    def dtype(self):
        return self.func.dtype


@replace(p.Project)
def project_to_select(_, **kwargs):
    """Convert a Project node to a Select node."""
    return Select(_.parent, selections=_.values)


@replace(p.Filter)
def filter_to_select(_, **kwargs):
    """Convert a Filter node to a Select node."""
    return Select(_.parent, selections=_.values, predicates=_.predicates)


@replace(p.Sort)
def sort_to_select(_, **kwargs):
    """Convert a Sort node to a Select node."""
    return Select(_.parent, selections=_.values, sort_keys=_.keys)


@replace(p.WindowFunction)
def window_function_to_window(_, **kwargs):
    """Convert a WindowFunction node to a Window node."""
    return Window(
        how=_.frame.how,
        func=_.func,
        start=_.frame.start,
        end=_.frame.end,
        group_by=_.frame.group_by,
        order_by=_.frame.order_by,
    )


@replace(Object(Select, Object(Select)))
def merge_select_select(_, **kwargs):
    """Merge subsequent Select relations into one.

    This rewrites eliminates `_.parent` by merging the outer and the inner
    `predicates`, `sort_keys` and keeping the outer `selections`. All selections
    from the inner Select are inlined into the outer Select.
    """
    # don't merge if either the outer or the inner select has window functions
    for v in _.selections.values():
        if v.find(Window, filter=ops.Value):
            return _
    for v in _.parent.selections.values():
        if v.find((Window, ops.Unnest), filter=ops.Value):
            return _
    for v in _.predicates:
        if v.find((ops.ExistsSubquery, ops.InSubquery), filter=ops.Value):
            return _

    subs = {ops.Field(_.parent, k): v for k, v in _.parent.values.items()}
    selections = {k: v.replace(subs, filter=ops.Value) for k, v in _.selections.items()}
    predicates = tuple(p.replace(subs, filter=ops.Value) for p in _.predicates)
    sort_keys = tuple(s.replace(subs, filter=ops.Value) for s in _.sort_keys)

    unique_predicates = toolz.unique(_.parent.predicates + predicates)
    unique_sort_keys = {s.expr: s for s in _.parent.sort_keys + sort_keys}

    return Select(
        _.parent.parent,
        selections=selections,
        predicates=unique_predicates,
        sort_keys=unique_sort_keys.values(),
    )


def extract_ctes(node):
    result = []
    cte_types = (Select, ops.Aggregate, ops.JoinChain, ops.Set, ops.Limit, ops.Sample)

    g = Graph.from_bfs(node, filter=(ops.Relation, ops.Subquery, ops.JoinLink))
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
    node = node.replace(reduce(operator.or_, rewrites))

    # lower the expression graph to a SQL-like relational algebra
    context = {"params": params}
    sqlized = node.replace(
        replace_parameter
        | project_to_select
        | filter_to_select
        | sort_to_select
        | window_function_to_window,
        context=context,
    )

    # squash subsequent Select nodes into one
    simplified = sqlized.replace(merge_select_select)

    # extract common table expressions while wrapping them in a CTE node
    ctes = extract_ctes(simplified)
    subs = {cte: CTE(cte) for cte in ctes}
    result = simplified.replace(subs)

    return result, ctes


# supplemental rewrites selectively used on a per-backend basis

"""Replace `log2` and `log10` with `log`."""
replace_log2 = p.Log2 >> d.Log(_.arg, base=2)
replace_log10 = p.Log10 >> d.Log(_.arg, base=10)


"""Add an ORDER BY clause to rank window functions that don't have one."""
add_order_by_to_empty_ranking_window_functions = p.WindowFunction(
    func=p.NTile(y),
    frame=p.WindowFrame(order_by=()) >> _.copy(order_by=(y,)),
)

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


@replace(p.WindowFunction(p.First(x, where=y)))
def rewrite_first_to_first_value(_, x, y, **kwargs):
    """Rewrite Ibis's first to first_value when used in a window function."""
    if y is not None:
        raise com.UnsupportedOperationError(
            "`first` with `where` is unsupported in a window function"
        )
    return _.copy(func=ops.FirstValue(x))


@replace(p.WindowFunction(p.Last(x, where=y)))
def rewrite_last_to_last_value(_, x, y, **kwargs):
    """Rewrite Ibis's last to last_value when used in a window function."""
    if y is not None:
        raise com.UnsupportedOperationError(
            "`last` with `where` is unsupported in a window function"
        )
    return _.copy(func=ops.LastValue(x))


@replace(p.WindowFunction(frame=y @ p.WindowFrame(order_by=())))
def rewrite_empty_order_by_window(_, y, **kwargs):
    return _.copy(frame=y.copy(order_by=(ops.NULL,)))


@replace(p.WindowFunction(p.RowNumber | p.NTile, y))
def exclude_unsupported_window_frame_from_row_number(_, y):
    return ops.Subtract(_.copy(frame=y.copy(start=None, end=0)), 1)


@replace(p.WindowFunction(p.MinRank | p.DenseRank, y @ p.WindowFrame(start=None)))
def exclude_unsupported_window_frame_from_rank(_, y):
    return ops.Subtract(
        _.copy(frame=y.copy(start=None, end=0, order_by=y.order_by or (ops.NULL,))), 1
    )


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All,
        y @ p.WindowFrame(start=None),
    )
)
def exclude_unsupported_window_frame_from_ops(_, y, **kwargs):
    return _.copy(frame=y.copy(start=None, end=0, order_by=y.order_by or (ops.NULL,)))
