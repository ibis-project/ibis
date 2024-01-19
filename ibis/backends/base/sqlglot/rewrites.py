"""Lower the ibis expression graph to a SQL-like relational algebra."""


from __future__ import annotations

from typing import Literal, Optional

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
from ibis.common.patterns import Object, replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import p
from ibis.expr.schema import Schema

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
def project_to_select(_):
    """Convert a Project node to a Select node."""
    return Select(_.parent, selections=_.values)


@replace(p.Filter)
def filter_to_select(_):
    """Convert a Filter node to a Select node."""
    return Select(_.parent, selections=_.values, predicates=_.predicates)


@replace(p.Sort)
def sort_to_select(_):
    """Convert a Sort node to a Select node."""
    return Select(_.parent, selections=_.values, sort_keys=_.keys)


@replace(p.WindowFunction)
def window_function_to_window(_):
    """Convert a WindowFunction node to a Window node."""
    if isinstance(_.frame, ops.RowsWindowFrame) and _.frame.max_lookback is not None:
        raise NotImplementedError("max_lookback is not supported for SQL backends")
    return Window(
        how=_.frame.how,
        func=_.func,
        start=_.frame.start,
        end=_.frame.end,
        group_by=_.frame.group_by,
        order_by=_.frame.order_by,
    )


@replace(p.Log2)
def replace_log2(_):
    return ops.Log(_.arg, base=2)


@replace(p.Log10)
def replace_log10(_):
    return ops.Log(_.arg, base=10)


@replace(Object(Select, Object(Select)))
def merge_select_select(_):
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

    return Select(
        _.parent.parent,
        selections=selections,
        predicates=tuple(toolz.unique(_.parent.predicates + predicates)),
        sort_keys=tuple(toolz.unique(_.parent.sort_keys + sort_keys)),
    )


def extract_ctes(node):
    result = []
    cte_types = (Select, ops.Aggregate, ops.JoinChain, ops.Set, ops.Limit, ops.Sample)

    g = Graph.from_bfs(node, filter=(ops.Relation, ops.Subquery, ops.JoinLink))
    for node, dependents in g.invert().items():
        if len(dependents) > 1 and isinstance(node, cte_types):
            result.append(node)

    return result


def sqlize(node):
    """Lower the ibis expression graph to a SQL-like relational algebra."""
    assert isinstance(node, ops.Relation)

    step1 = node.replace(
        window_function_to_window
        | project_to_select
        | filter_to_select
        | sort_to_select
    )
    step2 = step1.replace(merge_select_select)

    ctes = extract_ctes(step2)
    subs = {cte: CTE(cte) for cte in ctes}
    step3 = step2.replace(subs)

    return step3, ctes


@replace(p.WindowFunction(p.First(x, y)))
def rewrite_first_to_first_value(_, x, y):
    """Rewrite Ibis's first to first_value when used in a window function."""
    if y is not None:
        raise com.UnsupportedOperationError(
            "`first` with `where` is unsupported in a window function"
        )
    return _.copy(func=ops.FirstValue(x))


@replace(p.WindowFunction(p.Last(x, y)))
def rewrite_last_to_last_value(_, x, y):
    """Rewrite Ibis's last to last_value when used in a window function."""
    if y is not None:
        raise com.UnsupportedOperationError(
            "`last` with `where` is unsupported in a window function"
        )
    return _.copy(func=ops.LastValue(x))


@replace(p.WindowFunction(frame=y @ p.WindowFrame(order_by=())))
def rewrite_empty_order_by_window(_, y, **__):
    return _.copy(frame=y.copy(order_by=(ops.NULL,)))


@replace(p.WindowFunction(p.RowNumber | p.NTile, y))
def exclude_unsupported_window_frame_from_row_number(_, y):
    return ops.Subtract(_.copy(frame=y.copy(start=None, end=0)), 1)


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All,
        y @ p.WindowFrame(start=None),
    )
)
def exclude_unsupported_window_frame_from_ops(_, y):
    return _.copy(frame=y.copy(start=None, end=0, order_by=y.order_by or (ops.NULL,)))
