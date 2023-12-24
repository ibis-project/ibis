"""Lower the ibis expression graph to a SQL-like relational algebra."""


from __future__ import annotations

from typing import Literal, Optional

from public import public

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.patterns import Object, replace
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.rewrites import p
from ibis.expr.schema import Schema


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
        if v.find(ops.ExistsSubquery, filter=ops.Value):
            return _

    subs = {ops.Field(_.parent, k): v for k, v in _.parent.values.items()}
    selections = {k: v.replace(subs) for k, v in _.selections.items()}
    predicates = tuple(p.replace(subs, filter=ops.Value) for p in _.predicates)
    sort_keys = tuple(s.replace(subs) for s in _.sort_keys)

    return Select(
        _.parent.parent,
        selections=selections,
        predicates=_.parent.predicates + predicates,
        sort_keys=_.parent.sort_keys + sort_keys,
    )


def sqlize(node):
    """Lower the ibis expression graph to a SQL-like relational algebra."""
    step1 = node.replace(
        window_function_to_window
        | project_to_select
        | filter_to_select
        | sort_to_select
    )
    step2 = step1.replace(merge_select_select)
    return step2
