"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import toolz

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import ExpressionError
from ibis.common.patterns import Check, pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


y = var("y")
name = var("name")


@replace(p.Field(p.JoinChain))
def peel_join_field(_):
    return _.rel.values[_.name]


@replace(p.Alias(p.ScalarParameter))
def unwrap_scalar_parameter(_):
    """Replace aliased scalar parameters with the parameter itself."""
    return _.arg


def replace_scalar_parameter(params):
    """Replace scalar parameters with their values."""

    @replace(p.ScalarParameter)
    def repl(_):
        return ops.Literal(value=params[_], dtype=_.dtype)

    return repl


@replace(p.FillNa)
def rewrite_fillna(_):
    """Rewrite FillNa expressions to use more common operations."""
    if isinstance(_.replacements, Mapping):
        mapping = _.replacements
    else:
        mapping = {
            name: _.replacements
            for name, type in _.parent.schema.items()
            if type.nullable
        }

    if not mapping:
        return _.parent

    selections = []
    for name in _.parent.schema.names:
        col = ops.TableColumn(_.parent, name)
        if (value := mapping.get(name)) is not None:
            col = ops.Alias(ops.Coalesce((col, value)), name)
        selections.append(col)

    return ops.Project(_.parent, selections)


@replace(p.DropNa)
def rewrite_dropna(_):
    """Rewrite DropNa expressions to use more common operations."""
    if _.subset is None:
        columns = [ops.TableColumn(_.parent, name) for name in _.parent.schema.names]
    else:
        columns = _.subset

    if columns:
        preds = [
            functools.reduce(
                ops.And if _.how == "any" else ops.Or,
                [ops.NotNull(c) for c in columns],
            )
        ]
    elif _.how == "all":
        preds = [ops.Literal(False, dtype=dt.bool)]
    else:
        return _.parent

    return ops.Filter(_.parent, tuple(preds))


@replace(p.Sample)
def rewrite_sample(_):
    """Rewrite Sample as `t.filter(random() <= fraction)`.

    Errors as unsupported if a `seed` is specified.
    """

    if _.seed is not None:
        raise com.UnsupportedOperationError(
            "`Table.sample` with a random seed is unsupported"
        )

    return ops.Filter(_.parent, (ops.LessEqual(ops.RandomScalar(), _.fraction),))


@replace(ops.Analytic)
def project_wrap_analytic(_, rel):
    # Wrap analytic functions in a window function
    return ops.WindowFunction(_, ops.RowsWindowFrame(rel))


@replace(ops.Reduction)
def project_wrap_reduction(_, rel):
    # Query all the tables that the reduction depends on
    if _.relations == {rel}:
        # The reduction is fully originating from the `rel`, so turn
        # it into a window function of `rel`
        return ops.WindowFunction(_, ops.RowsWindowFrame(rel))
    else:
        # 1. The reduction doesn't depend on any table, constructed from
        #    scalar values, so turn it into a scalar subquery.
        # 2. The reduction is originating from `rel` and other tables,
        #    so this is a correlated scalar subquery.
        # 3. The reduction is originating entirely from other tables,
        #    so this is an uncorrelated scalar subquery.
        return ops.ScalarSubquery(_.to_expr().as_table())


def rewrite_project_input(value, relation):
    # we need to detect reductions which are either turned into window functions
    # or scalar subqueries depending on whether they are originating from the
    # relation
    return value.replace(
        project_wrap_analytic | project_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context={"rel": relation},
    )


ReductionLike = p.Reduction | p.Field(p.Aggregate(groups={}))


@replace(ReductionLike)
def filter_wrap_reduction(_):
    # Wrap reductions or fields referencing an aggregation without a group by -
    # which are scalar fields - in a scalar subquery. In the latter case we
    # use the reduction value from the aggregation.
    if isinstance(_, ops.Field):
        value = _.rel.values[_.name]
    else:
        value = _
    return ops.ScalarSubquery(value.to_expr().as_table())


def rewrite_filter_input(value):
    return value.replace(filter_wrap_reduction, filter=p.Value & ~p.WindowFunction)


@replace(p.Analytic | p.Reduction)
def window_wrap_reduction(_, frame):
    # Wrap analytic and reduction functions in a window function. Used in the
    # value.over() API.
    return ops.WindowFunction(_, frame)


@replace(p.WindowFunction)
def window_merge_frames(_, frame):
    # Merge window frames, used in the value.over() and groupby.select() APIs.
    if _.frame.start and frame.start and _.frame.start != frame.start:
        raise ExpressionError(
            "Unable to merge windows with conflicting `start` boundary"
        )
    if _.frame.end and frame.end and _.frame.end != frame.end:
        raise ExpressionError("Unable to merge windows with conflicting `end` boundary")

    start = _.frame.start or frame.start
    end = _.frame.end or frame.end
    group_by = tuple(toolz.unique(_.frame.group_by + frame.group_by))

    order_by = {}
    for sort_key in _.frame.order_by + frame.order_by:
        order_by[sort_key.expr] = sort_key.ascending
    order_by = tuple(ops.SortKey(k, v) for k, v in order_by.items())

    frame = _.frame.copy(start=start, end=end, group_by=group_by, order_by=order_by)
    return ops.WindowFunction(_.func, frame)


def rewrite_window_input(value, frame):
    context = {"frame": frame}
    # if self is a reduction or analytic function, wrap it in a window function
    node = value.replace(
        window_wrap_reduction,
        filter=p.Value & ~p.WindowFunction,
        context=context,
    )
    # if self is already a window function, merge the existing window frame
    # with the requested window frame
    return node.replace(window_merge_frames, filter=p.Value, context=context)


@replace(p.InValues(..., ()))
def empty_in_values_right_side(_):
    """Replace checks against an empty right side with `False`."""
    return ops.Literal(False, dtype=dt.bool)


@replace(
    p.WindowFunction(
        p.PercentRank(y) | p.RankBase(y) | p.CumeDist(y) | p.NTile(y),
        p.WindowFrame(..., order_by=()) >> _.copy(order_by=(y,)),
    )
)
def add_order_by_to_empty_ranking_window_functions(_):
    """Add an ORDER BY clause to rank window functions that don't have one."""
    return _


@replace(
    p.WindowFunction(p.RankBase | p.NTile)
    | p.StringFind
    | p.FindInSet
    | p.ArrayPosition
)
def one_to_zero_index(_, **__):
    """Subtract one from one-index functions."""
    return ops.Subtract(_, 1)


@replace(ops.NthValue)
def add_one_to_nth_value_input(_, **__):
    if isinstance(_.nth, ops.Literal):
        nth = ops.Literal(_.nth.value + 1, dtype=_.nth.dtype)
    else:
        nth = ops.Add(_.nth, 1)
    return _.copy(nth=nth)


# TODO(kszucs): schema comparison should be updated to not distinguish between
# different column order
@replace(p.Project(y @ p.Relation) & Check(_.schema == y.schema))
def complete_reprojection(_, y):
    # TODO(kszucs): this could be moved to the pattern itself but not sure how
    # to express it, especially in a shorter way then the following check
    for name in _.schema:
        if _.values[name] != ops.Field(y, name):
            return _
    return y


@replace(p.Project(y @ p.Project))
def subsequent_projects(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    values = {k: v.replace(rule, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)
    return ops.Filter(y.parent, y.predicates + preds)


@replace(p.Filter(y @ p.Project))
def reorder_filter_project(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    preds = tuple(v.replace(rule, filter=ops.Value) for v in _.predicates)

    inner = ops.Filter(y.parent, preds)
    rule = p.Field(y.parent, name) >> d.Field(inner, name)
    projs = {k: v.replace(rule, filter=ops.Value) for k, v in y.values.items()}

    return ops.Project(inner, projs)


def simplify(node):
    # TODO(kszucs): add a utility to the graph module to do rewrites in multiple
    # passes after each other
    node = node.replace(reorder_filter_project)
    node = node.replace(reorder_filter_project)
    node = node.replace(subsequent_projects | subsequent_filters)
    node = node.replace(complete_reprojection)
    return node
