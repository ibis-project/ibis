"""Some common rewrite functions to be shared between backends."""
from __future__ import annotations

import toolz

import ibis.expr.operations as ops
from ibis.common.deferred import Item, _, deferred, var
from ibis.common.exceptions import ExpressionError
from ibis.common.patterns import Check, pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)


y = var("y")
name = var("name")


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


ReductionValue = p.Reduction | p.Field(p.Aggregate(groups={}))


@replace(ReductionValue)
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
    values = {k: v.replace(rule) for k, v in _.values.items()}
    return ops.Project(y.parent, values)


@replace(p.Filter(y @ p.Filter))
def subsequent_filters(_, y):
    rule = p.Field(y, name) >> d.Field(y.parent, name)
    preds = tuple(v.replace(rule) for v in _.predicates)
    return ops.Filter(y.parent, y.predicates + preds)


@replace(p.Filter(y @ p.Project))
def reorder_filter_project(_, y):
    rule = p.Field(y, name) >> Item(y.values, name)
    preds = tuple(v.replace(rule) for v in _.predicates)

    inner = ops.Filter(y.parent, preds)
    rule = p.Field(y.parent, name) >> d.Field(inner, name)
    projs = {k: v.replace(rule) for k, v in y.values.items()}

    return ops.Project(inner, projs)


def simplify(node):
    # TODO(kszucs): add a utility to the graph module to do rewrites in multiple
    # passes after each other
    node = node.replace(reorder_filter_project)
    node = node.replace(reorder_filter_project)
    node = node.replace(subsequent_projects | subsequent_filters)
    node = node.replace(complete_reprojection)
    return node
