"""Code for computing window functions with ibis.
"""

import operator
import re

from collections import OrderedDict

import six

import numpy as np

import toolz

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.common as com
import ibis.expr.window as win
import ibis.expr.operations as ops
import ibis.expr.datatypes as dt

import ibis.pandas.aggcontext as agg_ctx

from ibis.pandas.core import (
    execute, integer_types, simple_types, date_types, timestamp_types,
    timedelta_types
)
from ibis.pandas.dispatch import execute_node, pre_execute
from ibis.pandas.execution import util


def _post_process_empty(scalar, parent, order_by, group_by):
    assert not order_by and not group_by
    index = parent.index
    result = pd.Series([scalar]).repeat(len(index))
    result.index = index
    return result


def _post_process_group_by(series, parent, order_by, group_by):
    assert not order_by and group_by
    return series


def _post_process_order_by(series, parent, order_by, group_by):
    assert order_by and not group_by
    indexed_parent = parent.set_index(order_by)
    index = indexed_parent.index
    names = index.names
    if len(names) > 1:
        series = series.reorder_levels(names)
    series = series.iloc[index.argsort(kind='mergesort')]
    return series


def _post_process_group_by_order_by(series, parent, order_by, group_by):
    indexed_parent = parent.set_index(group_by + order_by, append=True)
    index = indexed_parent.index

    # get the names of the levels that will be in the result
    series_index_names = frozenset(series.index.names)

    # get the levels common to series.index, in the order that they occur in
    # the parent's index
    reordered_levels = [
        name for name in index.names if name in series_index_names
    ]

    if len(reordered_levels) > 1:
        series = series.reorder_levels(reordered_levels)
    return series


@execute_node.register(ops.WindowOp, pd.Series, win.Window)
def execute_window_op(
    op, data, window, scope=None, aggcontext=None, clients=None, **kwargs
):
    operand = op.expr
    # pre execute "manually" here because otherwise we wouldn't pickup
    # relevant scope changes from the child operand since we're managing
    # execution of that by hand
    operand_op = operand.op()
    pre_executed_scope = pre_execute(
        operand_op, *clients, scope=scope, aggcontext=aggcontext, **kwargs)
    scope = toolz.merge(scope, pre_executed_scope)

    root, = op.root_tables()
    root_expr = root.to_expr()
    data = execute(root_expr, scope=scope, aggcontext=aggcontext, **kwargs)

    following = window.following
    order_by = window._order_by

    if order_by and following != 0 and not isinstance(
        operand_op, ops.ShiftBase
    ):
        raise com.OperationNotDefinedError(
            'Window functions affected by following with order_by are not '
            'implemented'
        )

    group_by = window._group_by
    grouping_keys = [
        key_op.name if isinstance(key_op, ops.TableColumn) else execute(
            key,
            aggcontext=aggcontext,
            **kwargs
        ) for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by
    if not order_by:
        ordering_keys = ()

    if group_by:
        if order_by:
            sorted_df, grouping_keys, ordering_keys = (
                util.compute_sorted_frame(
                    data, order_by, group_by=group_by, **kwargs))
            source = sorted_df.groupby(grouping_keys, sort=True)
            post_process = _post_process_group_by_order_by
        else:
            source = data.groupby(grouping_keys, sort=False)
            post_process = _post_process_group_by
    else:
        if order_by:
            source, grouping_keys, ordering_keys = (
                util.compute_sorted_frame(data, order_by, **kwargs))
            post_process = _post_process_order_by
        else:
            source = data
            post_process = _post_process_empty

    new_scope = toolz.merge(
        scope,
        OrderedDict((t, source) for t in operand.op().root_tables()),
        factory=OrderedDict,
    )

    # figure out what the dtype of the operand is
    operand_type = operand.type()
    if isinstance(operand_type, dt.Integer) and operand_type.nullable:
        operand_dtype = np.float64
    else:
        operand_dtype = operand.type().to_pandas()

    # no order by or group by: default summarization aggcontext
    #
    # if we're reducing and we have an order by expression then we need to
    # expand or roll.
    #
    # otherwise we're transforming
    if not grouping_keys and not ordering_keys:
        aggcontext = agg_ctx.Summarize()
    elif isinstance(operand.op(), ops.Reduction) and ordering_keys:
        # XXX(phillipc): What a horror show
        preceding = window.preceding
        if preceding is not None:
            aggcontext = agg_ctx.Moving(
                preceding,
                parent=source,
                group_by=grouping_keys,
                order_by=ordering_keys,
                dtype=operand_dtype,
            )
        else:
            # expanding window
            aggcontext = agg_ctx.Cumulative(
                parent=source,
                group_by=grouping_keys,
                order_by=ordering_keys,
                dtype=operand_dtype,
            )
    else:
        # groupby transform (window with a partition by clause in SQL parlance)
        aggcontext = agg_ctx.Transform(
            parent=source,
            group_by=grouping_keys,
            order_by=ordering_keys,
            dtype=operand_dtype,
        )

    result = execute(operand, scope=new_scope, aggcontext=aggcontext, **kwargs)
    series = post_process(result, data, ordering_keys, grouping_keys)
    assert len(data) == len(series), \
        'input data source and computed column do not have the same length'
    return series


@execute_node.register(ops.CumulativeSum, pd.Series)
def execute_series_cumsum(op, data, **kwargs):
    return data.cumsum()


@execute_node.register(ops.CumulativeMin, pd.Series)
def execute_series_cummin(op, data, **kwargs):
    return data.cummin()


@execute_node.register(ops.CumulativeMax, pd.Series)
def execute_series_cummax(op, data, **kwargs):
    return data.cummax()


@execute_node.register(ops.CumulativeOp, pd.Series)
def execute_series_cumulative_op(op, data, **kwargs):
    typename = type(op).__name__
    match = re.match(r'^Cumulative([A-Za-z_][A-Za-z0-9_]*)$', typename)
    if match is None:
        raise ValueError('Unknown operation {}'.format(typename))

    try:
        operation_name, = match.groups()
    except ValueError:
        raise ValueError(
            'More than one operation name found in {} class'.format(typename)
        )
    return agg_ctx.Cumulative(
        dtype=op.to_expr().type().to_pandas(),
    ).agg(data, operation_name.lower())


def post_lead_lag(result, default):
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(
    (ops.Lead, ops.Lag),
    (pd.Series, SeriesGroupBy),
    integer_types + (type(None),),
    simple_types + (type(None),),
)
def execute_series_lead_lag(op, data, offset, default, **kwargs):
    func = toolz.identity if isinstance(op, ops.Lag) else operator.neg
    result = data.shift(func(1 if offset is None else offset))
    return post_lead_lag(result, default)


@execute_node.register(
    (ops.Lead, ops.Lag),
    (pd.Series, SeriesGroupBy),
    timedelta_types,
    date_types + timestamp_types + six.string_types + (type(None),),
)
def execute_series_lead_lag_timedelta(
    op, data, offset, default, aggcontext=None, **kwargs
):
    """An implementation of shifting a column relative to another one that is
    in units of time rather than rows.
    """
    # lagging adds time (delayed), leading subtracts time (moved up)
    func = operator.add if isinstance(op, ops.Lag) else operator.sub
    group_by = aggcontext.group_by
    order_by = aggcontext.order_by

    # get the parent object from which `data` originated
    parent = aggcontext.parent

    # get the DataFrame from the parent object, handling the DataFrameGroupBy
    # case
    parent_df = getattr(parent, 'obj', parent)

    # index our parent df by grouping and ordering keys
    indexed_original_df = parent_df.set_index(group_by + order_by)

    # perform the time shift
    adjusted_parent_df = parent_df.assign(
        **{k: func(parent_df[k], offset) for k in order_by})

    # index the parent *after* adjustment
    adjusted_indexed_parent = adjusted_parent_df.set_index(group_by + order_by)

    # get the column we care about
    result = adjusted_indexed_parent[getattr(data, 'obj', data).name]

    # reindex the shifted data by the original frame's index
    result = result.reindex(indexed_original_df.index)

    # add a default if necessary
    return post_lead_lag(result, default)


@execute_node.register(ops.FirstValue, pd.Series)
def execute_series_first_value(op, data, **kwargs):
    return data.values[0]


@execute_node.register(ops.FirstValue, SeriesGroupBy)
def execute_series_group_by_first_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, 'first')


@execute_node.register(ops.LastValue, pd.Series)
def execute_series_last_value(op, data, **kwargs):
    return data.values[-1]


@execute_node.register(ops.LastValue, SeriesGroupBy)
def execute_series_group_by_last_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, 'last')


@execute_node.register(ops.MinRank, (pd.Series, SeriesGroupBy))
def execute_series_min_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='min', ascending=True)


@execute_node.register(ops.DenseRank, (pd.Series, SeriesGroupBy))
def execute_series_dense_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='dense', ascending=True)


@execute_node.register(ops.PercentRank, (pd.Series, SeriesGroupBy))
def execute_series_percent_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='min', ascending=True, pct=True)
