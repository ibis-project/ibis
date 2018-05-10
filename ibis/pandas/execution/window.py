"""Code for computing window functions with ibis.
"""

import operator
import re

from collections import OrderedDict

import toolz

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.common as com
import ibis.expr.window as win
import ibis.expr.operations as ops

import ibis.pandas.aggcontext as agg_ctx

from ibis.pandas.core import integer_types
from ibis.pandas.dispatch import execute_node
from ibis.pandas.core import execute
from ibis.pandas.execution import util


def _post_process_empty(scalar, index):
    return pd.Series([scalar], index=index)


def _post_process_group_by(series, index):
    return series


def _post_process_order_by(series, index):
    return series.reindex(index)


def _post_process_group_by_order_by(series, index):
    level_list = list(range(series.index.nlevels - 1))
    series_with_reset_index = series.reset_index(level=level_list, drop=True)
    reindexed_series = series_with_reset_index.reindex(index)
    return reindexed_series


@execute_node.register(ops.WindowOp, pd.Series, win.Window)
def execute_window_op(op, data, window, scope=None, context=None, **kwargs):
    operand = op.expr
    root, = op.root_tables()
    try:
        data = scope[root]
    except KeyError:
        data = execute(root.to_expr(), scope=scope, context=context, **kwargs)

    following = window.following
    order_by = window._order_by

    if order_by and following != 0:
        raise com.OperationNotDefinedError(
            'Following with a value other than 0 (current row) with order_by '
            'is not yet implemented in the pandas backend. Use '
            'ibis.trailing_window or ibis.cumulative_window to '
            'construct windows when using the pandas backend.'
        )

    group_by = window._group_by
    grouping_keys = [
        key_op.name if isinstance(key_op, ops.TableColumn) else execute(
            key,
            context=context,
            **kwargs
        ) for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by

    if grouping_keys:
        source = data.groupby(grouping_keys, sort=False, as_index=not order_by)

        if order_by:
            sorted_df = source.apply(
                lambda df, order_by=order_by, kwargs=kwargs: (
                    util.compute_sorted_frame(order_by, df, **kwargs)
                )
            )
            source = sorted_df.groupby(grouping_keys, sort=False)
            post_process = _post_process_group_by_order_by
        else:
            post_process = _post_process_group_by
    else:
        if order_by:
            source = util.compute_sorted_frame(order_by, data, **kwargs)
            post_process = _post_process_order_by
        else:
            source = data
            post_process = _post_process_empty

    new_scope = toolz.merge(
        scope,
        OrderedDict((t, source) for t in operand.op().root_tables()),
        factory=OrderedDict,
    )

    # no order by or group by: default summarization context
    #
    # if we're reducing and we have an order by expression then we need to
    # expand or roll.
    #
    # otherwise we're transforming
    if not grouping_keys and not order_by:
        context = agg_ctx.Summarize()
    elif isinstance(operand.op(), ops.Reduction) and order_by:
        # XXX(phillipc): What a horror show
        preceding = window.preceding
        if preceding is not None:
            context = agg_ctx.Moving(preceding)
        else:
            # expanding window
            context = agg_ctx.Cumulative()
    else:
        # groupby transform (window with a partition by clause in SQL parlance)
        context = agg_ctx.Transform()

    result = execute(operand, new_scope, context=context, **kwargs)
    series = post_process(result, data.index)
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
    return agg_ctx.Cumulative().agg(data, operation_name.lower())


@execute_node.register(
    ops.Lag, (pd.Series, SeriesGroupBy),
    integer_types + (type(None),),
    object,
)
def execute_series_lag(op, data, offset, default, **kwargs):
    result = data.shift(1 if offset is None else offset)
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(
    ops.Lead, (pd.Series, SeriesGroupBy),
    integer_types + (type(None),),
    object,
)
def execute_series_lead(op, data, offset, default, **kwargs):
    result = data.shift(-(1 if offset is None else offset))
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(
    ops.Lag, (pd.Series, SeriesGroupBy), pd.Timedelta, object,
)
def execute_series_lag_timedelta(op, data, offset, default, **kwargs):
    result = data.tshift(freq=offset)
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(
    ops.Lead, (pd.Series, SeriesGroupBy), pd.Timedelta, object
)
def execute_series_lead_timedelta(op, data, offset, default, **kwargs):
    result = data.tshift(freq=-offset)
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(ops.FirstValue, pd.Series)
def execute_series_first_value(op, data, **kwargs):
    return data.iloc[0]


@execute_node.register(ops.FirstValue, SeriesGroupBy)
def execute_series_group_by_first_value(op, data, context=None, **kwargs):
    return context.agg(data, 'first')


@execute_node.register(ops.LastValue, pd.Series)
def execute_series_last_value(op, data, **kwargs):
    return data.iloc[-1]


@execute_node.register(ops.LastValue, SeriesGroupBy)
def execute_series_group_by_last_value(op, data, context=None, **kwargs):
    return context.agg(data, 'last')


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
