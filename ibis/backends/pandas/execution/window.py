"""Code for computing window functions with ibis and pandas."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, NoReturn

import numpy as np
import pandas as pd
import toolz
from multipledispatch import Dispatcher
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import (
    TimeContext,
    construct_time_context_aware_series,
    get_time_col,
)
from ibis.backends.pandas import aggcontext as agg_ctx
from ibis.backends.pandas.core import (
    compute_time_context,
    date_types,
    execute,
    integer_types,
    simple_types,
    timedelta_types,
    timestamp_types,
)
from ibis.backends.pandas.dispatch import execute_node, pre_execute
from ibis.backends.pandas.execution import util

if TYPE_CHECKING:
    from ibis.backends.pandas.aggcontext import AggregationContext


def _post_process_empty(
    result: Any,
    parent: pd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
) -> pd.Series:
    # This is the post process of the no groupby nor orderby window
    # `result` could be a Series, DataFrame, or a scalar. generated
    # by `agg` method of class `Window`. For window without grouby or
    # orderby, `agg` calls pands method directly. So if timecontext is
    # present, we need to insert 'time' column into index for trimming the
    # result. For cases when grouby or orderby is present, `agg` calls
    # Ibis method `window_agg_built_in` and `window_agg_udf`, time
    # context is already inserted there.
    assert not order_by and not group_by
    if isinstance(result, (pd.Series, pd.DataFrame)):
        if timecontext:
            result = construct_time_context_aware_series(result, parent)
        return result
    else:
        # `result` is a scalar when a reduction operation is being
        # applied over the window, since reduction operations are N->1
        # in this case we do not need to trim result by timecontext,
        # just expand reduction result to be a Series with `index`.
        index = parent.index
        result = pd.Series([result]).repeat(len(index))
        result.index = index
        return result


def _post_process_group_by(
    series: pd.Series,
    parent: pd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
) -> pd.Series:
    assert not order_by and group_by
    return series


def _post_process_order_by(
    series,
    parent: pd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
) -> pd.Series:
    assert order_by and not group_by
    indexed_parent = parent.set_index(order_by)
    index = indexed_parent.index

    # get the names of the levels that will be in the result
    series_index_names = frozenset(series.index.names)

    # get the levels common to series.index, in the order that they occur in
    # the parent's index
    reordered_levels = [name for name in index.names if name in series_index_names]

    if len(reordered_levels) > 1:
        series = series.reorder_levels(reordered_levels)

    series = series.iloc[index.argsort(kind="mergesort")]
    return series


def _post_process_group_by_order_by(
    series: pd.Series,
    parent: pd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
) -> pd.Series:
    indexed_parent = parent.set_index(group_by + order_by, append=True)
    index = indexed_parent.index

    # get the names of the levels that will be in the result
    series_index_names = frozenset(series.index.names)

    # get the levels common to series.index, in the order that they occur in
    # the parent's index
    reordered_levels = [name for name in index.names if name in series_index_names]

    if len(reordered_levels) > 1:
        series = series.reorder_levels(reordered_levels)
    return series


get_aggcontext = Dispatcher("get_aggcontext")


@get_aggcontext.register(object)
def get_aggcontext_default(
    window,
    *,
    scope,
    operand,
    parent,
    group_by,
    order_by,
    **kwargs,
) -> NoReturn:
    raise NotImplementedError(
        f"get_aggcontext is not implemented for {type(window).__name__}"
    )


@get_aggcontext.register(ops.WindowFrame)
def get_aggcontext_window(
    frame,
    *,
    scope,
    operand,
    parent,
    group_by,
    order_by,
    **kwargs,
) -> AggregationContext:
    # no order by or group by: default summarization aggcontext
    #
    # if we're reducing and we have an order by expression then we need to
    # expand or roll.
    #
    # otherwise we're transforming
    output_type = operand.dtype

    if not group_by and not order_by:
        aggcontext = agg_ctx.Summarize(parent=parent, output_type=output_type)
    elif group_by and not order_by:
        # groupby transform (window with a partition by clause in SQL parlance)
        aggcontext = agg_ctx.Transform(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
        )
    elif frame.start is not None:
        if isinstance(frame, ops.RowsWindowFrame):
            max_lookback = frame.max_lookback
        else:
            max_lookback = None

        aggcontext = agg_ctx.Moving(
            frame.start,
            # FIXME(kszucs): I don't think that we have a proper max_lookback test
            # case because passing None here is not braking anything
            max_lookback=max_lookback,
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
        )
    else:
        # expanding window
        aggcontext = agg_ctx.Cumulative(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
        )

    return aggcontext


def trim_window_result(data: pd.Series | pd.DataFrame, timecontext: TimeContext | None):
    """Trim data within time range defined by timecontext.

    This is a util function used in ``execute_window_op``, where time
    context might be adjusted for calculation. Data must be trimmed
    within the original time context before return.
    `data` is a pd.Series with Multiindex for most cases, for multi
    column udf result, `data` could be a pd.DataFrame

    Params
    ------
    data: pd.Series or pd.DataFrame
    timecontext: Optional[TimeContext]

    Returns
    -------
    a trimmed pd.Series or or pd.DataFrame with the same Multiindex
    as data's
    """
    # noop if timecontext is None
    if not timecontext:
        return data
    assert isinstance(
        data, (pd.Series, pd.DataFrame)
    ), "window computed columns is not a pd.Series nor a pd.DataFrame"

    # reset multiindex, convert Series into a DataFrame
    df = data.reset_index()

    # Filter the data, here we preserve the time index so that when user is
    # computing a single column, the computation and the relevant time
    # indexes are returned.
    time_col = get_time_col()
    if time_col not in df:
        return data

    subset = df.loc[df[time_col].between(*timecontext)]

    # Get columns to set for index
    if isinstance(data, pd.Series):
        # if Series doesn't contain a name, reset_index will assign
        # '0' as the column name for the column of value
        name = data.name if data.name else 0
        index_columns = list(subset.columns.difference([name]))
    else:
        name = data.columns
        index_columns = list(subset.columns.difference(name))

    # set the correct index for return Series / DataFrame
    indexed_subset = subset.set_index(index_columns)
    return indexed_subset[name]


@execute_node.register(ops.WindowFunction, [pd.Series])
def execute_window_op(
    op,
    *data,
    scope: Scope | None = None,
    timecontext: TimeContext | None = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    func, frame = op.func, op.frame

    if frame.how == "range" and any(
        not col.dtype.is_temporal() for col in frame.order_by
    ):
        raise NotImplementedError(
            "The pandas backend only implements range windows with temporal "
            "ordering keys"
        )

    # pre execute "manually" here because otherwise we wouldn't pickup
    # relevant scope changes from the child operand since we're managing
    # execution of that by hand

    adjusted_timecontext = None
    if timecontext:
        arg_timecontexts = compute_time_context(
            op, timecontext=timecontext, clients=clients, scope=scope
        )
        # timecontext is the original time context required by parent node
        # of this Window, while adjusted_timecontext is the adjusted context
        # of this Window, since we are doing a manual execution here, use
        # adjusted_timecontext in later execution phases
        adjusted_timecontext = arg_timecontexts[0]

    pre_executed_scope = pre_execute(
        func,
        *clients,
        scope=scope,
        timecontext=adjusted_timecontext,
        aggcontext=aggcontext,
        **kwargs,
    )
    if scope is None:
        scope = pre_executed_scope
    else:
        scope = scope.merge_scope(pre_executed_scope)

    root_table = an.find_first_base_table(op)
    data = execute(
        root_table,
        scope=scope,
        timecontext=adjusted_timecontext,
        clients=clients,
        aggcontext=aggcontext,
        **kwargs,
    )

    grouping_keys = [
        key.name
        if isinstance(key, ops.TableColumn)
        else execute(
            key,
            scope=scope,
            clients=clients,
            timecontext=adjusted_timecontext,
            aggcontext=aggcontext,
            **kwargs,
        )
        for key in frame.group_by
    ]

    if not frame.order_by:
        ordering_keys = []

    post_process: Callable[
        [Any, pd.DataFrame, list[str], list[str], TimeContext | None],
        pd.Series,
    ]
    if frame.group_by:
        if frame.order_by:
            sorted_df, grouping_keys, ordering_keys = util.compute_sorted_frame(
                data,
                frame.order_by,
                group_by=frame.group_by,
                timecontext=adjusted_timecontext,
                **kwargs,
            )
            source = sorted_df.groupby(grouping_keys, sort=True, group_keys=False)
            post_process = _post_process_group_by_order_by
        else:
            source = data.groupby(grouping_keys, sort=False, group_keys=False)
            post_process = _post_process_group_by
    elif frame.order_by:
        source, grouping_keys, ordering_keys = util.compute_sorted_frame(
            data, frame.order_by, timecontext=adjusted_timecontext, **kwargs
        )
        post_process = _post_process_order_by
    else:
        source = data
        post_process = _post_process_empty

    # Here groupby object should be add to the corresponding node in scope
    # for execution, data will be overwrite to a groupby object, so we
    # force an update regardless of time context
    new_scope = scope.merge_scopes(
        [
            Scope({t: source}, adjusted_timecontext)
            for t in an.find_immediate_parent_tables(func)
        ],
        overwrite=True,
    )

    aggcontext = get_aggcontext(
        frame,
        scope=scope,
        operand=func,
        parent=source,
        group_by=grouping_keys,
        order_by=ordering_keys,
        **kwargs,
    )
    result = execute(
        func,
        scope=new_scope,
        timecontext=adjusted_timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    result = post_process(
        result,
        data,
        ordering_keys,
        grouping_keys,
        adjusted_timecontext,
    )
    assert len(data) == len(
        result
    ), "input data source and computed column do not have the same length"

    # trim data to original time context
    result = trim_window_result(result, timecontext)
    return result


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
    date_types + timestamp_types + (str, type(None)),
)
def execute_series_lead_lag_timedelta(
    op, data, offset, default, aggcontext=None, **kwargs
):
    """Shift a column relative to another one in units of time instead of rows."""
    # lagging adds time (delayed), leading subtracts time (moved up)
    func = operator.add if isinstance(op, ops.Lag) else operator.sub
    group_by = aggcontext.group_by
    order_by = aggcontext.order_by

    # get the parent object from which `data` originated
    parent = aggcontext.parent

    # get the DataFrame from the parent object, handling the DataFrameGroupBy
    # case
    parent_df = getattr(parent, "obj", parent)

    # index our parent df by grouping and ordering keys
    indexed_original_df = parent_df.set_index(group_by + order_by)

    # perform the time shift
    adjusted_parent_df = parent_df.assign(
        **{k: func(parent_df[k], offset) for k in order_by}
    )

    # index the parent *after* adjustment
    adjusted_indexed_parent = adjusted_parent_df.set_index(group_by + order_by)

    # get the column we care about
    result = adjusted_indexed_parent[getattr(data, "obj", data).name]

    # reindex the shifted data by the original frame's index
    result = result.reindex(indexed_original_df.index)

    # add a default if necessary
    return post_lead_lag(result, default)


@execute_node.register(ops.FirstValue, pd.Series)
def execute_series_first_value(op, data, **kwargs):
    return data.iloc[np.repeat(0, len(data))]


def _getter(x: pd.Series | np.ndarray, idx: int):
    return getattr(x, "values", x)[idx]


@execute_node.register(ops.FirstValue, SeriesGroupBy)
def execute_series_group_by_first_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, lambda x: _getter(x, 0))


@execute_node.register(ops.LastValue, pd.Series)
def execute_series_last_value(op, data, **kwargs):
    return data.iloc[np.repeat(-1, len(data))]


@execute_node.register(ops.LastValue, SeriesGroupBy)
def execute_series_group_by_last_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, lambda x: _getter(x, -1))


@execute_node.register(ops.MinRank)
def execute_series_min_rank(op, aggcontext=None, **kwargs):
    (key,) = aggcontext.order_by
    df = aggcontext.parent
    data = df[key]
    return data.rank(method="min", ascending=True).astype("int64") - 1


@execute_node.register(ops.DenseRank)
def execute_series_dense_rank(op, aggcontext=None, **kwargs):
    (key,) = aggcontext.order_by
    df = aggcontext.parent
    data = df[key]
    return data.rank(method="dense", ascending=True).astype("int64") - 1


@execute_node.register(ops.PercentRank)
def execute_series_group_by_percent_rank(op, aggcontext=None, **kwargs):
    (key,) = aggcontext.order_by
    df = aggcontext.parent
    data = df[key]

    result = data.rank(method="min", ascending=True) - 1

    if isinstance(data, SeriesGroupBy):
        nrows = data.transform("count")
    else:
        nrows = len(data)

    result /= nrows - 1
    return result


@execute_node.register(ops.CumeDist)
def execute_series_group_by_cume_dist(op, aggcontext=None, **kwargs):
    (key,) = aggcontext.order_by
    df = aggcontext.parent
    data = df[key]
    return data.rank(method="min", ascending=True, pct=True)
