"""Code for computing window functions in the dask backend."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, NoReturn

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import pandas as pd
import toolz
from multipledispatch import Dispatcher

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.dask import aggcontext as agg_ctx
from ibis.backends.dask.core import compute_time_context, execute
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    _pandas_dtype_from_dd_scalar,
    _wrap_dd_scalar,
    add_globally_consecutive_column,
    compute_sorted_frame,
    make_meta_series,
)
from ibis.backends.pandas.core import (
    date_types,
    integer_types,
    simple_types,
    timedelta_types,
    timestamp_types,
)
from ibis.backends.pandas.execution.window import _post_process_group_by_order_by

if TYPE_CHECKING:
    from ibis.backends.base.df.timecontext import (
        TimeContext,
    )
    from ibis.backends.pandas.aggcontext import AggregationContext


def _check_valid_window_frame(frame):
    # TODO consolidate this with pandas
    if frame.how == "range" and any(
        not col.dtype.is_temporal() for col in frame.order_by
    ):
        raise NotImplementedError(
            "The Dask backend only implements range windows with temporal "
            "ordering keys"
        )

    if len(frame.order_by) > 1:
        raise NotImplementedError(
            "Multiple order_bys are not supported in the dask backend"
        )

    if frame.order_by and frame.group_by:
        raise NotImplementedError(
            "Grouped and order windows are not supported in the dask backend."
        )


def _get_post_process_function(frame: ops.WindowFrame) -> Callable:
    # TODO consolidate with pandas
    if frame.group_by:
        if frame.order_by:
            return _post_process_group_by_order_by
        else:
            return _post_process_group_by
    elif frame.order_by:
        return _post_process_order_by
    else:
        return _post_process_empty


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


# TODO consolidate with pandas
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
            window=len(parent),
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
        )

    return aggcontext


def _post_process_empty(
    result: Any,
    parent: dd.Series | dd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
    **kwargs,
) -> dd.Series:
    """Post process non grouped, non ordered windows.

    dd.Series/dd.DataFrame objects are passed through, otherwise we conform
    the output to the parent input (i.e. so the shape an partitioning matches).

    dd.core.Scalar needs special handling so downstream functions can work
    with it.
    """
    if isinstance(result, (dd.Series, dd.DataFrame)):
        return result
    elif isinstance(result, dd.core.Scalar):
        # TODO this should be refactored with similar logic in util.py
        # both solve the generalish problem we have of wrapping a
        # dd.core.Scalar into something dask can work with downstream
        # TODO computation
        lens = parent.index.map_partitions(len).compute().values
        out_dtype = _pandas_dtype_from_dd_scalar(result)
        meta = make_meta_series(dtype=out_dtype)
        delayeds = [_wrap_dd_scalar(result, None, out_len) for out_len in lens]
        series = dd.from_delayed(delayeds, meta=meta)
        series = add_globally_consecutive_column(series)
        return series[0]
    else:
        # Project any non delayed object to the shape of "parent"
        return parent.apply(
            lambda row, result=result: result, meta=(None, "object"), axis=1
        )


def _post_process_order_by(
    series,
    parent: dd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
    **kwargs,
) -> dd.Series:
    """Functions like pandas with dasky argsorting."""
    assert order_by and not group_by
    if isinstance(series, dd.core.Scalar):
        lens = parent.index.map_partitions(len).compute().values
        out_dtype = _pandas_dtype_from_dd_scalar(series)
        meta = make_meta_series(dtype=out_dtype)
        delayeds = [_wrap_dd_scalar(series, None, out_len) for out_len in lens]
        series = dd.from_delayed(delayeds, meta=meta)
        series = add_globally_consecutive_column(series)
        return series[0]

    series_index_name = "index" if series.index.name is None else series.index.name
    # Need to sort series back before returning.
    series = series.reset_index().set_index(series_index_name).iloc[:, 0]

    return series


def _post_process_group_by(
    series,
    parent: dd.DataFrame,
    order_by: list[str],
    group_by: list[str],
    timecontext: TimeContext | None,
    op,
    **kwargs,
) -> dd.Series:
    assert not order_by and group_by
    # FIXME This is likely not needed anymore.
    return series


@execute_node.register(ops.WindowFunction, dd.Series)
def execute_window_op(
    op,
    data,
    scope: Scope,
    timecontext: TimeContext | None = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    func, frame = op.func, op.frame
    _check_valid_window_frame(frame)

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

    root_table = an.find_first_base_table(op)
    root_data = execute(
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

    if frame.group_by:
        if frame.order_by:
            raise NotImplementedError("Grouped and order windows not supported yet")
            # TODO finish implementing grouped/order windows.
        else:
            if len(grouping_keys) == 1 and isinstance(grouping_keys[0], dd.Series):
                # Dask will raise an exception about not supporting multiple Series in group by key
                # even if it is passed a length 1 list of Series.
                # For this case we just make group_by_cols a single Series.
                group_by_cols = grouping_keys[0]
            else:
                group_by_cols = grouping_keys
            source = root_data.groupby(group_by_cols, sort=False, group_keys=False)
    elif frame.order_by:
        source, grouping_keys, ordering_keys = compute_sorted_frame(
            df=root_data,
            order_by=frame.order_by,
            timecontext=timecontext,
            **kwargs,
        )
    else:
        source = root_data

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

    result = _get_post_process_function(frame)(
        result,
        root_data,
        ordering_keys,
        grouping_keys,
        timecontext,
        op=op,
    )

    # If the grouped operation we performed is not an analytic UDF we may need
    # to realign the output to the input.
    if (
        not isinstance(op.func, ops.AnalyticVectorizedUDF)
        and not result.known_divisions
    ):
        if root_data.index.name != result.index.name:
            result = dd.merge(
                root_data[result.index.name].to_frame(),
                result.to_frame(),
                left_on=result.index.name,
                right_index=True,
            )[result.name]

        result.divisions = root_data.divisions

    return result


@execute_node.register(
    (ops.Lead, ops.Lag),
    (dd.Series, ddgb.SeriesGroupBy),
    integer_types + (type(None),),
    simple_types + (type(None),),
)
def execute_series_lead_lag(op, data, offset, default, **kwargs):
    func = toolz.identity if isinstance(op, ops.Lag) else operator.neg
    result = data.shift(func(1 if offset is None else offset))
    return post_lead_lag(result, default)


@execute_node.register(
    (ops.Lead, ops.Lag),
    (dd.Series, ddgb.SeriesGroupBy),
    timedelta_types,
    date_types + timestamp_types + (str, type(None)),
)
def execute_series_lead_lag_timedelta(
    op, data, offset, default, aggcontext=None, **kwargs
):
    """Shift a column relative to another one that is in units of time rather than rows."""
    # lagging adds time (delayed), leading subtracts time (moved up)
    func = operator.add if isinstance(op, ops.Lag) else operator.sub
    group_by = aggcontext.group_by
    order_by = aggcontext.order_by

    # get the parent object from which `data` originated
    parent = aggcontext.parent

    # get the DataFrame from the parent object, handling the DataFrameGroupBy
    # case
    parent_df = getattr(parent, "obj", parent)

    # perform the time shift
    adjusted_parent_df = parent_df.assign(
        **{k: func(parent_df[k], offset) for k in order_by}
    )

    # index the parent *after* adjustment
    adjusted_indexed_parent = adjusted_parent_df.set_index(group_by + order_by)

    # get the column we care about
    result = adjusted_indexed_parent[getattr(data, "obj", data).name]

    # add a default if necessary
    return post_lead_lag(result, default)


def post_lead_lag(result, default):
    if not pd.isnull(default):
        return result.fillna(default)
    return result


@execute_node.register(ops.FirstValue, dd.Series)
def execute_series_first_value(op, data, **kwargs):
    arr = data.head(1, compute=False).values
    # normally you shouldn't do this but we know that there is one row
    # consider upstreaming to dask
    arr._chunks = ((1,),)
    return arr[0]


@execute_node.register(ops.FirstValue, ddgb.SeriesGroupBy)
def execute_series_group_by_first_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "first")


@execute_node.register(ops.LastValue, dd.Series)
def execute_series_last_value(op, data, **kwargs):
    arr = data.tail(1, compute=False).values

    # normally you shouldn't do this but we know that there is one row
    # consider upstreaming to dask
    arr._chunks = ((1,),)
    return arr[0]


@execute_node.register(ops.LastValue, ddgb.SeriesGroupBy)
def execute_series_group_by_last_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "last")
