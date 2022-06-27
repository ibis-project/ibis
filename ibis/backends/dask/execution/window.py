"""Code for computing window functions in the dask backend."""

import operator
from typing import Any, Callable, List, Optional, Union

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import pandas
import toolz

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.window as win
from ibis.backends.dask.core import (
    compute_time_context,
    execute,
    execute_with_scope,
)
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    _pandas_dtype_from_dd_scalar,
    _wrap_dd_scalar,
    add_partitioned_sorted_column,
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
from ibis.backends.pandas.execution.window import (
    _post_process_group_by_order_by,
    get_aggcontext,
)
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext


def _check_valid_window(window: win.Window, operand_op):
    # TODO consolidate this with pandas
    if window.how == "range" and any(
        not isinstance(ob.type(), (dt.Time, dt.Date, dt.Timestamp))
        for ob in window._order_by
    ):
        raise NotImplementedError(
            "The pandas backend only implements range windows with temporal "
            "ordering keys"
        )

    if (
        window._order_by
        and window.following != 0
        and not isinstance(operand_op, ops.ShiftBase)
    ):
        raise com.OperationNotDefinedError(
            'Window functions affected by following with order_by are not '
            'implemented'
        )

    if len(window._order_by) > 1:
        raise NotImplementedError(
            "Multiple order_bys are not supported in the dask backend"
        )

    if window._order_by and window._group_by:
        raise NotImplementedError(
            "Grouped and order windows are not supported in the dask backend."
        )


def _get_post_process_function(window: win.Window) -> Callable:
    # TODO consolidate with pandas
    if window._group_by:
        if window._order_by:
            return _post_process_group_by_order_by
        else:
            return _post_process_group_by
    else:
        if window._order_by:
            return _post_process_order_by
        else:
            return _post_process_empty


def _post_process_empty(
    result: Any,
    parent: Union[dd.Series, dd.DataFrame],
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
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
        series = add_partitioned_sorted_column(series)
        return series[0]
    else:
        # Project any non delayed object to the shape of "parent"
        if isinstance(parent, dd.DataFrame):
            parent = parent[parent.columns[0]]

        return parent.apply(
            lambda row, result=result: result, meta=(None, 'object')
        )


def _post_process_order_by(
    series,
    parent: dd.DataFrame,
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
    **kwargs,
) -> dd.Series:
    """Functions like pandas with dasky argsorting"""
    assert order_by and not group_by
    return series


def _post_process_group_by(
    series,
    parent: dd.DataFrame,
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
    op,
    **kwargs,
) -> dd.Series:
    if not isinstance(op.expr._arg, ops.AnalyticVectorizedUDF):
        series = dd.merge(
            parent[series.index.name].to_frame(),
            series.to_frame(),
            left_on=series.index.name,
            right_index=True,
        )[series.name]
        series.divisions = parent.divisions

    return series


@execute_node.register(ops.Window, dd.Series, win.Window)
def execute_window_op(
    op,
    data,
    window,
    scope: Scope,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    _check_valid_window(window, op.expr.op())

    # extract the parent
    (root,) = op.root_tables()
    root_expr = root.to_expr()

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

    root_data = execute(
        root_expr,
        scope=scope,
        timecontext=timecontext,
        clients=clients,
        aggcontext=aggcontext,
        **kwargs,
    )

    if not (window._order_by or window._group_by):
        source = root_data
    if window._order_by:
        source, order_by_col = compute_sorted_frame(
            root_data, window._order_by[0], timecontext
        )
        order_by_col = [order_by_col]
    else:
        order_by_col = []

    if window._group_by:
        group_by_cols = get_grouping_keys(window._group_by)
        source = root_data.groupby(group_by_cols)
    else:
        group_by_cols = []

    # breakpoint()
    scope = scope.merge_scopes(
        [
            Scope({t: source}, adjusted_timecontext)
            for t in op.expr.op().root_tables()
        ],
        overwrite=True,
    )
    if not window._group_by:
        aggcontext = get_aggcontext(
            window,
            scope=scope,
            operand=op.expr,
            parent=source,
            group_by=group_by_cols,
            order_by=order_by_col,
            **kwargs,
        )

    result = execute_with_scope(
        expr=op.expr,
        scope=scope,
        timecontext=adjusted_timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )

    return _get_post_process_function(window)(
        result,
        root_data,
        order_by_col,
        group_by_cols,
        timecontext,
        op=op,
    )


def get_grouping_keys(group_by):
    return [
        key_op.name for key_op in map(operator.methodcaller('op'), group_by)
    ]


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

    # perform the time shift
    adjusted_parent_df = parent_df.assign(
        **{k: func(parent_df[k], offset) for k in order_by}
    )

    # index the parent *after* adjustment
    adjusted_indexed_parent = adjusted_parent_df.set_index(group_by + order_by)

    # get the column we care about
    result = adjusted_indexed_parent[getattr(data, 'obj', data).name]

    # add a default if necessary
    return post_lead_lag(result, default)


def post_lead_lag(result, default):
    if not pandas.isnull(default):
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
    return aggcontext.agg(data, 'first')


@execute_node.register(ops.LastValue, dd.Series)
def execute_series_last_value(op, data, **kwargs):
    arr = data.tail(1, compute=False).values

    # normally you shouldn't do this but we know that there is one row
    # consider upstreaming to dask
    arr._chunks = ((1,),)
    return arr[0]


@execute_node.register(ops.LastValue, ddgb.SeriesGroupBy)
def execute_series_group_by_last_value(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, 'last')
