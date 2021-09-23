"""Code for computing window functions with ibis and pandas."""

import operator
import re
from typing import Any, Callable, List, NoReturn, Optional, Union

import pandas as pd
import toolz
from multipledispatch import Dispatcher
from pandas.core.groupby import SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.window as win
from ibis.expr.scope import Scope
from ibis.expr.timecontext import (
    construct_time_context_aware_series,
    get_time_col,
)
from ibis.expr.typing import TimeContext

from .. import aggcontext as agg_ctx
from ..aggcontext import AggregationContext
from ..core import (
    compute_time_context,
    date_types,
    execute,
    integer_types,
    simple_types,
    timedelta_types,
    timestamp_types,
)
from ..dispatch import execute_node, pre_execute
from ..execution import util


def _post_process_empty(
    result: Any,
    parent: pd.DataFrame,
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
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
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
) -> pd.Series:
    assert not order_by and group_by
    return series


def _post_process_order_by(
    series,
    parent: pd.DataFrame,
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
) -> pd.Series:
    assert order_by and not group_by
    indexed_parent = parent.set_index(order_by)
    index = indexed_parent.index
    names = index.names
    if len(names) > 1:
        series = series.reorder_levels(names)
    series = series.iloc[index.argsort(kind='mergesort')]
    return series


def _post_process_group_by_order_by(
    series: pd.Series,
    parent: pd.DataFrame,
    order_by: List[str],
    group_by: List[str],
    timecontext: Optional[TimeContext],
) -> pd.Series:
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


get_aggcontext = Dispatcher('get_aggcontext')


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


@get_aggcontext.register(win.Window)
def get_aggcontext_window(
    window,
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
    output_type = operand.type()

    aggcontext: agg_ctx.AggregationContext
    if not group_by and not order_by:
        aggcontext = agg_ctx.Summarize(parent=parent, output_type=output_type)
    elif (
        isinstance(
            operand.op(), (ops.Reduction, ops.CumulativeOp, ops.Any, ops.All)
        )
        and order_by
    ):
        # XXX(phillipc): What a horror show
        preceding = window.preceding
        if preceding is not None:
            max_lookback = window.max_lookback
            assert not isinstance(operand.op(), ops.CumulativeOp)
            aggcontext = agg_ctx.Moving(
                preceding,
                max_lookback,
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
    else:
        # groupby transform (window with a partition by clause in SQL parlance)
        aggcontext = agg_ctx.Transform(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
        )

    return aggcontext


def trim_window_result(
    data: Union[pd.Series, pd.DataFrame], timecontext: Optional[TimeContext]
):
    """Trim data within time range defined by timecontext

    This is a util function used in ``execute_window_op``, where time
    context might be adjusted for calculation. Data must be trimmed
    within the original time context before return.
    `data` is a pd.Series with Multiindex for most cases, for multi
    column udf result, `data` could be a pd.DataFrame

    Params
    ------
    data: pd.Series or pd.DataFrame
    timecontext: Optional[TimeContext]

    Returns:
    ------
    a trimmed pd.Series or or pd.DataFrame with the same Multiindex
    as data's

    """
    # noop if timecontext is None
    if not timecontext:
        return data
    assert isinstance(
        data, (pd.Series, pd.DataFrame)
    ), 'window computed columns is not a pd.Series nor a pd.DataFrame'

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
        # if Series dosen't contain a name, reset_index will assign
        # '0' as the column name for the column of value
        name = data.name if data.name else 0
        index_columns = list(subset.columns.difference([name]))
    else:
        name = data.columns
        index_columns = list(subset.columns.difference(name))

    # set the correct index for return Series / DataFrame
    indexed_subset = subset.set_index(index_columns)
    return indexed_subset[name]


@execute_node.register(ops.WindowOp, pd.Series, win.Window)
def execute_window_op(
    op,
    data,
    window,
    scope: Scope = None,
    timecontext: Optional[TimeContext] = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    operand = op.expr
    # pre execute "manually" here because otherwise we wouldn't pickup
    # relevant scope changes from the child operand since we're managing
    # execution of that by hand
    operand_op = operand.op()

    adjusted_timecontext = None
    if timecontext:
        arg_timecontexts = compute_time_context(
            op, timecontext=timecontext, clients=clients
        )
        # timecontext is the original time context required by parent node
        # of this WindowOp, while adjusted_timecontext is the adjusted context
        # of this Window, since we are doing a manual execution here, use
        # adjusted_timecontext in later execution phases
        adjusted_timecontext = arg_timecontexts[0]

    pre_executed_scope = pre_execute(
        operand_op,
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
    (root,) = op.root_tables()
    root_expr = root.to_expr()

    data = execute(
        root_expr,
        scope=scope,
        timecontext=adjusted_timecontext,
        clients=clients,
        aggcontext=aggcontext,
        **kwargs,
    )
    following = window.following
    order_by = window._order_by

    if (
        order_by
        and following != 0
        and not isinstance(operand_op, ops.ShiftBase)
    ):
        raise com.OperationNotDefinedError(
            'Window functions affected by following with order_by are not '
            'implemented'
        )

    group_by = window._group_by
    grouping_keys = [
        key_op.name
        if isinstance(key_op, ops.TableColumn)
        else execute(
            key,
            scope=scope,
            clients=clients,
            timecontext=adjusted_timecontext,
            aggcontext=aggcontext,
            **kwargs,
        )
        for key, key_op in zip(
            group_by, map(operator.methodcaller('op'), group_by)
        )
    ]

    order_by = window._order_by
    if not order_by:
        ordering_keys = []

    post_process: Callable[
        [Any, pd.DataFrame, List[str], List[str], Optional[TimeContext]],
        pd.Series,
    ]
    if group_by:
        if order_by:
            (
                sorted_df,
                grouping_keys,
                ordering_keys,
            ) = util.compute_sorted_frame(
                data,
                order_by,
                group_by=group_by,
                timecontext=adjusted_timecontext,
                **kwargs,
            )
            source = sorted_df.groupby(grouping_keys, sort=True)
            post_process = _post_process_group_by_order_by
        else:
            source = data.groupby(grouping_keys, sort=False)
            post_process = _post_process_group_by
    else:
        if order_by:
            source, grouping_keys, ordering_keys = util.compute_sorted_frame(
                data, order_by, timecontext=adjusted_timecontext, **kwargs
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
            for t in operand.op().root_tables()
        ],
        overwrite=True,
    )

    aggcontext = get_aggcontext(
        window,
        scope=scope,
        operand=operand,
        parent=source,
        group_by=grouping_keys,
        order_by=ordering_keys,
        **kwargs,
    )
    result = execute(
        operand,
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
    ), 'input data source and computed column do not have the same length'

    # trim data to original time context
    result = trim_window_result(result, timecontext)
    return result


@execute_node.register(
    (ops.CumulativeSum, ops.CumulativeMax, ops.CumulativeMin),
    (pd.Series, SeriesGroupBy),
)
def execute_series_cumulative_sum_min_max(op, data, **kwargs):
    typename = type(op).__name__
    method_name = (
        re.match(r"^Cumulative([A-Za-z_][A-Za-z0-9_]*)$", typename)
        .group(1)
        .lower()
    )
    method = getattr(data, f"cum{method_name}")
    return method()


@execute_node.register(ops.CumulativeMean, (pd.Series, SeriesGroupBy))
def execute_series_cumulative_mean(op, data, **kwargs):
    # TODO: Doesn't handle the case where we've grouped/sorted by. Handling
    # this here would probably require a refactor.
    return data.expanding().mean()


@execute_node.register(ops.CumulativeOp, (pd.Series, SeriesGroupBy))
def execute_series_cumulative_op(op, data, aggcontext=None, **kwargs):
    assert aggcontext is not None, "aggcontext is none in {} operation".format(
        type(op)
    )
    typename = type(op).__name__
    match = re.match(r'^Cumulative([A-Za-z_][A-Za-z0-9_]*)$', typename)
    if match is None:
        raise ValueError(f'Unknown operation {typename}')

    try:
        (operation_name,) = match.groups()
    except ValueError:
        raise ValueError(
            f'More than one operation name found in {typename} class'
        )

    dtype = op.to_expr().type().to_pandas()
    assert isinstance(aggcontext, agg_ctx.Cumulative), f'Got {type()}'
    result = aggcontext.agg(data, operation_name.lower())

    # all expanding window operations are required to be int64 or float64, so
    # we need to cast back to preserve the type of the operation
    try:
        return result.astype(dtype)
    except TypeError:
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
        **{k: func(parent_df[k], offset) for k in order_by}
    )

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
    return data.rank(method='min', ascending=True).astype('int64') - 1


@execute_node.register(ops.DenseRank, (pd.Series, SeriesGroupBy))
def execute_series_dense_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='dense', ascending=True).astype('int64') - 1


@execute_node.register(ops.PercentRank, (pd.Series, SeriesGroupBy))
def execute_series_percent_rank(op, data, **kwargs):
    # TODO(phillipc): Handle ORDER BY
    return data.rank(method='min', ascending=True, pct=True)
