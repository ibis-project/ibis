"""Code for computing window functions in the dask backend."""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import TimeContext
from ibis.backends.dask.core import execute, execute_with_scope
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import (
    _pandas_dtype_from_dd_scalar,
    _wrap_dd_scalar,
    add_partitioned_sorted_column,
    make_meta_series,
)


def _post_process_empty(
    result: Any,
    parent: dd.Series | dd.DataFrame,
    timecontext: TimeContext | None,
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
        return parent.apply(lambda row: result, meta=(None, 'object'))


@execute_node.register(ops.Window, dd.Series)
def execute_window_frame(
    op,
    data,
    scope: Scope,
    timecontext: TimeContext | None = None,
    aggcontext=None,
    clients=None,
    **kwargs,
):
    # Currently this acts as an "unwrapper" for trivial windows (i.e. those
    # with no ordering/grouping/preceding/following functionality).
    if not all([op.frame.start is None, op.frame.end is None, not op.frame.order_by]):
        raise NotImplementedError(
            "Window operations are unsupported in the dask backend"
        )

    if op.frame.group_by:
        # there's lots of complicated logic that only applies to grouped
        # windows
        return execute_grouped_window_op(
            op,
            data,
            scope,
            timecontext,
            aggcontext,
            clients,
            **kwargs,
        )

    result = execute_with_scope(
        op.func,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    return _post_process_empty(result, data, timecontext)


def execute_grouped_window_op(
    op,
    data,
    scope,
    timecontext,
    aggcontext,
    clients,
    **kwargs,
):
    # extract the parent
    root = an.find_first_base_table(op)

    root_data = execute(
        root,
        scope=scope,
        timecontext=timecontext,
        clients=clients,
        aggcontext=aggcontext,
        **kwargs,
    )

    grouping_keys = [key.name for key in op.frame.group_by]

    grouped_root_data = root_data.groupby(grouping_keys)
    scope = scope.merge_scopes(
        [
            Scope({t: grouped_root_data}, timecontext)
            for t in an.find_immediate_parent_tables(op.func)
        ],
        overwrite=True,
    )

    result = execute_with_scope(
        op.func,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    # If the grouped operation we performed is not an analytic UDF we have to
    # realign the output to the input.
    if not isinstance(op.func, ops.AnalyticVectorizedUDF):
        result = dd.merge(
            root_data[result.index.name].to_frame(),
            result.to_frame(),
            left_on=result.index.name,
            right_index=True,
        )[result.name]
        result.divisions = root_data.divisions

    return result
