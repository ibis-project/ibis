"""Code for computing window functions in the dask backend."""

from typing import Any, Optional, Union

import dask.dataframe as dd

import ibis.expr.operations as ops
import ibis.expr.window as win
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..core import execute_with_scope
from ..dispatch import execute_node
from .util import (
    _pandas_dtype_from_dd_scalar,
    _wrap_dd_scalar,
    add_partitioned_sorted_column,
    make_meta_series,
)


def _post_process_empty(
    result: Any,
    parent: Union[dd.Series, dd.DataFrame],
    timecontext: Optional[TimeContext],
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


@execute_node.register(ops.WindowOp, dd.Series, win.Window)
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
    # Currently this acts as an "unwrapper" for trivial windows (i.e. those
    # with no ordering/grouping/preceding/following functionality).
    if not all(
        [
            window.preceding is None,
            window.following is None,
            window._group_by == [],
            window._order_by == [],
        ]
    ):
        raise NotImplementedError(
            "Window operations are unsuported in the dask backend"
        )
    result = execute_with_scope(
        expr=op.expr,
        scope=scope,
        timecontext=timecontext,
        aggcontext=aggcontext,
        clients=clients,
        **kwargs,
    )
    return _post_process_empty(result, data, timecontext)
