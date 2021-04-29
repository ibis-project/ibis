import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.arrays import (
    execute_array_index,
    execute_array_length,
    execute_array_slice,
)

from ..dispatch import execute_node
from .util import TypeRegistrationDict, register_types_to_dispatcher

DASK_DISPATCH_TYPES: TypeRegistrationDict = {
    ops.ArrayLength: [((dd.Series,), execute_array_length)],
    ops.ArrayIndex: [((dd.Series, int), execute_array_index)],
    ops.ArraySlice: [
        ((dd.Series, int, (int, type(None))), execute_array_slice),
    ],
}

register_types_to_dispatcher(execute_node, DASK_DISPATCH_TYPES)


collect_array = dd.Aggregation(
    name="collect_array",
    chunk=lambda s: s.apply(np.array),
    agg=lambda s0: s0.apply(np.concatenate),
)


@execute_node.register(ops.ArrayColumn, list)
def execute_array_column(op, cols, **kwargs):
    df = dd.concat(cols, axis=1)
    return df.apply(
        lambda row: np.array(row, dtype=object), axis=1, meta=(None, 'object')
    )


@execute_node.register(ops.ArrayConcat, dd.Series, dd.Series)
def execute_array_concat_series(op, left, right, **kwargs):
    df = dd.concat([left, right], axis=1)
    return df.apply(lambda row: np.concatenate(row), axis=1, meta=left._meta,)


@execute_node.register(ops.ArrayConcat, dd.Series, np.ndarray)
@execute_node.register(ops.ArrayConcat, np.ndarray, dd.Series)
def execute_array_concat_mixed(op, left, right, **kwargs):
    if isinstance(left, dd.Series):
        # "Iterate" over `left` Series, concatenating each array/element
        # in the Series with `right`, which is a single ndarray.
        return left.apply(
            lambda arr: np.concatenate([arr, right]), meta=left._meta,
        )
    elif isinstance(right, dd.Series):
        return right.apply(
            lambda arr: np.concatenate([left, arr]), meta=right._meta,
        )


@execute_node.register(ops.ArrayConcat, np.ndarray, np.ndarray)
def execute_array_concat_scalar(op, left, right, **kwargs):
    return np.concatenate([left, right])


@execute_node.register(ops.ArrayRepeat, dd.Series, int)
def execute_array_repeat(op, data, n, **kwargs):
    # Negative n will be treated as 0 (repeat will produce empty array)
    n = max(n, 0)
    return data.apply(lambda arr: np.repeat(arr, n), meta=data._meta)


@execute_node.register(ops.ArrayRepeat, np.ndarray, int)
def execute_array_repeat_scalar(op, data, n, **kwargs):
    # Negative n will be treated as 0 (repeat will produce empty array)
    return np.repeat(data, max(n, 0))


# TODO - aggregations - #2553
# Currently, Summarize aggcontext with non-string `function` does not
# work on Dask backend, because it tries to call `dd.Series.agg`, which
# does not exist.
@execute_node.register(ops.ArrayCollect, dd.Series)
def execute_array_collect(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, np.array)


@execute_node.register(ops.ArrayCollect, ddgb.SeriesGroupBy)
def execute_array_collect_grouped_series(op, data, aggcontext=None, **kwargs):
    return data.agg(collect_array)
