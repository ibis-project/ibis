import operator
from typing import Any, Collection

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.expr.operations as ops

from ..dispatch import execute_node


@execute_node.register(ops.ArrayColumn, list)
def execute_array_column(op, cols, **kwargs):
    df = pd.concat(cols, axis=1)
    return df.apply(lambda row: np.array(row, dtype=object), axis=1)


@execute_node.register(ops.ArrayLength, pd.Series)
def execute_array_length(op, data, **kwargs):
    return data.apply(len)


@execute_node.register(ops.ArrayLength, np.ndarray)
def execute_array_length_scalar(op, data, **kwargs):
    return len(data)


@execute_node.register(ops.ArraySlice, pd.Series, int, (int, type(None)))
def execute_array_slice(op, data, start, stop, **kwargs):
    return data.apply(operator.itemgetter(slice(start, stop)))


@execute_node.register(ops.ArraySlice, np.ndarray, int, (int, type(None)))
def execute_array_slice_scalar(op, data, start, stop, **kwargs):
    return data[start:stop]


@execute_node.register(ops.ArrayIndex, pd.Series, int)
def execute_array_index(op, data, index, **kwargs):
    return data.apply(
        lambda array, index=index: (
            array[index] if -len(array) <= index < len(array) else None
        )
    )


@execute_node.register(ops.ArrayIndex, np.ndarray, int)
def execute_array_index_scalar(op, data, index, **kwargs):
    try:
        return data[index]
    except IndexError:
        return None


def _concat_iterables_to_series(
    iter1: Collection[Any],
    iter2: Collection[Any],
) -> pd.Series:
    """Concatenate two collections elementwise ("horizontally") to create a
    Series. The two collections are assumed to have the same length.

    Used for ArrayConcat implementation.
    """
    assert len(iter1) == len(iter2)
    # Doing the iteration using `map` is much faster than doing the iteration
    # using `Series.apply` due to Pandas-related overhead.
    result = pd.Series(map(lambda x, y: np.concatenate([x, y]), iter1, iter2))
    return result


@execute_node.register(ops.ArrayConcat, pd.Series, pd.Series)
def execute_array_concat_series(op, left, right, **kwargs):
    return _concat_iterables_to_series(left, right)


@execute_node.register(ops.ArrayConcat, pd.Series, np.ndarray)
@execute_node.register(ops.ArrayConcat, np.ndarray, pd.Series)
def execute_array_concat_mixed(op, left, right, **kwargs):
    # ArrayConcat given a column (pd.Series) and a scalar (np.ndarray).
    # We will broadcast the scalar to the length of the column.
    if isinstance(left, np.ndarray):
        # Broadcast `left` to the length of `right`
        left = np.tile(left, (len(right), 1))
    elif isinstance(right, np.ndarray):
        # Broadcast `right` to the length of `left`
        right = np.tile(right, (len(left), 1))
    return _concat_iterables_to_series(left, right)


@execute_node.register(ops.ArrayConcat, np.ndarray, np.ndarray)
def execute_array_concat_scalar(op, left, right, **kwargs):
    return np.concatenate([left, right])


@execute_node.register(ops.ArrayRepeat, pd.Series, int)
def execute_array_repeat(op, data, n, **kwargs):
    # Negative n will be treated as 0 (repeat will produce empty array)
    n = max(n, 0)
    return pd.Series(
        map(
            lambda arr: np.tile(arr, n),
            data,
        )
    )


@execute_node.register(ops.ArrayRepeat, np.ndarray, int)
def execute_array_repeat_scalar(op, data, n, **kwargs):
    # Negative n will be treated as 0 (repeat will produce empty array)
    return np.tile(data, max(n, 0))


@execute_node.register(ops.ArrayCollect, (pd.Series, SeriesGroupBy))
def execute_array_collect(op, data, aggcontext=None, **kwargs):
    return aggcontext.agg(data, np.array)
