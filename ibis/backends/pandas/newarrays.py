from __future__ import annotations

import itertools
import operator

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.common.exceptions import OperationNotDefinedError


@execute.register(ops.ArrayLength)
def execute_array_length(op, arg):
    if isinstance(arg, pd.Series):
        return arg.apply(len)
    else:
        return len(arg)


@execute.register(ops.ArrayIndex)
def execute_array_index(op, arg, index):
    if isinstance(arg, pd.Series):
        return arg.apply(
            lambda array, index=index: (
                array[index] if -len(array) <= index < len(array) else None
            )
        )
    else:
        try:
            return arg[index]
        except IndexError:
            return None


@execute.register(ops.ArrayRepeat)
def execute_array_repeat(op, arg, times):
    if isinstance(times, pd.Series):
        raise OperationNotDefinedError("ArrayRepeat with Series times")

    if isinstance(arg, pd.Series):
        return arg.apply(lambda array, times=times: np.tile(array, times))
    else:
        return np.tile(arg, times)


@execute.register(ops.ArraySlice)
def execute_array_slice(op, arg, start, stop):
    if isinstance(start, pd.Series) or isinstance(stop, pd.Series):
        raise OperationNotDefinedError("ArraySlice with Series start or stop")

    if isinstance(arg, pd.Series):
        return arg.apply(operator.itemgetter(slice(start, stop)))
    else:
        return arg[start:stop]


@execute.register(ops.ArrayColumn)
def execute_array_column(op, cols):
    df = pd.concat(cols, axis=1)
    return df.apply(lambda row: np.array(row, dtype=object), axis=1)


@execute.register(ops.ArrayFlatten)
def execute_array_flatten(op, arg):
    return arg.map(lambda v: list(itertools.chain.from_iterable(v)), na_action="ignore")


# @execute.register(ops.Unnest)
# def execute_unnest(op, arg):
#     return arg.explode()


# @execute_node.register(ops.ArrayContains, np.ndarray, object)
# def execute_node_contains_value_array(op, haystack, needle, **kwargs):
#     return needle in haystack


# def _concat_iterables_to_series(*iters: Collection[Any]) -> pd.Series:
#     """Concatenate two collections to create a Series.

#     The two collections are assumed to have the same length.

#     Used for ArrayConcat implementation.
#     """
#     first, *rest = iters
#     assert all(len(series) == len(first) for series in rest)
#     # Doing the iteration using `map` is much faster than doing the iteration
#     # using `Series.apply` due to Pandas-related overhead.
#     return pd.Series(map(lambda *args: np.concatenate(args), first, *rest))


# @execute_node.register(ops.ArrayConcat, tuple)
# def execute_array_concat(op, args, **kwargs):
#     return execute_node(op, *map(partial(execute, **kwargs), args), **kwargs)


# @execute_node.register(ops.ArrayConcat, pd.Series, pd.Series, [pd.Series])
# def execute_array_concat_series(op, first, second, *args, **kwargs):
#     return _concat_iterables_to_series(first, second, *args)


# @execute_node.register(
#     ops.ArrayConcat, np.ndarray, pd.Series, [(pd.Series, np.ndarray)]
# )
# def execute_array_concat_mixed_left(op, left, right, *args, **kwargs):
#     # ArrayConcat given a column (pd.Series) and a scalar (np.ndarray).
#     # We will broadcast the scalar to the length of the column.
#     # Broadcast `left` to the length of `right`
#     left = np.tile(left, (len(right), 1))
#     return _concat_iterables_to_series(left, right)


# @execute_node.register(
#     ops.ArrayConcat, pd.Series, np.ndarray, [(pd.Series, np.ndarray)]
# )
# def execute_array_concat_mixed_right(op, left, right, *args, **kwargs):
#     # Broadcast `right` to the length of `left`
#     right = np.tile(right, (len(left), 1))
#     return _concat_iterables_to_series(left, right)


# @execute_node.register(ops.ArrayConcat, np.ndarray, np.ndarray, [np.ndarray])
# def execute_array_concat_scalar(op, left, right, *args, **kwargs):
#     return np.concatenate([left, right, *args])


# @execute_node.register(ops.ArrayCollect, pd.Series, (type(None), pd.Series))
# def execute_array_collect(op, data, where, aggcontext=None, **kwargs):
#     return aggcontext.agg(data.loc[where] if where is not None else data, np.array)


# @execute_node.register(ops.ArrayCollect, SeriesGroupBy, (type(None), pd.Series))
# def execute_array_collect_groupby(op, data, where, aggcontext=None, **kwargs):
#     return aggcontext.agg(
#         (
#             data.obj.loc[where].groupby(data.grouping.grouper)
#             if where is not None
#             else data
#         ),
#         np.array,
#     )
